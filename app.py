from __future__ import annotations

import os
import re
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import docx
import faiss
import gradio as gr
import numpy as np
import pandas as pd
import pypdf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

load_dotenv()

CLAVE_API = os.getenv("GEMINI_API_KEY")
TOKEN_HF = os.getenv("HF_TOKEN")
MODELO_EMBEDDING_LOCAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_LLM = "gemini-3.5-flash-lite"
TOP_K = 3


@dataclass
class Documento:
    ruta_archivo: Path
    contenido: str


@dataclass
class Chunk:
    id_documento: str
    indice: int
    contenido: str


def procesar_csv(archivo: Path) -> list[Chunk]:
    chunks_csv: list[Chunk] = []
    nombre_doc = archivo.stem.replace("_", " ").title()

    try:
        df = (
            pd.read_csv(
                archivo,
                sep=None,
                engine="python",
                encoding="utf-8-sig",
                on_bad_lines="skip",
            )
            .dropna(how="all", axis=0)
            .dropna(how="all", axis=1)
        )
    except (OSError, ValueError, RuntimeError, pd.errors.EmptyDataError) as error:
        print(f"⚠️ Error al leer el archivo CSV {archivo.name}: {error}")
        return chunks_csv

    tamanio_bloque_filas = 20
    registros = df.to_dict(orient="records")
    total_registros = len(registros)

    for inicio in range(0, total_registros, tamanio_bloque_filas):
        fin = min(inicio + tamanio_bloque_filas, total_registros)
        sub_registros = registros[inicio:fin]

        lineas_bloque = []
        for reg in sub_registros:
            detalles = ", ".join(
                [
                    f"{col}: {val}"
                    for col, val in reg.items()
                    if pd.notna(val) and str(val).strip() != ""
                ]
            )
            if detalles:
                lineas_bloque.append(f"• {detalles}")

        if lineas_bloque:
            contenido = (
                f"[{nombre_doc} (Registros {inicio + 1}-{fin} de {total_registros})]\n"
                + "\n".join(lineas_bloque)
            )
            chunks_csv.append(
                Chunk(
                    id_documento=archivo.name,
                    indice=len(chunks_csv),
                    contenido=contenido,
                )
            )

    return chunks_csv


def extraer_texto_de_archivo(archivo: Path) -> str:
    ext = archivo.suffix.lower()
    if ext in {".md", ".txt"}:
        return archivo.read_text(encoding="utf-8")

    if ext == ".pdf":
        reader = pypdf.PdfReader(archivo)
        paginas = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(paginas)

    if ext in {".docx", ".doc"}:
        doc = docx.Document(str(archivo))
        parrafos = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(parrafos)

    return ""


def cargar_documentos(
    directorio_datos: Path = Path("datos"),
) -> tuple[list[Documento], list[Chunk]]:
    documentos: list[Documento] = []
    chunks_directos: list[Chunk] = []

    if not directorio_datos.exists():
        return documentos, chunks_directos

    extensiones_validas = {".md", ".txt", ".pdf", ".docx", ".doc", ".csv"}
    for archivo in directorio_datos.glob("*"):
        ext = archivo.suffix.lower()
        if ext in extensiones_validas:
            try:
                if ext == ".csv":
                    chunks_csv = procesar_csv(archivo)
                    chunks_directos.extend(chunks_csv)
                else:
                    contenido = extraer_texto_de_archivo(archivo)
                    if contenido.strip():
                        documentos.append(
                            Documento(ruta_archivo=archivo, contenido=contenido)
                        )
            except (
                OSError,
                ValueError,
                RuntimeError,
                pd.errors.EmptyDataError,
            ) as error:
                print(f"Error al leer el archivo {archivo.name}: {error}")

    return documentos, chunks_directos


def crear_chunks(documentos: list[Documento]) -> list[Chunk]:
    chunks: list[Chunk] = []
    tamano_bloque = 1500
    solapamiento = 200

    for doc in documentos:
        texto = doc.contenido.strip()
        if not texto:
            continue

        nombre_doc = doc.ruta_archivo.stem.replace("_", " ").title()

        if len(texto) <= tamano_bloque:
            chunks.append(
                Chunk(
                    id_documento=doc.ruta_archivo.name,
                    indice=0,
                    contenido=f"[{nombre_doc}]\n{texto}",
                )
            )
            continue

        posicion = 0
        inicio = 0
        largo_total = len(texto)

        while inicio < largo_total:
            fin = min(inicio + tamano_bloque, largo_total)
            fragmento = texto[inicio:fin].strip()

            if fragmento:
                contenido_chunk = f"[{nombre_doc}]\n{fragmento}"
                chunks.append(
                    Chunk(
                        id_documento=doc.ruta_archivo.name,
                        indice=posicion,
                        contenido=contenido_chunk,
                    )
                )
                posicion += 1

            if fin >= largo_total:
                break

            inicio += tamano_bloque - solapamiento

    return chunks


import unicodedata


def normalizar_texto(texto: str) -> str:

    texto_nfkd = unicodedata.normalize("NFKD", texto)
    return "".join([c for c in texto_nfkd if not unicodedata.combining(c)]).lower()


class BuscadorVectorialFAISS:
    def __init__(self) -> None:
        print(
            "🧠 Cargando modelo de embeddings 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'..."
        )
        token_hf = TOKEN_HF if TOKEN_HF else None
        self.modelo_embedding = SentenceTransformer(
            MODELO_EMBEDDING_LOCAL,
            token=token_hf,
        )
        self.chunks: list[Chunk] = []
        self.indice_faiss: faiss.IndexFlatL2 | None = None
        self.dimension = self.modelo_embedding.get_embedding_dimension()

    def obtener_embedding(self, texto: str) -> np.ndarray:
        vector = self.modelo_embedding.encode(
            texto, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.array(vector, dtype=np.float32)

    def indexar_chunks(self, lista_chunks: list[Chunk]) -> None:
        if not lista_chunks:
            return
        self.chunks = lista_chunks

        print(
            f"⚡ Indizando {len(lista_chunks)} fragmentos de texto en la base vectorial FAISS..."
        )
        textos = [chk.contenido for chk in lista_chunks]
        matriz_embeddings = self.modelo_embedding.encode(
            textos,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        faiss.normalize_L2(matriz_embeddings)

        self.dimension = matriz_embeddings.shape[1]
        self.indice_faiss = faiss.IndexFlatL2(self.dimension)
        self.indice_faiss.add(matriz_embeddings)
        print("✅ Base vectorial FAISS indizada con éxito.")

    def buscar_similares(self, consulta: str, top_k: int = TOP_K) -> list[Chunk]:
        if self.indice_faiss is None or len(self.chunks) == 0:
            return []

        vector_consulta = self.obtener_embedding(consulta).reshape(1, -1)
        faiss.normalize_L2(vector_consulta)
        _, indices = self.indice_faiss.search(vector_consulta, top_k * 3)

        resultados_vectoriales: list[Chunk] = []
        for i in indices[0]:
            if 0 <= i < len(self.chunks):
                resultados_vectoriales.append(self.chunks[i])

        palabras_clave = [normalizar_texto(p) for p in consulta.split() if len(p) >= 3]
        coincidencias_exactas: list[Chunk] = []

        if palabras_clave:
            for chk in self.chunks:
                contenido_norm = normalizar_texto(chk.contenido)
                matches = 0
                for p in palabras_clave:
                    pattern = r"\b" + re.escape(p)
                    if re.search(pattern, contenido_norm):
                        matches += 1

                if (
                    matches == len(palabras_clave)
                    and chk not in coincidencias_exactas
                    and chk not in resultados_vectoriales
                ):
                    coincidencias_exactas.append(chk)

        combinados = coincidencias_exactas + resultados_vectoriales
        return combinados[:top_k]


class AgenteRAG:
    def __init__(self) -> None:
        if not CLAVE_API:
            raise ValueError(
                "No se encontró la variable GEMINI_API_KEY en el archivo .env"
            )
        self.cliente = genai.Client(api_key=CLAVE_API)
        self.buscador = BuscadorVectorialFAISS()
        self.inicializar_conocimientos()

    def inicializar_conocimientos(self) -> None:
        docs, chunks_csv = cargar_documentos()
        chunks = crear_chunks(docs) + chunks_csv
        self.buscador.indexar_chunks(chunks)

    def generar_respuesta_stream(
        self, mensaje_usuario: str, historial: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        chunks_relacionados = self.buscador.buscar_similares(mensaje_usuario)
        contexto_recuperado = "\n\n---\n\n".join(
            [c.contenido for c in chunks_relacionados]
        )

        prompt_sistema = f"""Eres el Agente de Información Empresarial interno.
Responde de forma clara, directa, precisa y estructurada en español a los empleados y colaboradores usando ÚNICAMENTE el contexto de información provisto a continuación.

Reglas obligatorias:
- Si el mensaje es un saludo inicial y no hay historial previo, saluda cordialmente. Si ya hay mensajes previos en el historial, responde directamente sin repetir saludos redundantes.
- No menciones nombres de archivos técnicos ni rutas internas de sistema.
- Si la información no está presente en el contexto, indica amablemente que no dispones de esa información en los documentos empresariales.

CONTEXTO DE INFORMACIÓN RECUPERADO:
{contexto_recuperado}
"""

        mensajes_chat = []
        for h in historial:
            if isinstance(h, dict):
                rol = "user" if h.get("role") == "user" else "model"
                contenido = h.get("content", "")
                if contenido:
                    mensajes_chat.append(
                        types.Content(
                            role=rol,
                            parts=[types.Part.from_text(text=str(contenido))],
                        )
                    )

        mensajes_chat.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=mensaje_usuario)],
            )
        )

        configuracion = types.GenerateContentConfig(
            system_instruction=prompt_sistema,
            temperature=0.2,
        )

        respuesta_stream = self.cliente.models.generate_content_stream(
            model=MODELO_LLM,
            contents=mensajes_chat,
            config=configuracion,
        )

        texto_acumulado = ""
        for chunk in respuesta_stream:
            if chunk.text:
                texto_acumulado += chunk.text
                yield texto_acumulado


def responder_gradio(
    mensaje: str, historial: list[dict[str, str]]
) -> Generator[str, None, None]:
    global agente
    if agente is None:
        agente = AgenteRAG()
    yield from agente.generar_respuesta_stream(mensaje, historial)


agente: AgenteRAG | None = None


def construir_interfaz() -> gr.Blocks:
    demo = gr.ChatInterface(
        fn=responder_gradio,
        title="💼 Agente de Información Empresarial",
        description="Asistente inteligente interno para consultas sobre políticas, procedimientos y documentación corporativa.",
    )
    return demo


def main() -> None:
    global agente
    print("🚀 Inicializando Agente RAG de Información Empresarial...")
    agente = AgenteRAG()
    demo = construir_interfaz()
    print("🌐 Lanzando servidor web Gradio en http://127.0.0.1:7860 ...")
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
