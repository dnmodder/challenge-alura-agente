from __future__ import annotations

import csv
import os
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

load_dotenv()

CLAVE_API = os.getenv("GEMINI_API_KEY")
MODELO_EMBEDDING = "gemini-embedding-2"
MODELO_LLM = "gemini-3.5-flash-lite"
TOP_K = 3
TAMANO_CHUNK = 1000


@dataclass
class Documento:
    ruta_archivo: Path
    contenido: str


@dataclass
class Chunk:
    id_documento: str
    indice: int
    contenido: str


def extraer_texto_de_archivo(archivo: Path) -> str:
    ext = archivo.suffix.lower()
    if ext in {".md", ".txt"}:
        return archivo.read_text(encoding="utf-8")

    if ext == ".pdf":
        reader = pypdf.PdfReader(archivo)
        paginas = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(paginas)

    if ext == ".docx":
        doc = docx.Document(archivo)
        parrafos = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(parrafos)

    if ext == ".csv":
        df = pd.read_csv(archivo)
        return df.to_string(index=False)

    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(archivo)
        return df.to_string(index=False)

    return ""


def cargar_documentos(directorio_datos: Path = Path("datos")) -> list[Documento]:
    documentos: list[Documento] = []
    if not directorio_datos.exists():
        return documentos

    extensiones_validas = {".md", ".txt", ".pdf", ".docx", ".csv", ".xlsx", ".xls"}
    for archivo in directorio_datos.glob("*"):
        if archivo.suffix.lower() in extensiones_validas:
            try:
                contenido = extraer_texto_de_archivo(archivo)
                if contenido.strip():
                    documentos.append(
                        Documento(ruta_archivo=archivo, contenido=contenido)
                    )
            except Exception as error:
                print(f"Error al leer el archivo {archivo.name}: {error}")
    return documentos


def crear_chunks(documentos: list[Documento]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documentos:
        texto = doc.contenido
        nombre_doc = doc.ruta_archivo.stem.replace("_", " ").title()
        lineas = texto.split("\n")

        encabezado_actual = nombre_doc
        buffer_texto = []
        posicion = 0

        for linea in lineas:
            if linea.startswith("#"):
                encabezado_actual = f"{nombre_doc} > {linea.lstrip('#').strip()}"
            buffer_texto.append(linea)
            bloque = "\n".join(buffer_texto)

            if len(bloque) >= TAMANO_CHUNK:
                contenido_chunk = f"[{encabezado_actual}]\n{bloque}"
                chunks.append(
                    Chunk(
                        id_documento=doc.ruta_archivo.name,
                        indice=posicion,
                        contenido=contenido_chunk,
                    )
                )
                posicion += 1
                buffer_texto = buffer_texto[-3:]

        if buffer_texto:
            bloque = "\n".join(buffer_texto)
            contenido_chunk = f"[{encabezado_actual}]\n{bloque}"
            chunks.append(
                Chunk(
                    id_documento=doc.ruta_archivo.name,
                    indice=posicion,
                    contenido=contenido_chunk,
                )
            )

    return chunks


class BuscadorVectorialFAISS:
    def __init__(self, cliente_gemini: genai.Client) -> None:
        self.cliente = cliente_gemini
        self.chunks: list[Chunk] = []
        self.indice_faiss: faiss.IndexFlatL2 | None = None
        self.dimension = 3072

    def obtener_embedding(self, texto: str) -> np.ndarray:
        respuesta = self.cliente.models.embed_content(
            model=MODELO_EMBEDDING,
            contents=texto,
        )
        if not respuesta.embeddings or len(respuesta.embeddings) == 0:
            raise ValueError(
                f"No se pudieron generar embeddings para el texto provisto: {texto[:50]}..."
            )
        vector = np.array(respuesta.embeddings[0].values, dtype=np.float32)
        return vector

    def indexar_chunks(self, lista_chunks: list[Chunk]) -> None:
        if not lista_chunks:
            return
        self.chunks = lista_chunks

        embeddings_list = []
        for chk in lista_chunks:
            v = self.obtener_embedding(chk.contenido)
            embeddings_list.append(v)

        matriz_embeddings = np.vstack(embeddings_list).astype(np.float32)
        self.dimension = matriz_embeddings.shape[1]

        self.indice_faiss = faiss.IndexFlatL2(self.dimension)
        self.indice_faiss.add(matriz_embeddings)

    def buscar_similares(self, consulta: str, top_k: int = TOP_K) -> list[Chunk]:
        if self.indice_faiss is None or len(self.chunks) == 0:
            return []

        vector_consulta = self.obtener_embedding(consulta).reshape(1, -1)
        _, indices = self.indice_faiss.search(vector_consulta, top_k)

        resultados: list[Chunk] = []

        for i in indices[0]:
            if 0 <= i < len(self.chunks):
                resultados.append(self.chunks[i])
        return resultados


class AgenteRAG:
    def __init__(self) -> None:
        if not CLAVE_API:
            raise ValueError(
                "No se encontró la variable GEMINI_API_KEY en el archivo .env"
            )
        self.cliente = genai.Client(api_key=CLAVE_API)
        self.buscador = BuscadorVectorialFAISS(self.cliente)
        self.inicializar_conocimientos()

    def inicializar_conocimientos(self) -> None:
        docs = cargar_documentos()
        chunks = crear_chunks(docs)
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
    agente = AgenteRAG()
    demo = construir_interfaz()
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
