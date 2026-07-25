import io
import os
import re
import sys
import unicodedata
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docx
import faiss
import gradio as gr
import numpy as np
import pandas as pd
import pypdf
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

CLAVE_API = os.getenv("GEMINI_API_KEY")
MODELO_LLM = "gemini-3.5-flash-lite"
MODELO_EMBEDDING_ONNX = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


# Almacenamiento global de DataFrames cargados desde datos/
DATAFRAMES_DISPONIBLES: dict[str, pd.DataFrame] = {}


@dataclass
class Documento:
    ruta_archivo: Path
    contenido: str


@dataclass
class Chunk:
    id_documento: str
    indice: int
    contenido: str


def normalizar_texto(texto: str) -> str:
    texto_nfkd = unicodedata.normalize("NFKD", texto)
    return "".join([c for c in texto_nfkd if not unicodedata.combining(c)]).lower()


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


def cargar_tablas(directorio_datos: Path = Path("datos")) -> None:
    DATAFRAMES_DISPONIBLES.clear()

    if not directorio_datos.exists():
        return

    for archivo in directorio_datos.glob("*"):
        ext = archivo.suffix.lower()
        if ext == ".csv":
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
                nombre_clave = archivo.stem.replace(" ", "_")
                DATAFRAMES_DISPONIBLES[nombre_clave] = df
                print(
                    f"📊 Tabla CSV cargada: '{nombre_clave}' ({len(df)} filas, {len(df.columns)} columnas)"
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                pd.errors.EmptyDataError,
            ) as error:
                print(f"⚠️ Error al cargar CSV {archivo.name}: {error}")

        elif ext in {".xlsx", ".xls"}:
            try:
                hojas = pd.read_excel(str(archivo), sheet_name=None)
                for nombre_hoja, df_hoja in hojas.items():
                    df_limpio = df_hoja.dropna(how="all", axis=0).dropna(
                        how="all", axis=1
                    )
                    if not df_limpio.empty:
                        nombre_clave = f"{archivo.stem}_{nombre_hoja}".replace(" ", "_")
                        DATAFRAMES_DISPONIBLES[nombre_clave] = df_limpio
                        print(
                            f"📊 Hoja Excel cargada: '{nombre_clave}' ({len(df_limpio)} filas, {len(df_limpio.columns)} columnas)"
                        )
            except (
                OSError,
                ValueError,
                RuntimeError,
                pd.errors.EmptyDataError,
            ) as error:
                print(f"⚠️ Error al cargar Excel {archivo.name}: {error}")


def cargar_documentos_texto(
    directorio_datos: Path = Path("datos"),
) -> list[Documento]:
    documentos: list[Documento] = []
    if not directorio_datos.exists():
        return documentos

    extensiones_validas = {".md", ".txt", ".pdf", ".docx", ".doc"}
    for archivo in directorio_datos.glob("*"):
        ext = archivo.suffix.lower()
        if ext in extensiones_validas:
            try:
                contenido = extraer_texto_de_archivo(archivo)
                if contenido.strip():
                    documentos.append(
                        Documento(ruta_archivo=archivo, contenido=contenido)
                    )
            except (OSError, ValueError, RuntimeError) as error:
                print(f"⚠️ Error al leer documento de texto {archivo.name}: {error}")

    return documentos


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


class BuscadorVectorialFAISS:
    def __init__(self) -> None:
        print("🧠 Inicializando motor de embeddings ONNX (fastembed)...")
        self.modelo_embedding = TextEmbedding(MODELO_EMBEDDING_ONNX)
        self.chunks: list[Chunk] = []
        self.indice_faiss: faiss.IndexFlatL2 | None = None
        self.dimension = 384

    def obtener_embedding(self, texto: str) -> np.ndarray:
        generator = self.modelo_embedding.embed([texto])
        vector = next(iter(generator))
        arr = np.array(vector, dtype=np.float32)
        faiss.normalize_L2(arr.reshape(1, -1))
        return arr


    def indexar_chunks(self, lista_chunks: list[Chunk]) -> None:
        if not lista_chunks:
            return
        self.chunks = lista_chunks

        print(
            f"⚡ Indizando {len(lista_chunks)} fragmentos de texto en FAISS con ONNX..."
        )
        textos = [chk.contenido for chk in lista_chunks]
        generator = self.modelo_embedding.embed(textos)
        embeddings_list = list(generator)
        matriz_embeddings = np.array(embeddings_list, dtype=np.float32)

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


# Instancia global del buscador FAISS
BUSCADOR_FAISS: BuscadorVectorialFAISS | None = None


@tool
def consultar_documentos_texto(consulta: str) -> str:
    """Consulta la base de conocimientos RAG sobre documentos de texto (PDF, DOCX, MD, TXT).
    Úsalo cuando el usuario pregunte sobre políticas, reglamentos, guías o procedimientos.
    """
    if BUSCADOR_FAISS is None:
        return "No hay base de conocimientos de texto inicializada."

    chunks_relacionados = BUSCADOR_FAISS.buscar_similares(consulta)
    if not chunks_relacionados:
        return "No se encontró información relevante en los documentos de texto."

    return "\n\n---\n\n".join([c.contenido for c in chunks_relacionados])


@tool
def listar_tablas_y_columnas() -> str:
    """Devuelve la lista de tablas/archivos tabulares disponibles (CSV, XLSX) con el nombre de sus columnas,
    cantidad de filas y tipos de datos.
    Úsalo ANTES de ejecutar código Pandas si necesitas conocer el esquema exacto de las tablas.
    """
    if not DATAFRAMES_DISPONIBLES:
        return "No hay tablas o archivos tabulares disponibles en el sistema."

    resumenes = []
    for nombre, df in DATAFRAMES_DISPONIBLES.items():
        cols = ", ".join(
            [f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)]
        )
        resumenes.append(
            f"📌 Tabla: '{nombre}' | Filas: {len(df)} | Columnas:\n   {cols}"
        )

    return "\n\n".join(resumenes)


@tool
def ejecutar_analisis_pandas(codigo_python: str) -> str:
    """Ejecuta código Python/Pandas para consultar, filtrar, agrupar o calcular estadísticas
    sobre los DataFrames disponibles en el diccionario global `DATAFRAMES_DISPONIBLES`.

    Instrucciones de código:
    - Las tablas están en el diccionario `DATAFRAMES_DISPONIBLES`, por ejemplo: `df = DATAFRAMES_DISPONIBLES['Reporte_General_de_Asignaturas_Virtuales']`.
    - Asigna el resultado final a una variable llamada `resultado` o imprímelo con `print()`.
    - Puedes realizar operaciones como:
      `df[df['Asignatura'].str.contains('Lenguaje', case=False, na=False)][['Clave', 'Asignatura', 'Nombre', 'Apellidos']]`
    """
    if not DATAFRAMES_DISPONIBLES:
        return "No hay tablas cargadas en el sistema para analizar."

    buffer_stdout = io.StringIO()
    globals_scope = {
        "pd": pd,
        "np": np,
        "DATAFRAMES_DISPONIBLES": DATAFRAMES_DISPONIBLES,
    }
    locals_scope: dict[str, Any] = {}

    stdout_original = sys.stdout
    try:
        sys.stdout = buffer_stdout
        exec(codigo_python, globals_scope, locals_scope)  # noqa: S102
        salida_print = buffer_stdout.getvalue().strip()

        resultado_var = locals_scope.get("resultado") or globals_scope.get("resultado")

        partes_resultado = []
        if salida_print:
            partes_resultado.append(salida_print)
        if resultado_var is not None:
            if isinstance(resultado_var, (pd.DataFrame, pd.Series)):
                partes_resultado.append(resultado_var.to_string())
            else:
                partes_resultado.append(str(resultado_var))

        if partes_resultado:
            return "\n\n".join(partes_resultado)
        return "El código se ejecutó correctamente sin salida."

    except Exception as error:  # noqa: BLE001
        return f"⚠️ Error al ejecutar código Pandas: {error}"
    finally:
        sys.stdout = stdout_original


def extraer_texto_de_mensaje(contenido: Any) -> str:
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        textos = []
        for elem in contenido:
            if isinstance(elem, dict) and elem.get("type") == "text":
                textos.append(str(elem.get("text", "")))
            elif isinstance(elem, str):
                textos.append(elem)
        return "".join(textos)
    return str(contenido) if contenido is not None else ""


class AgenteLangGraph:
    def __init__(self) -> None:
        if not CLAVE_API:
            raise ValueError("No se encontró GEMINI_API_KEY en el archivo .env")

        print("🚀 Inicializando Agente LangGraph con Gemini y Herramientas...")
        global BUSCADOR_FAISS

        cargar_tablas()
        docs_texto = cargar_documentos_texto()
        chunks_texto = crear_chunks(docs_texto)

        BUSCADOR_FAISS = BuscadorVectorialFAISS()
        BUSCADOR_FAISS.indexar_chunks(chunks_texto)

        self.llm = ChatGoogleGenerativeAI(
            model=MODELO_LLM,
            google_api_key=CLAVE_API,
            streaming=True,
        )

        herramientas = [
            consultar_documentos_texto,
            listar_tablas_y_columnas,
            ejecutar_analisis_pandas,
        ]

        prompt_sistema = """Eres el Agente de Información Empresarial interno.
Tu función es ayudar a los usuarios a consultar la información corporativa almacenada en la base de conocimientos, combinando documentos de texto y archivos tabulares.

Instrucciones de Uso de Herramientas:
1. Para preguntas sobre documentos no estructurados (normativas, políticas, guías, manuales o textos generales), utiliza la herramienta `consultar_documentos_texto`.
2. Para preguntas sobre datos estructurados, estadísticas, conteos o registros almacenados en tablas (CSV, Excel):
   - Si no conoces las tablas disponibles o sus columnas, llama primero a `listar_tablas_y_columnas`.
   - Luego, genera y ejecuta código Pandas preciso utilizando la herramienta `ejecutar_analisis_pandas`.
3. Responde siempre de forma clara, amable, estructurada y en español, basándote exclusivamente en la información recuperada por las herramientas.
"""

        self.grafo = create_agent(
            model=self.llm,
            tools=herramientas,
            system_prompt=prompt_sistema,
        )

    def generar_respuesta_stream(
        self, mensaje_usuario: str, historial: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        mensajes_input = []
        for h in historial:
            if isinstance(h, dict):
                rol = "user" if h.get("role") == "user" else "assistant"
                contenido = h.get("content", "")
                if contenido:
                    mensajes_input.append((rol, str(contenido)))

        mensajes_input.append(("user", mensaje_usuario))

        texto_acumulado = ""
        for event in self.grafo.stream({"messages": mensajes_input}):
            if isinstance(event, dict):
                nodo = event.get("model") or event.get("agent")
                if isinstance(nodo, dict) and "messages" in nodo and nodo["messages"]:
                    ultimo_msg = nodo["messages"][-1]
                    if hasattr(ultimo_msg, "content") and ultimo_msg.content:
                        texto = extraer_texto_de_mensaje(ultimo_msg.content)
                        if texto:
                            texto_acumulado = texto
                            yield texto_acumulado


# Instancia global del agente
AGENTE_LANGGRAPH: AgenteLangGraph | None = None


def responder_gradio(
    mensaje: str, historial: list[dict[str, str]]
) -> Generator[str, None, None]:
    global AGENTE_LANGGRAPH
    if AGENTE_LANGGRAPH is None:
        AGENTE_LANGGRAPH = AgenteLangGraph()
    yield from AGENTE_LANGGRAPH.generar_respuesta_stream(mensaje, historial)


def construir_interfaz() -> gr.Blocks:
    demo = gr.ChatInterface(
        fn=responder_gradio,
        title="Agente de Información Empresarial",
        description="Asistente corporativo diseñado para responder consultas de empleados sobre políticas internas, procedimientos de la empresa y datos de archivos de trabajo.",
    )
    return demo


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))


def main() -> None:
    global AGENTE_LANGGRAPH
    AGENTE_LANGGRAPH = AgenteLangGraph()
    demo = construir_interfaz()
    print(f"🌐 Lanzando servidor web Gradio disponible en http://{HOST}:{PORT} ...")
    demo.launch(server_name=HOST, server_port=PORT)


if __name__ == "__main__":
    main()
