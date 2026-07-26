import io
import os
import sys
import unicodedata
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import connect
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


class DictDataFramesDisponibles(dict[str, pd.DataFrame]):
    def __init__(self) -> None:
        super().__init__()
        self.rutas_tablas: dict[str, tuple[Path, str | None]] = {}
        self.esquemas: dict[str, list[tuple[str, str]]] = {}

    def registrar_tabla(
        self,
        nombre: str,
        ruta: Path,
        hoja: str | None,
        esquema: list[tuple[str, str]],
    ) -> None:
        self.rutas_tablas[nombre] = (ruta, hoja)
        self.esquemas[nombre] = esquema

    def __getitem__(self, key: str) -> pd.DataFrame:
        if dict.__contains__(self, key):
            return super().__getitem__(key)

        if key in self.rutas_tablas:
            ruta, hoja = self.rutas_tablas[key]
            if hoja is None:
                df = (
                    pd.read_csv(
                        str(ruta),
                        engine="python",
                        encoding="utf-8-sig",
                        on_bad_lines="skip",
                    )
                    .dropna(how="all", axis=0)
                    .dropna(how="all", axis=1)
                )
            else:
                df = (
                    pd.read_excel(str(ruta), sheet_name=hoja)
                    .dropna(how="all", axis=0)
                    .dropna(how="all", axis=1)
                )
            self[key] = df
            return df
        raise KeyError(f"Tabla '{key}' no encontrada en el diccionario.")


# Almacenamiento global perezoso de DataFrames cargados desde datos/
DATAFRAMES_DISPONIBLES = DictDataFramesDisponibles()


def cargar_tablas(directorio_datos: Path = Path("datos")) -> None:
    DATAFRAMES_DISPONIBLES.clear()
    DATAFRAMES_DISPONIBLES.rutas_tablas.clear()
    DATAFRAMES_DISPONIBLES.esquemas.clear()

    if not directorio_datos.exists():
        return

    for archivo in directorio_datos.glob("*"):
        ext = archivo.suffix.lower()
        if ext == ".csv":
            try:
                df_head = pd.read_csv(
                    archivo,
                    nrows=2,
                    engine="python",
                    encoding="utf-8-sig",
                    on_bad_lines="skip",
                )
                nombre_clave = archivo.stem.replace(" ", "_")
                esquema = [
                    (col, str(dtype))
                    for col, dtype in zip(df_head.columns, df_head.dtypes)
                ]
                DATAFRAMES_DISPONIBLES.registrar_tabla(
                    nombre_clave, archivo, None, esquema
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                pd.errors.EmptyDataError,
            ) as error:
                print(f"⚠️ Error al inspeccionar CSV {archivo.name}: {error}")

        elif ext in {".xlsx", ".xls"}:
            try:
                excel_file = pd.ExcelFile(str(archivo))
                for nombre_hoja in excel_file.sheet_names:
                    df_head = pd.read_excel(excel_file, sheet_name=nombre_hoja, nrows=2)
                    if not df_head.columns.empty:
                        nombre_clave = f"{archivo.stem}_{nombre_hoja}".replace(" ", "_")
                        esquema = [
                            (col, str(dtype))
                            for col, dtype in zip(df_head.columns, df_head.dtypes)
                        ]
                        DATAFRAMES_DISPONIBLES.registrar_tabla(
                            nombre_clave, archivo, nombre_hoja, esquema
                        )
            except (
                OSError,
                ValueError,
                RuntimeError,
                pd.errors.EmptyDataError,
            ) as error:
                print(f"⚠️ Error al inspeccionar Excel {archivo.name}: {error}")


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


RUTA_DB_SQLITE = Path("indice_conocimiento.db")
RUTA_FAISS_INDEX = Path("indice_faiss.index")


class BuscadorVectorialFAISS:
    def __init__(self) -> None:
        print(
            "🧠 Inicializando motor de conocimientos RAG en disco (SQLite + FAISS)..."
        )
        self.modelo_embedding = TextEmbedding(MODELO_EMBEDDING_ONNX)
        self.indice_faiss: faiss.IndexFlatL2 | None = None
        self.dimension = 384
        self.inicializar_db()

    def inicializar_db(self) -> None:
        with connect(RUTA_DB_SQLITE) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    id_documento TEXT,
                    indice INTEGER,
                    contenido TEXT
                )
                """
            )
            conn.commit()

    def indexar_chunks(self, lista_chunks: list[Chunk]) -> None:
        if not lista_chunks:
            return

        print(
            f"⚡ Indizando y guardando {len(lista_chunks)} fragmentos de texto en disco..."
        )
        textos = [chk.contenido for chk in lista_chunks]
        generator = self.modelo_embedding.embed(textos)
        embeddings_list = list(generator)
        matriz_embeddings = np.array(embeddings_list, dtype=np.float32)

        faiss.normalize_L2(matriz_embeddings)

        self.dimension = matriz_embeddings.shape[1]

        indice_int8 = faiss.IndexScalarQuantizer(
            self.dimension, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2
        )
        indice_int8.train(matriz_embeddings)
        indice_int8.add(matriz_embeddings)
        self.indice_faiss = indice_int8

        faiss.write_index(self.indice_faiss, str(RUTA_FAISS_INDEX))

        with connect(RUTA_DB_SQLITE) as conn:
            conn.execute("DELETE FROM chunks")
            conn.executemany(
                "INSERT INTO chunks (id, id_documento, indice, contenido) VALUES (?, ?, ?, ?)",
                [
                    (i, chk.id_documento, chk.indice, chk.contenido)
                    for i, chk in enumerate(lista_chunks)
                ],
            )
            conn.commit()

        print("✅ Base RAG e índice FAISS guardados en disco con éxito.")

    def cargar_desde_disco(self) -> bool:
        if RUTA_FAISS_INDEX.exists() and RUTA_DB_SQLITE.exists():
            try:
                self.indice_faiss = faiss.read_index(str(RUTA_FAISS_INDEX))
                print(
                    f"✅ Índice FAISS cargado desde disco ({self.indice_faiss.ntotal} vectores)."
                )
                return True
            except (OSError, RuntimeError):
                return False
        return False

    def buscar_similares(self, consulta: str, top_k: int = TOP_K) -> list[Chunk]:
        if (
            self.indice_faiss is None or self.indice_faiss.ntotal == 0
        ) and not self.cargar_desde_disco():
            return []

        generator = self.modelo_embedding.embed([consulta])
        vector = next(iter(generator))
        arr = np.array(vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(arr)

        if self.indice_faiss is None:
            return []

        _, indices = self.indice_faiss.search(arr, top_k * 3)

        ids_encontrados = [int(i) for i in indices[0] if i >= 0]
        if not ids_encontrados:
            return []

        resultados: list[Chunk] = []
        with connect(RUTA_DB_SQLITE) as conn:
            placeholders = ",".join("?" * len(ids_encontrados))
            query = f"SELECT id_documento, indice, contenido FROM chunks WHERE id IN ({placeholders})"
            rows = conn.execute(query, ids_encontrados).fetchall()

            for row in rows:
                resultados.append(
                    Chunk(id_documento=row[0], indice=row[1], contenido=row[2])
                )

        return resultados[:top_k]


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
    """Devuelve la lista de tablas/archivos tabulares disponibles (CSV, XLSX) con el nombre de sus columnas
    y tipos de datos.
    Úsalo ANTES de ejecutar código Pandas si necesitas conocer el esquema exacto de las tablas.
    """
    if not DATAFRAMES_DISPONIBLES.rutas_tablas:
        return "No hay tablas o archivos tabulares disponibles en el sistema."

    resumenes = []
    for nombre, esquema in DATAFRAMES_DISPONIBLES.esquemas.items():
        cols = ", ".join([f"{col} ({dtype})" for col, dtype in esquema])
        resumenes.append(f"📌 Tabla: '{nombre}' | Columnas:\n   {cols}")

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
    if not DATAFRAMES_DISPONIBLES.rutas_tablas:
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

        print("🚀 Inicializando Agente LangGraph...")
        global BUSCADOR_FAISS

        cargar_tablas()
        BUSCADOR_FAISS = BuscadorVectorialFAISS()

        if BUSCADOR_FAISS.cargar_desde_disco():
            print(
                "✅ Base RAG cargada instantáneamente desde disco (0 PDFs procesados en RAM)."
            )
        else:
            print("📑 Primera ejecución o nuevo índice: procesando PDFs en datos/...")
            documentos = cargar_documentos_texto()
            chunks = crear_chunks(documentos)
            BUSCADOR_FAISS.indexar_chunks(chunks)

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
