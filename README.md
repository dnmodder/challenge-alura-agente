# 🤖 Challenge Agente Alura (LangGraph Edition)

Un agente de inteligencia artificial avanzado basado en **LangGraph** y **LangChain** diseñado para responder consultas corporativas combinando **recuperación RAG para documentos no estructurados** (PDF, DOCX, MD, TXT) y **ejecución dinámica de código Pandas para datos estructurados** (CSV, XLSX, XLS).

---

## 🏗️ Descripción de la Arquitectura

El sistema utiliza un **Estado de Grafo (`StateGraph`)** en LangGraph que analiza la intención de la consulta y delega el procesamiento a la herramienta correspondiente:

```
                                 [ 💬 Consulta del Usuario ]
                                             │
                                             ▼
                               [ 🧠 Agente LangGraph (Gemini) ]
                                      │              │
                   ┌──────────────────┘              └──────────────────┐
                   ▼                                                    ▼
   [ 📄 Herramienta Texto RAG ]                         [ 📊 Herramienta Tabular Pandas ]
   - Archivos: .pdf, .docx, .md, .txt                   - Archivos: .csv, .xlsx, .xls
   - Embeddings: MiniLM (Local)                         - Esquema: Inspección dinámica de columnas
   - Base Vectorial: FAISS                              - Ejecución: Código Pandas dinámico
                   │                                                    │
                   └──────────────────┐              ┌──────────────────┘
                                      ▼              ▼
                                [ 💬 Respuesta en Streaming ]
                                (Gradio ChatInterface - Port 7860)
```

### 🛠️ Componentes Clave:

1. **🧠 Orquestador LangGraph (`create_react_agent`):** Utiliza `gemini-3.5-flash-lite` vía `langchain-google-genai` para razonar sobre qué herramienta invocar según la pregunta del usuario.
2. **📊 Ejecutor Dinámico de Código Pandas (`ejecutar_analisis_pandas`):** Permite al modelo consultar, filtrar, agrupar o calcular estadísticas sobre cualquier archivo CSV o libro de Excel (`.xlsx`, `.xls`) cargado en `datos/`, sin importar el esquema o nombres de columnas.
3. **📄 Buscador Semántico FAISS (`consultar_documentos_texto`):** Procesa archivos de texto no estructurado vectorizando localmente en CPU con `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
4. **💬 Interfaz Gradio Streaming:** Servidor web interactivo desplegado en `http://127.0.0.1:7860`.

---

## 📋 Requisitos Previos

* 🐍 **Python >= 3.10**
* ⚡ **`uv` instalado**
* 🔑 **Clave API de Google Gemini (`GEMINI_API_KEY`)**
* 🔑 **Token de Hugging Face opcional (`HF_TOKEN`)**

---

## ⚙️ Instalación y Configuración

1. **Instalar dependencias con `uv`:**
   ```bash
   uv sync
   ```

2. **Configurar las variables de entorno:**
   Crea un archivo `.env` en la raíz del proyecto basándote en la plantilla `.env.example`:
   ```bash
   GEMINI_API_KEY=tu_clave_api_de_gemini_aqui
   HF_TOKEN=tu_token_de_huggingface_opcional_aqui
   ```

3. **Documentos de la empresa:**
   Coloca tus documentos (`.md`, `.pdf`, `.docx`, `.doc`, `.txt`, `.csv`, `.xlsx`, `.xls`) en la carpeta `datos/`.

---

## 🚀 Ejecución

Lanza el agente ejecutando:

```bash
uv run python app.py
```

La interfaz web estará disponible en `http://127.0.0.1:7860`.

---

## ❓ Preguntas Frecuentes (FAQ)

### 📊 ¿Cómo maneja el agente los archivos CSV y Excel?
LangGraph identifica los archivos tabulares cargados en `datos/`, inspecciona sus nombres de columnas y genera expresiones de Pandas en tiempo real para obtener filtrados, conteos o búsquedas exactas.

### 📄 ¿Qué ocurre con los archivos de texto (PDF, DOCX, MD)?
Se fragmentan (*chunking*) e indizan localmente en memoria utilizando `FAISS` y `SentenceTransformer`, permitiendo búsquedas semánticas eficientes para responder preguntas sobre políticas corporativas.
