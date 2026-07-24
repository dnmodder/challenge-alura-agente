# 🤖 Challenge Agente Alura

Un agente de inteligencia artificial basado en arquitectura **RAG (Retrieval-Augmented Generation)** diseñado para responder consultas internas de empleados sobre la documentación, políticas y procedimientos corporativos de la empresa.

---

## 🏗️ Descripción de la Arquitectura

El sistema está construido siguiendo una arquitectura por capas modular, local y desacoplada:

```
[ 📄 Archivos empresariales (datos/) ]
           │ (.md, .pdf, .docx, .doc, .csv, .txt)
           ▼
[ 🧩 Extractor Multiformato y Chunker ] ──> Genera fragmentos contextuales por bloques
           │
           ▼
[ 💻 Local Embeddings (MiniLM-L12 Multilingüe) ] ──> Vectorización e indización semántica local
           │
           ▼
[ ⚡ Base Vectorial FAISS (IndexFlatL2) ] ──> Almacenamiento e indización en memoria local
           │
           ▼ (Consulta del empleado)
[ 🧠 Motor RAG & Gemini 3.5 Flash Lite ] ──> Generación de respuesta con Streaming
           │
           ▼
[ 💬 Interfaz Gradio ChatInterface ] ──> Experiencia de chat en tiempo real (HTTP)
```

### 🛠️ Componentes Clave:

1. **📄 Extractor de Texto Multiformato:** Procesa automáticamente archivos `.md`, `.txt`, `.pdf`, `.docx`, `.doc` y `.csv`, convirtiendo registros tabulares y documentos en texto formateado para indización.
2. **🧩 Segmentación Eficiente por Bloques:** Fragmenta el contenido en bloques contextuales optimizados para maximizar la calidad de recuperación vectorial sin saturación.
3. **💻 Embeddings Locales Multilingües:** Emplea `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` para vectorizar e indizar localmente la información en español.
4. **⚡ Indización con FAISS:** Almacena los vectores en un índice `faiss.IndexFlatL2` en memoria para búsquedas semánticas e híbridas inmediatas.
5. **🧠 LLM con Streaming (Google Gemini API):** Emplea `gemini-3.5-flash-lite` para generar respuestas precisas, amables y formateadas en tiempo real mediante transmisión streaming en `Gradio ChatInterface`.

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
   Coloca la documentación interna en la carpeta `datos/` (esta carpeta está excluida del control de versiones).

---

## 🚀 Ejecución

Lanza la aplicación ejecutando:

```bash
uv run python app.py
```

La interfaz web estará disponible en `http://127.0.0.1:7860`.

---

## ❓ Preguntas Frecuentes (FAQ)

### 📄 ¿Qué tipos de archivos puedo agregar a la carpeta `datos/`?
El sistema admite archivos `.md`, `.txt`, `.pdf`, `.docx`, `.doc` y `.csv`.

### 🔑 ¿Es obligatorio configurar `HF_TOKEN`?
No, es totalmente opcional. El modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` es de libre acceso y se descarga automáticamente sin necesidad de autenticación.

### 🔄 ¿Cómo agrego nuevos documentos a la base de conocimientos?
Simplemente copia los nuevos archivos dentro de la carpeta `datos/` y vuelve a ejecutar `uv run python app.py`. El sistema indizará automáticamente los nuevos contenidos localmente al iniciar.

### 💾 ¿Cómo funciona el almacenamiento de datos en la aplicación?
La indización y búsqueda vectorial se realizan de forma 100% local con `SentenceTransformer` y `FAISS`. La API de Gemini se utiliza únicamente para generar la respuesta final en streaming al usuario.

### 🎯 ¿Se envían los documentos completos al modelo de lenguaje?
No. Únicamente se envían los 3 fragmentos (*chunks*) más relevantes recuperados por la búsqueda híbrida, lo que optimiza el consumo de tokens y garantiza respuestas rápidas.
