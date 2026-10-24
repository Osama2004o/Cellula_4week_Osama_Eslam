# 🚀 NLP & Database Assistant: AI Coding Partner & Voice-to-SQLite Studio

An advanced, dual-capability system combining:
1. 🎙️ **Voice & Text-to-SQLite Studio (`main.py` + `index.html`)**: An end-to-end voice and text to SQLite pipeline featuring speech transcription with `faster-whisper`, natural language to SQLite query translation with LangChain LCEL & OpenRouter, live execution against a seeded relational database, and a rich, glassmorphism web studio.
2. 🤖 **AI Coding Assistant (`app.py`)**: An intelligent programming assistant that answers questions, generates code, explains code, maintains conversation memory, retrieves knowledge from ChromaDB via RAG, and executes generated Python code in a Streamlit interface.

---

## 📋 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Features Matrix](#-features-matrix)
- [Project Structure](#-project-structure)
- [Voice-to-SQLite Studio (`main.py` & `index.html`)](#-voice-to-sqlite-studio-mainpy--indexhtml)
  - [FastAPI Application (`main.py`) Functions & Endpoints](#fastapi-application-mainpy-functions--endpoints)
  - [Interactive Web Studio (`index.html`) Features](#interactive-web-studio-indexhtml-features)
  - [End-to-End Workflow](#end-to-end-voice-to-sql-workflow)
- [AI Coding Assistant (`app.py`)](#-ai-coding-assistant-apppy)
  - [RAG & Classifier Workflow](#rag--classifier-workflow)
  - [Components Deep Dive](#components-deep-dive)
- [Database Schema Reference](#-database-schema-reference)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [How to Run](#-how-to-run)
- [Technologies](#-technologies)
- [License](#-license)

---

## 🏗️ Overview & Architecture

### 1. Voice-to-SQLite Pipeline (`main.py` + `index.html`)

```
 User Voice (Mic) / Audio File / Text
                 │
                 ▼
 ┌───────────────────────────────┐
 │   Speech-to-Text Transcription│  ◄── faster-whisper (English & Arabic)
 │     POST /transcribe          │
 └───────────────┬───────────────┘
                 │ Transcribed Text
                 ▼
 ┌───────────────────────────────┐
 │   Text-to-SQLite Generator    │  ◄── LangChain LCEL + OpenRouter
 │     POST /text-to-sql         │  ◄── Auto-injected SQLite Schema DDL
 └───────────────┬───────────────┘
                 │ Executable SQL Query
                 ▼
 ┌───────────────────────────────┐
 │   SQLite Query Execution      │  ◄── Local SQLite Database (app_data.db)
 │     POST /execute-sql         │
 └───────────────┬───────────────┘
                 │ Tabular Rows & Latency
                 ▼
 ┌───────────────────────────────┐
 │   Interactive Web Studio      │  ◄── index.html
 │  - Live Waveform Visualizer   │  - Client-side Table Filter
 │  - Editable SQL Code Editor   │  - One-click CSV Export
 └───────────────────────────────┘
```

### 2. AI Coding Assistant Pipeline (`app.py`)

```
User Query ──► Intent Classifier (EXPLAIN / GENERATE / FEEDBACK)
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
      EXPLAIN    GENERATE   FEEDBACK
         │          │          │
     Direct LLM    RAG         Embed & Store
     Analysis    Retrieval     in ChromaDB
                    │
             Relevance Check
               ┌────┴────┐
           Relevant   Not Relevant
               │         │
              LLM      Ask User
           Generation  for Help
```

---

## ✨ Features Matrix

| Module | Feature | Description |
|:-------|:--------|:------------|
| **Voice-to-SQL** | 🎙️ **Speech-to-Text** | High-performance speech recognition via `faster-whisper` (`base` model, `int8` quantization) supporting English and Arabic. |
| **Voice-to-SQL** | 🎛️ **Live Audio Visualizer** | HTML5 AudioContext recording with real-time dynamic animated canvas waveform and audio playback. |
| **Voice-to-SQL** | 🧠 **Text-to-SQLite LLM** | LangChain LCEL chain with OpenRouter models to turn English or Arabic prompts into valid, idiomatic SQLite queries. |
| **Voice-to-SQL** | 🛡️ **Schema-Grounded SQL** | Automatically injects the active database schema DDL into prompts for zero-hallucination queries. |
| **Voice-to-SQL** | ⚡ **Live SQL Execution** | Executes queries against SQLite, returning latency (ms), row counts, column names, and row tuples. |
| **Voice-to-SQL** | 📊 **Data Table & CSV Export** | Responsive glassmorphism table with real-time search/filter and client-side CSV download. |
| **Voice-to-SQL** | 🗄️ **Schema Viewer & Reset** | Interactive modal to view all relational tables and a 1-click database reset button. |
| **Coding Assistant** | 🔍 **Intent Classification** | LLM classifies developer queries into `EXPLAIN`, `GENERATE`, or `FEEDBACK`. |
| **Coding Assistant** | 📖 **RAG Pipeline** | Vector similarity retrieval from ChromaDB seeded with HumanEval programming benchmarks. |
| **Coding Assistant** | ✅ **Relevance Evaluation** | LLM evaluates retrieved context before generating code to avoid irrelevant hallucinations. |
| **Coding Assistant** | 📚 **Human Feedback Learning** | Users can submit corrections and new solutions to grow the ChromaDB vector database. |
| **Coding Assistant** | 🧠 **Conversation Memory** | Rolling conversational memory with automatic developer preference detection. |
| **Coding Assistant** | ▶️ **Sandboxed Execution** | Safely executes Python code blocks with timeout and stdout/stderr capture. |

---

## 📁 Project Structure

```
Cellula_4week_Osama_Eslam/
│
├── main.py                     # FastAPI server (Speech-to-Text, Text-to-SQL, execution API & UI server)
├── index.html                  # Responsive Web Studio (Audio recording, SQL editor, results table, schema viewer)
├── app.py                      # Main Streamlit application (AI Coding Assistant + RAG + Code Execution)
├── audio.m4a                   # Sample audio recording file for fallback/testing
├── requirements.txt            # Python package dependencies
├── .env.example                # Example environment configuration
├── .gitignore                  # Git ignore rules
├── README.md                   # Comprehensive project documentation
│
├── config/                     # Configuration module
│   ├── __init__.py
│   └── settings.py             # Centralized settings (API keys, model IDs, parameters, DB paths)
│
├── core/                       # Core AI and pipeline logic
│   ├── __init__.py
│   ├── sql_generator.py        # LangChain SQLite generation engine with clean SQL output parser
│   ├── classifier.py           # LLM-based intent classifier (EXPLAIN / GENERATE / FEEDBACK)
│   ├── router.py               # Request router & pipeline orchestrator for coding assistant
│   ├── rag_pipeline.py         # ChromaDB retrieval + relevance evaluation pipeline
│   └── memory.py               # Conversation memory with developer preference tracking
│
├── database/                   # Database & storage layer
│   ├── __init__.py
│   ├── sqlite_manager.py       # SQLite manager (connection, enterprise seed data, schema DDL, query runner)
│   ├── app_data.db             # Persistent SQLite database file
│   └── vector_db.py            # LangChain ChromaDB wrapper (HuggingFaceBgeEmbeddings, seeding, search)
│
├── schemas/                    # Pydantic data validation models
│   ├── __init__.py
│   ├── sql.py                  # Schemas for TextToSQLRequest/Response & ExecuteSQLRequest/Response
│   └── transcription.py        # Schemas for LanguageOption & TranscriptionResponse
│
├── tools/                      # Execution tools
│   ├── __init__.py
│   └── code_executor.py        # Sandboxed Python code execution via subprocess
│
└── chroma_db/                  # Persistent Chroma vector store for RAG (auto-generated)
```

---

## 🎙️ Voice-to-SQLite Studio (`main.py` & `index.html`)

The Voice-to-SQLite Studio provides a full-stack, production-grade API and Web interface that turns spoken or written natural language into executed SQLite queries.

### FastAPI Application (`main.py`) Functions & Endpoints

[`main.py`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/main.py) is built with **FastAPI** and orchestrates speech transcription, SQL generation, and database execution.

#### 1. Core Functions & Helpers

- **`init_database()`**:
  Called on startup to verify and initialize the SQLite database (`database/app_data.db`). Seeds enterprise tables (`categories`, `products`, `customers`, `employees`, `sales`, `orders`, `order_items`) if they do not exist.
- **`model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")`**:
  Loads the `faster-whisper` speech recognition model into memory once at server startup for fast transcription without per-request model loading overhead.
- **`transcribe_audio_file(audio_path: str, language: str = "en", beam_size: int = 5) -> dict`**:
  Core speech-to-text helper using `faster-whisper`. Extracts individual timestamped segments (`start`, `end`, `text`), cleans whitespace, and compiles the combined transcript and detected language.
- **`resolve_audio_source(file: Optional[UploadFile]) -> tuple[str, Optional[str]]`**:
  Inspects incoming audio requests. If an audio file was uploaded or recorded in the browser, it saves it safely to a temporary file (`.webm`, `.wav`, etc.) for processing. If no file is passed, it falls back to the bundled [`audio.m4a`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/audio.m4a). Ensures temporary files are cleaned up in a `finally` block.

#### 2. REST Endpoints Reference

| Endpoint | Method | Description | Request / Parameters | Response |
|:---------|:------:|:------------|:---------------------|:---------|
| `/` or `/ui` | `GET` | **Serve Web UI**: Serves the single-page [`index.html`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/index.html) application. | None | `HTMLResponse` (index.html) |
| `/api/info` | `GET` | **API Health & Info**: Returns server health status, Swagger docs link, workflow steps, and endpoint registry. | None | JSON metadata |
| `/transcribe` | `POST` | **Step 1: Audio Transcription**: Transcribes voice or audio files using `faster-whisper`. Supports browser-recorded WebM/WAV or uploaded audio files in English and Arabic. | `file`: UploadFile (optional), `language`: `english`/`arabic`, `beam_size`: int (default `5`) | `TranscriptionResponse` (`language`, `transcript`, `segments`) |
| `/text-to-sql` | `POST` | **Step 2: Text-to-SQLite**: Translates natural language questions into clean, executable SQLite syntax via LangChain & OpenRouter. Automatically injects the active database schema if `schema_context` is omitted. | `TextToSQLRequest` (`text`: str, `schema_context`: Optional[str]) | `TextToSQLResponse` (`query`, `sql`, `dialect`, `model_used`) |
| `/execute-sql` | `POST` | **Step 3: SQLite Query Execution**: Executes a SQLite query against `database/app_data.db`. Returns column names, 2D row data, row count, and execution time in milliseconds. | `ExecuteSQLRequest` (`sql`: str) | `ExecuteSQLResponse` (`success`, `query`, `columns`, `rows`, `row_count`, `execution_time_ms`, `error`) |
| `/database/schema` | `GET` | **Schema Inspector**: Fetches the complete SQLite schema DDL (`CREATE TABLE ...`) and detailed table definitions with column types. | None | JSON (`schema_ddl`, `tables`) |
| `/database/reset` | `POST` | **Reset Database**: Re-creates all sample tables and restores fresh seed records. | None | JSON (`success`, `message`) |

---

### Interactive Web Studio (`index.html`) Features

[`index.html`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/index.html) is a standalone, modern frontend with a custom glassmorphism dark aesthetic:

1. **Header & Status Bar**:
   - Real-time server connection indicator pill (`Connecting...` ➔ `Online` / `Offline`).
   - Quick action to open the **Active SQLite Database Schema Modal**.
   - **Reset DB** button to restore seed data with one click.
2. **Quick Query Chips**:
   - Pre-loaded sample queries in English and Arabic (e.g., *"Top 5 Expensive Products"*, *"Total Sales by Region"*, *"أعلى 5 منتجات سعراً"*).
   - Clicking any chip instantly fills the prompt for fast testing.
3. **Step 1: Audio Studio & Transcription**:
   - **Microphone Recording Zone**: One-click recording using the HTML5 `MediaRecorder` API.
   - **Live Waveform Visualizer**: Real-time canvas waveform animation driven by Web Audio API `AnalyserNode`.
   - **Recording Timer & Audio Preview Player**: Review recorded audio before transcribing.
   - **File Upload**: Direct file picker for `.wav`, `.mp3`, `.m4a`, `.webm` files.
   - **Language Selector**: Choose between English (`en`) and Arabic (`ar`).
   - **Editable Transcript Box**: Spoken or uploaded transcripts appear in an editable text box.
4. **Step 2: Text-to-SQLite Generator**:
   - **"Generate SQL" Button**: Triggers the LangChain LLM pipeline to produce valid SQLite.
   - **SQL Code Editor**: Monospace code area allowing developers to review, edit, or write custom SQL.
   - **Copy to Clipboard**: Quick button with animated "Copied!" toast notification.
   - **Target Engine Indicator**: Tagged for SQLite 3.
5. **Step 3: Database Query Results Table**:
   - **Live Execution Stats**: Tracks Query Latency in milliseconds (`ms`), Row Count, and Column Count.
   - **Dynamic Data Table**: Clean, scrollable table with sticky headers, zebra striping, and hover effects.
   - **Real-Time Row Search**: Filter query results instantly as you type.
   - **Export CSV**: Download query results as a `.csv` file.
   - **Error Alert Banner**: Catches syntax or runtime errors with descriptive troubleshooting messages.
6. **Schema Inspector Modal**:
   - Inspect all database tables, columns, data types, and primary key tags without leaving the page.

---

### End-to-End Voice-to-SQL Workflow

```
1. Spoken Query: "Show me the top 5 most expensive products and their stock"
                        │
                        ▼ (faster-whisper)
2. Transcribed Text: "Show me the top 5 most expensive products and their stock"
                        │
                        ▼ (LangChain + OpenRouter + SQLite Schema)
3. Generated SQL:
   SELECT name, price, stock_quantity 
   FROM products 
   ORDER BY price DESC 
   LIMIT 5;
                        │
                        ▼ (sqlite3 execute)
4. Tabular Results:
   ┌────────────────────────────────┬─────────┬────────────────┐
   │ name                           │ price   │ stock_quantity │
   ├────────────────────────────────┼─────────┼────────────────┤
   │ ProBook Laptop 15              │ 1299.99 │ 45             │
   │ Leather Office Sofa            │ 750.00  │ 10             │
   │ Ultra-Wide 4K Monitor 32-inch  │ 599.00  │ 30             │
   │ Standing Electric Desk         │ 499.00  │ 18             │
   │ Smart Espresso Machine         │ 399.00  │ 22             │
   └────────────────────────────────┴─────────┴────────────────┘
   [ Latency: 1.4 ms | Rows: 5 | Columns: 3 ]
```

---

## 🤖 AI Coding Assistant (`app.py`)

[`app.py`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/app.py) is an interactive Streamlit application serving as an intelligent coding partner.

### RAG & Classifier Workflow

1. **Code Explanation Route (No RAG)**:
   - Queries classified as `EXPLAIN` bypass retrieval and pass directly to the LLM for step-by-step structural analysis.
2. **Code Generation Route (With RAG)**:
   - Queries classified as `GENERATE` are embedded and searched against ChromaDB.
   - An LLM Relevance Evaluator validates whether retrieved HumanEval benchmark snippets are relevant.
   - If relevant: Context is injected into the generation prompt.
   - If not relevant: Falls back gracefully or prompts the user for guidance.
3. **Human Feedback Learning Route**:
   - Solutions provided by the user under `FEEDBACK` are embedded and committed into ChromaDB for continuous improvement.

### Components Deep Dive

- **Intent Classifier ([`core/classifier.py`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/core/classifier.py))**: LangChain LCEL chain classifying queries into `EXPLAIN`, `GENERATE`, or `FEEDBACK`.
- **Intelligent Router ([`core/router.py`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/core/router.py))**: Orchestrates requests and injects conversation context into LLM runs.
- **RAG Pipeline ([`core/rag_pipeline.py`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/core/rag_pipeline.py))**: Top-k vector retrieval from ChromaDB paired with LLM relevance evaluation.
- **Vector Database ([`database/vector_db.py`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/database/vector_db.py))**: LangChain ChromaDB wrapper using `HuggingFaceBgeEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`).
- **Conversation Memory ([`core/memory.py`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/core/memory.py))**: Rolling turn memory with automatic preference detection (preferred languages, coding styles).
- **Code Executor ([`tools/code_executor.py`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/tools/code_executor.py))**: Sandboxed Python execution engine with timeouts and output capture.

---

## 🗄️ Database Schema Reference

The SQLite database ([`database/app_data.db`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/database/app_data.db)) is initialized automatically by [`database/sqlite_manager.py`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/database/sqlite_manager.py):

| Table | Description | Key Columns |
|:------|:------------|:------------|
| `categories` | Product departments & categories | `id`, `name`, `description` |
| `products` | Catalog items, stock, and ratings | `id`, `name`, `category_id`, `price`, `stock_quantity`, `rating` |
| `customers` | Registered customer records | `id`, `name`, `email`, `city`, `country`, `balance`, `created_at` |
| `employees` | Staff and department data | `id`, `name`, `department`, `role`, `salary`, `hire_date` |
| `sales` | Regional transaction records | `id`, `product_id`, `units_sold`, `total_amount`, `region`, `sale_date` |
| `orders` | Customer order headers | `id`, `customer_id`, `order_date`, `total_amount`, `status` |
| `order_items`| Individual order line items | `id`, `order_id`, `product_id`, `quantity`, `unit_price` |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.9, 3.10, or 3.11
- An OpenRouter API Key ([Get one free here](https://openrouter.ai/keys))
- (Optional) NVIDIA GPU with CUDA for faster embeddings computation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Osama2004o/Cellula_4week_Osama_Eslam.git
cd Cellula_4week_Osama_Eslam
```

### Step 2: Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install fastapi uvicorn faster-whisper python-multipart
```

### Step 4: Configure Environment Variables

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and configure your OpenRouter API key:
```env
OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

---

## ⚙️ Configuration

Centralized settings in [`config/settings.py`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/config/settings.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENROUTER_API_KEY` | `os.getenv("OPENROUTER_API_KEY")` | Authentication key for OpenRouter API |
| `OPENROUTER_BASE_URL`| `https://openrouter.ai/api/v1` | OpenRouter OpenAI-compatible API base URL |
| `LLM_MODEL` | `openrouter/free` | OpenRouter model endpoint for reasoning & SQL generation |
| `GENERATION_TEMPERATURE` | `0.5` | Sampling temperature for code generation |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face sentence embedding model |
| `EMBEDDING_DEVICE` | `cuda` (or fallback `cpu`) | Embedding compute target |
| `CHROMA_COLLECTION_NAME` | `code_knowledge_base` | ChromaDB collection name |
| `TOP_K_RESULTS` | `3` | Number of documents retrieved during RAG |
| `CODE_EXECUTION_TIMEOUT` | `10` | Timeout in seconds for sandboxed Python code runs |

---

## 💻 How to Run

### Option A: Run the Voice-to-SQLite Studio & API (`main.py` + `index.html`)

Start the FastAPI application with Uvicorn:

```bash
python main.py
```
*Or using the Uvicorn CLI directly:*
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- **Interactive Web Studio**: Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) or [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui)
- **Interactive Swagger API Documentation**: Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: Navigate to [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### Option B: Run the AI Coding Assistant (`app.py`)

Start the Streamlit application:

```bash
streamlit run app.py
```

- **Streamlit Web UI**: Automatically launches at [http://localhost:8501](http://localhost:8501).
- First time running: Click **"🌱 Seed Knowledge Base"** in the sidebar to populate ChromaDB with the HumanEval dataset.

---

## 🛠️ Technologies

| Technology | Purpose |
|:-----------|:--------|
| **FastAPI** | High-performance async web framework powering `main.py` |
| **Uvicorn** | ASGI production server for the FastAPI application |
| **faster-whisper** | State-of-the-art CTranslate2 Whisper implementation for speech-to-text |
| **Vanilla HTML5/CSS3/JS** | Responsive, glassmorphism single-page application (`index.html`) with no heavy JS framework overhead |
| **Web Audio API** | Real-time audio waveform visualizer via `AudioContext` and `AnalyserNode` |
| **SQLite 3** | Embedded relational database engine (`database/app_data.db`) |
| **Streamlit** | Interactive UI framework for the AI Coding Assistant (`app.py`) |
| **LangChain Core & LCEL** | LLM chaining, prompts, output parsing, and SQLite query generation |
| **OpenRouter** | LLM API gateway (`openrouter/free`, Gemma, LLaMA, Mistral) |
| **ChromaDB** | Persistent vector database for RAG code retrieval |
| **Hugging Face Sentence Transformers** | Local embedding models (`all-MiniLM-L6-v2`) |
| **Pydantic v2** | Strict data validation and typing schemas (`schemas/`) |

---

## 📄 License

This repository is maintained for educational and development purposes as part of the NLP Cellula course project.
