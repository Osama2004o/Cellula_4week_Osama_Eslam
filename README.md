# 🤖 AI Coding Assistant

An Intelligent AI Coding Assistant that answers programming questions, generates code, explains existing code, maintains conversation memory, retrieves knowledge from a ChromaDB Vector Database using RAG (Retrieval-Augmented Generation), executes generated code, and provides an interactive Streamlit UI.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [System Workflow](#-system-workflow)
- [Components Deep Dive](#-components-deep-dive)
- [Dataset](#-dataset)
- [Technologies](#-technologies)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Intent Classification** | LLM-based classifier categorizes queries as EXPLAIN, GENERATE, or FEEDBACK |
| 🔀 **Intelligent Routing** | Routes requests through the appropriate processing pipeline |
| 📖 **RAG Pipeline** | Retrieves relevant programming knowledge from ChromaDB before generating code |
| ✅ **Relevance Evaluation** | LLM evaluates whether retrieved context is relevant before using it |
| 📚 **Human Feedback Learning** | Users can teach the assistant new solutions that get stored in the knowledge base |
| 🧠 **Conversation Memory** | Maintains long-term context including preferences and previous interactions |
| ▶️ **Code Execution** | Execute generated Python code directly from the UI |
| 💡 **Code Explanation** | Detailed, step-by-step explanations of any code snippet |
| ⚡ **Code Generation** | High-quality code with comments, error handling, and best practices |
| 📎 **File Upload** | Upload code files for analysis |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│ Intent Classifier│  ◄── LLM-based (LangChain + OpenRouter)
│   (3 classes)    │
└────────┬────────┘
         │
    ┌────┼─────────────────┐
    │    │                  │
    ▼    ▼                  ▼
EXPLAIN  GENERATE        FEEDBACK
    │    │                  │
    │    ▼                  ▼
    │  ┌──────────┐    ┌──────────┐
    │  │ RAG      │    │ Embed &  │
    │  │ Retrieval│    │ Store in │
    │  └────┬─────┘    │ ChromaDB │
    │       │          └──────────┘
    │       ▼
    │  ┌──────────┐
    │  │Relevance │  ◄── LLM Evaluator
    │  │  Check   │
    │  └────┬─────┘
    │       │
    │   ┌───┴───┐
    │   │       │
    │ Relevant  Not Relevant
    │   │       │
    │   ▼       ▼
    │  LLM    Ask User
    │  Gen    for Help
    │   │
    ▼   ▼
┌─────────────┐
│  Response   │  ◄── With memory context
│  + Code     │
│  Execution  │
└─────────────┘
```

---

## 📁 Project Structure

```
Cellula_4week_Osama_Eslam/
├── app.py                  # Main Streamlit application (UI + orchestration)
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
├── .gitignore              # Git ignore rules
├── README.md               # Documentation & setup guide
│
├── config/                 # Configuration
│   ├── __init__.py
│   └── settings.py         # Centralized settings (API keys, model params, paths)
│
├── core/                   # Core logic modules (LangChain LCEL & Chains)
│   ├── __init__.py
│   ├── classifier.py       # LLM-based intent classifier (EXPLAIN / GENERATE / FEEDBACK)
│   ├── router.py           # Intelligent request router & pipeline orchestrator
│   ├── rag_pipeline.py     # RAG retrieval + relevance evaluation pipeline
│   └── memory.py           # Conversation memory with preference detection
│
├── database/               # Database layer
│   ├── __init__.py
│   └── vector_db.py        # LangChain ChromaDB wrapper (HuggingFaceBgeEmbeddings, seeding, search, feedback)
│
├── tools/                  # External execution tools
│   ├── __init__.py
│   └── code_executor.py    # Sandboxed Python code execution via subprocess
│
└── chroma_db/              # Persistent Chroma vector store (auto-generated)
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.9 or higher
- An OpenRouter API key ([Get one free here](https://openrouter.ai/keys))
- (Optional) NVIDIA GPU with CUDA for hardware-accelerated embeddings

### Step 1: Clone / Navigate to the project

```bash
git clone https://github.com/Osama2004o/Cellula_4week_Osama_Eslam.git
cd Cellula_4week_Osama_Eslam
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure environment

```bash
# Copy the example env file
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# Edit .env and configure your OpenRouter API key
OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

> **Note**: You can also enter or update your OpenRouter API key directly in the Streamlit web sidebar at runtime.

### Step 5: Run the application

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

---

## ⚙️ Configuration

All settings are centralized in [`config/settings.py`](file:///d:/Courses/nlp%20cellula/Cellula_4week_Osama_Eslam/config/settings.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENROUTER_API_KEY` | `os.getenv("OPENROUTER_API_KEY")` | Authentication key for OpenRouter API |
| `OPENROUTER_BASE_URL`| `https://openrouter.ai/api/v1` | OpenRouter OpenAI-compatible API base URL |
| `LLM_MODEL` | `openrouter/free` | OpenRouter model endpoint for reasoning & generation |
| `GENERATION_TEMPERATURE` | `0.5` | Sampling temperature for code generation and explanations |
| `MAX_OUTPUT_TOKENS` | `8192` | Maximum token limit per response |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model used by `HuggingFaceBgeEmbeddings` |
| `EMBEDDING_DEVICE` | `cuda` | Hardware target for embedding computation (`cuda` or `cpu`) |
| `CHROMA_COLLECTION_NAME` | `code_knowledge_base` | Name of the ChromaDB collection |
| `TOP_K_RESULTS` | `3` | Number of most relevant documents to retrieve in RAG |
| `CHUNK_SIZE` | `1000` | Chunk size (characters) for `RecursiveCharacterTextSplitter` |
| `CHUNK_OVERLAP` | `200` | Overlap (characters) between consecutive document chunks |
| `MAX_MEMORY_TURNS` | `20` | Maximum rolling conversation turns kept in memory |
| `CODE_EXECUTION_TIMEOUT` | `10` | Maximum execution timeout in seconds for user code runs |
| `HUMANEVAL_DATASET` | `openai/openai_humaneval` | Hugging Face dataset identifier for benchmark seeding |
| `HUMANEVAL_SPLIT` | `test` | Dataset split used for seeding |

---

## 💻 Usage

### 1. Seed the Knowledge Base

On first launch, click **"🌱 Seed Knowledge Base"** in the sidebar to load the HumanEval dataset (164 programming problems) into ChromaDB.

### 2. Ask Questions

**Code Generation:**
```
Write a Python function to find the longest common subsequence of two strings
```

**Code Explanation:**
```
Explain this code:
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)
```

**Teach the Assistant:**
```
Here is the correct implementation of merge sort:
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
```

### 3. Execute Code

Click the **"▶️ Execute"** button below any generated code block to run it directly.

### 4. Upload Files

Use the file uploader in the sidebar to analyze existing code files.

---

## 🔄 System Workflow

### Route 1: Code Explanation (No RAG)

```
User → Classifier → EXPLAIN → LLM (Direct Analysis) → Explanation
```

- **No** document retrieval
- **No** vector database search
- **No** embeddings
- The LLM directly analyzes the provided code

### Route 2: Code Generation (With RAG)

```
User → Classifier → GENERATE → RAG Retrieval → Relevance Check → LLM → Code
```

1. User query is embedded and searched against ChromaDB
2. Top-k documents are retrieved
3. A Relevance Evaluator LLM checks if context is useful
4. If **relevant**: Generate code with context
5. If **not relevant**: Ask user for correct solution

### Route 3: Human Feedback Learning

```
User Solution → Embedding Model → Vector → ChromaDB
```

1. User provides correct implementation
2. System generates embeddings
3. Creates metadata tags
4. Stores in ChromaDB for future retrieval

---

## 🔍 Components Deep Dive

### Intent Classifier (`core/classifier.py`)
- Uses LangChain ChatOpenAI with an OpenRouter free model
- Binary classification: EXPLAIN vs GENERATE vs FEEDBACK
- Falls back to GENERATE on ambiguous input

### Intelligent Router (`core/router.py`)
- Central orchestrator for all request types
- Injects conversation memory into every LLM call
- Handles errors gracefully

### RAG Pipeline (`core/rag_pipeline.py`)
- Retrieves documents from ChromaDB via similarity search
- Evaluates relevance using a separate LLM call
- Formats context for injection into generation prompts

### Vector Database (`database/vector_db.py`)
- LangChain Chroma vector store integration with persistent storage
- Embeddings via LangChain `HuggingFaceBgeEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2` on CUDA)
- Text chunking via LangChain `RecursiveCharacterTextSplitter`
- Seeded with HumanEval dataset (164 problems) converted into LangChain `Document` objects
- Supports user feedback insertion and LangChain retriever interface

### Conversation Memory (`core/memory.py`)
- Rolling window of conversation turns
- Automatic preference detection (language, framework)
- Formatted context for LLM injection

### Code Executor (`tools/code_executor.py`)
- Sandboxed subprocess execution
- Timeout enforcement
- stdout/stderr capture
- Markdown code block extraction

---

## 📊 Dataset

**OpenAI HumanEval** — 164 handwritten programming problems.

| Field | Description |
|-------|-------------|
| `task_id` | Unique problem identifier (e.g., `HumanEval/0`) |
| `prompt` | Function signature + docstring |
| `canonical_solution` | Reference implementation |
| `test` | Unit test cases |
| `entry_point` | Function name to test |

Source: [HuggingFace - openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval)

---

## 🛠️ Technologies

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Interactive web UI |
| **OpenRouter** | Free LLM API gateway (`openrouter/free`, Gemma, LLaMA, Mistral, etc.) |
| **LangChain Core & LCEL** | LLM orchestration, prompts, output parsers, and runnable chains |
| **LangChain Chroma** | Persistent vector store integration (`langchain_community.vectorstores.Chroma`) |
| **LangChain Embeddings** | Document embedding (`HuggingFaceBgeEmbeddings` with `all-MiniLM-L6-v2` on CUDA) |
| **LangChain Text Splitters** | Code chunking (`RecursiveCharacterTextSplitter`) |
| **HuggingFace Datasets** | Loading HumanEval dataset |
| **Python subprocess** | Sandboxed code execution |

---

## 📄 License

This project is for educational purposes as part of an NLP course project.
