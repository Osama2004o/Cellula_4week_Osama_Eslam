import os
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
from faster_whisper import WhisperModel

# Import Pydantic schemas from schemas folder
from schemas import (
    LanguageOption,
    TranscriptionResponse,
    TextToSQLRequest,
    TextToSQLResponse,
    ExecuteSQLRequest,
    ExecuteSQLResponse,
)

# Import LangChain SQLite generation engine
from core.sql_generator import generate_sql_query

# Import SQLite Database Manager
from database.sqlite_manager import (
    execute_sqlite_query,
    get_sqlite_schema,
    get_database_tables_info,
    init_database,
)

app = FastAPI(
    title="Speech-to-Text, Text-to-SQL & SQLite Query Execution API",
    description="End-to-end voice and text to SQLite pipeline: audio transcription (faster-whisper), SQLite query generation (LangChain + OpenRouter), and interactive database execution.",
    version="3.0.0"
)

# Enable CORS for browser recording, frontend clients, and local files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure sample database is initialized at startup
init_database()

# Load faster-whisper model (loaded once at startup)
MODEL_SIZE = "base"
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe_audio_file(
    audio_path: str,
    language: str = "en",
    beam_size: int = 5
) -> dict:
    """Core transcription helper function using faster-whisper."""
    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        language=language
    )

    segment_list = []
    text_parts = []

    for seg in segments_gen:
        cleaned_text = seg.text.strip()
        if cleaned_text:
            text_parts.append(cleaned_text)
            segment_list.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": cleaned_text
            })

    full_transcript = " ".join(text_parts)
    return {
        "language": info.language,
        "transcript": full_transcript,
        "segments": segment_list
    }


def resolve_audio_source(file: Optional[UploadFile]) -> tuple[str, Optional[str]]:
    """Helper to resolve uploaded or recorded file or fallback to local audio.m4a."""
    if file is not None and file.filename:
        suffix = os.path.splitext(file.filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
        return temp_path, temp_path
    else:
        default_audio = os.path.join(os.path.dirname(__file__), "audio.m4a")
        if os.path.exists(default_audio):
            return default_audio, None
        raise HTTPException(
            status_code=400,
            detail="No audio file uploaded or recorded, and default 'audio.m4a' was not found."
        )


# ==============================================================================
# UI Web App Endpoint
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    """Serves the interactive web interface."""
    html_file = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_file):
        return FileResponse(html_file, media_type="text/html")
    return HTMLResponse(
        content="<h2>index.html not found. Please ensure index.html exists in the root directory.</h2>",
        status_code=404
    )


@app.get("/api/info")
def api_info():
    """Provides metadata about the API endpoints and active workflow."""
    return {
        "message": "Speech-to-Text, Text-to-SQLite & Execution API is running",
        "docs_url": "/docs",
        "workflow": [
            "Step 1: Record audio in browser or call POST /transcribe -> get transcribed text",
            "Step 2: Call POST /text-to-sql with text -> get generated SQLite query",
            "Step 3: Call POST /execute-sql with the query -> get table query results"
        ],
        "endpoints": {
            "transcribe": "POST /transcribe (Audio -> Text)",
            "text_to_sql": "POST /text-to-sql (Text -> SQLite query)",
            "execute_sql": "POST /execute-sql (SQLite query -> Results Table)",
            "database_schema": "GET /database/schema (View active DB tables and DDL)",
            "database_reset": "POST /database/reset (Reset DB to original seed data)"
        }
    }


# Step 1: Speech-to-Text Transcription
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: Optional[UploadFile] = File(
        None,
        description="Audio file to transcribe (.webm, .wav, .mp3, .m4a)"
    ),
    language: LanguageOption = Query(
        LanguageOption.english,
        description="Choose language: 'english' ('en') or 'arabic' ('ar')"
    ),
    beam_size: int = Query(5, description="Beam size for decoding")
):
    """
    Step 1: Transcribes spoken audio into text in English or Arabic.
    Supports browser-recorded audio (WebM/WAV) or uploaded files.
    """
    lang_code = "ar" if language in (LanguageOption.arabic, LanguageOption.ar) else "en"

    audio_source, temp_file_path = resolve_audio_source(file)
    try:
        result = transcribe_audio_file(
            audio_path=audio_source,
            language=lang_code,
            beam_size=beam_size if isinstance(beam_size, int) else 5
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


# Step 2: Text-to-SQLite Query Generation (LangChain + OpenRouter)
@app.post("/text-to-sql", response_model=TextToSQLResponse)
async def text_to_sql(payload: TextToSQLRequest):
    """
    Step 2: Converts transcribed natural language text (English or Arabic) into a clean SQLite query
    using LangChain and OpenRouter.
    If no schema_context is passed, it automatically injects the active SQLite database schema!
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="The input text cannot be empty.")

    try:
        # Default to the active SQLite schema if none is provided
        effective_schema = payload.schema_context.strip() if payload.schema_context else get_sqlite_schema()

        result = generate_sql_query(
            natural_language_query=payload.text,
            schema_context=effective_schema
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQLite generation error: {str(e)}")


# Step 3: SQLite Query Execution & Table Results
@app.post("/execute-sql", response_model=ExecuteSQLResponse)
async def execute_sql(payload: ExecuteSQLRequest):
    """
    Step 3: Takes a SQLite query from the text-to-sql function, executes it against
    the SQLite database, and returns column headers and rows for table rendering.
    """
    if not payload.sql or not payload.sql.strip():
        raise HTTPException(status_code=400, detail="SQL query cannot be empty.")

    try:
        result = execute_sqlite_query(payload.sql)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database execution error: {str(e)}")


# Database Schema & Management Endpoints
@app.get("/database/schema")
def get_db_schema():
    """Returns the SQLite schema DDL and table details for UI inspection."""
    try:
        return {
            "schema_ddl": get_sqlite_schema(),
            "tables": get_database_tables_info()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch schema: {str(e)}")


@app.post("/database/reset")
def reset_db():
    """Resets the sample SQLite database to its fresh initial state."""
    try:
        init_database(force=True)
        return {"success": True, "message": "Database successfully reset to seed data."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database reset error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
