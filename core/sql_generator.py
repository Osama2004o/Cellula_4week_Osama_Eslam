import re
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
)

# ==============================================================================
# Prompt for Text-to-SQLite Generation (Supports English & Arabic)
# ==============================================================================
SQL_SYSTEM_PROMPT = """You are an elite SQL Engineer and Database Architect specializing in SQLite.
Your task is to convert natural language queries (in English or Arabic) into clean, correct, and highly optimized SQLite queries.

### Instructions & Rules:
1. **Multilingual Understanding**:
   - The user query can be in **English**, **Arabic** (Standard or Dialectal), or mixed.
   - Accurately interpret the semantic intent of the query regardless of the input language.

2. **Schema & Tables**:
   - If **Schema Context** is provided below, you MUST strictly use the exact table and column names provided.
   - If NO schema context is provided, infer standard canonical relational table and column names (e.g. `sales`, `products`, `categories`, `orders`, `order_items`, `customers`).

3. **Database Dialect**:
   - Always generate valid, idiomatic **SQLite** syntax.
   - Use SQLite-compatible functions (e.g. `strftime` for dates, `COALESCE`, `ROUND`, `LIMIT`).
   - Avoid syntax specific to other database engines (e.g., do NOT use `ILIKE`, `TOP`, `DATEADD`, or non-SQLite functions).

4. **Correctness & Best Practices**:
   - Use appropriate `JOIN`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, and aggregate functions (`SUM`, `COUNT`, `AVG`, `MIN`, `MAX`).
   - Use meaningful column aliases (e.g. `AS total_sales`).
   - Default to safe `SELECT` queries unless explicitly instructed otherwise.

5. **Output Format**:
   - Always enclose your final SQLite query inside a ```sql ... ``` code block.
   - Do NOT include conversational commentary, preambles, or greetings.
"""

SQL_HUMAN_PROMPT = """Target Database Engine: SQLite

Database Schema Context:
{schema_context}

Natural Language Request:
{query}

Generate the SQLite query:"""


def _get_sql_llm() -> ChatOpenAI:
    """Initialize LangChain ChatOpenAI configured for OpenRouter free models."""
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.0,
        max_tokens=1000,
    )


def clean_sql_output(raw_output: str) -> str:
    """Extract clean SQL query from markdown code blocks or raw text."""
    text = raw_output.strip()

    # 1. Match markdown code block ```sql ... ``` or ``` ... ```
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. If no markdown block, strip any preambles before SQL keywords
    sql_start = re.search(r"\b(SELECT|WITH|SHOW|DESCRIBE|EXPLAIN|CREATE|INSERT|UPDATE|DELETE)\b", text, re.IGNORECASE)
    if sql_start:
        text = text[sql_start.start():].strip()

    return text


def generate_sql_query(
    natural_language_query: str,
    schema_context: Optional[str] = None
) -> dict:
    """
    Translates natural language text (in English or Arabic) into an executable SQLite query.
    
    Args:
        natural_language_query: The transcribed text or user request.
        schema_context: Optional DDL or table description to ground the query.

    Returns:
        A dictionary containing query, generated sql, dialect ('SQLite'), and model_used.
    """
    llm = _get_sql_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SQL_SYSTEM_PROMPT),
        ("human", SQL_HUMAN_PROMPT),
    ])

    chain = prompt | llm | StrOutputParser()

    formatted_context = schema_context.strip() if schema_context else "None provided (infer standard relational schema)."

    raw_result = chain.invoke({
        "query": natural_language_query,
        "schema_context": formatted_context,
    })

    sql_query = clean_sql_output(raw_result)

    return {
        "query": natural_language_query,
        "sql": sql_query,
        "dialect": "SQLite",
        "model_used": LLM_MODEL,
    }
