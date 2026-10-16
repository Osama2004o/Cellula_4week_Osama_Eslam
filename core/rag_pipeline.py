# ============================================
# RAG Pipeline Module
# ============================================
# Implements the Retrieval-Augmented Generation
# pipeline for code generation requests.
#
# Pipeline steps:
#   1. Receive the user's query
#   2. Search ChromaDB for relevant documents
#   3. Pass retrieved context to a Relevance
#      Evaluator (another LLM call via LangChain)
#   4. If relevant → generate code with context
#      If not relevant → ask user for help
# ============================================

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
)
from database.vector_db import VectorDatabase


class RAGPipeline:

    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db

    # ──────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────

    def retrieve(self, query: str) -> list[dict]:
        results = self.vector_db.search(query)
        return results

    # ──────────────────────────────────────────
    # Relevance Evaluation
    # ──────────────────────────────────────────

    def check_relevance(self, query: str, retrieved_docs: list[dict]) -> bool:
        if not retrieved_docs:
            return False

        # Build the context string from retrieved docs
        context = self._format_context(retrieved_docs)

        evaluator_prompt = """You are a relevance evaluator for a coding assistant.

You will be given:
1. A user's coding request
2. Retrieved context from a knowledge base

Your job is to determine if the retrieved context is sufficiently relevant
to help answer the user's request accurately.

Rules:
- Respond with ONLY one word: RELEVANT or NOT_RELEVANT
- The context is RELEVANT if it contains code patterns, algorithms, functions,
  or concepts directly applicable to the user's request.
- The context is NOT_RELEVANT if it is about a completely different topic,
  or if it would not meaningfully help answer the request.
- Partial relevance (e.g., same language, related concepts) counts as RELEVANT.
"""

        try:
            llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base=OPENROUTER_BASE_URL,
                temperature=0,
                max_tokens=10,
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", evaluator_prompt),
                ("human", "User Request: {query}\n\nRetrieved Context:\n{context}"),
            ])

            eval_chain = prompt | llm | StrOutputParser()
            result = eval_chain.invoke({"query": query, "context": context}).strip().upper()

            return "RELEVANT" in result

        except Exception as e:
            print(f"[Relevance Check Error] {e}")
            # Default to relevant on error to avoid blocking
            return True

    # ──────────────────────────────────────────
    # Context Formatting
    # ──────────────────────────────────────────

    def _format_context(self, docs: list[dict]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc["metadata"].get("source", "Unknown")
            task_id = doc["metadata"].get("task_id", "N/A")
            distance = doc.get("distance", "N/A")

            parts.append(
                f"--- Document {i} (Source: {source}, "
                f"Task: {task_id}, Distance: {distance}) ---\n"
                f"{doc['document']}\n"
            )

        return "\n".join(parts)

    def get_context_string(self, docs: list[dict]) -> str:
        return self._format_context(docs)

