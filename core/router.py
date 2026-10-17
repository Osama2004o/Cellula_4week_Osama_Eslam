from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    GENERATION_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
)
from core.classifier import classify_intent
from core.rag_pipeline import RAGPipeline
from core.memory import ConversationMemory
from database.vector_db import VectorDatabase


def _get_llm(temperature: float = None, max_tokens: int = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature if temperature is not None else GENERATION_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else MAX_OUTPUT_TOKENS,
    )


class Router:

    def __init__(
        self,
        vector_db: VectorDatabase,
        memory: ConversationMemory,
    ):
        self.vector_db = vector_db
        self.memory = memory
        self.rag_pipeline = RAGPipeline(vector_db)

    def process(self, user_message: str) -> dict:
        # Step 1: Classify intent
        intent = classify_intent(user_message)

        # Step 2: Route to the appropriate pipeline
        if intent == "EXPLAIN":
            result = self._handle_explain(user_message)
        elif intent == "FEEDBACK":
            result = self._handle_feedback(user_message)
        else:  # GENERATE
            result = self._handle_generate(user_message)

        # Step 3: Update conversation memory
        self.memory.add_turn("user", user_message)
        self.memory.add_turn("assistant", result["response"])

        return result

    # ──────────────────────────────────────────
    # Route 1: Code Explanation (No RAG)
    # ──────────────────────────────────────────

    def _handle_explain(self, user_message: str) -> dict:
        system_prompt = """You are an expert programming tutor and code analyst.
Your task is to explain code clearly, thoroughly, and in a way that helps the user understand.

Guidelines:
- Break down the code step by step
- Explain the purpose of each section
- Highlight any important patterns or techniques used
- Point out potential issues or improvements if relevant
- Use clear, accessible language
- If the user asks about specific aspects (e.g., time complexity), focus on those
- Format your response with Markdown for readability
- Use code blocks with syntax highlighting when referencing code
"""

        # Build the full prompt with conversation context
        context = self.memory.get_context_string()
        preferences = self.memory.get_preferences_string()

        full_prompt = (
            f"Conversation History:\n{context}\n\n"
            f"User Preferences:\n{preferences}\n\n"
            f"User Request:\n{user_message}"
        )

        try:
            llm = _get_llm()

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{full_prompt}"),
            ])

            chain = prompt | llm | StrOutputParser()
            response_text = chain.invoke({"full_prompt": full_prompt})

            return {
                "intent": "EXPLAIN",
                "response": response_text,
                "retrieved_docs": [],
                "is_relevant": None,
                "needs_feedback": False,
            }

        except Exception as e:
            return {
                "intent": "EXPLAIN",
                "response": f"⚠️ An error occurred while explaining: {str(e)}",
                "retrieved_docs": [],
                "is_relevant": None,
                "needs_feedback": False,
            }

    # ──────────────────────────────────────────
    # Route 2: Code Generation (With RAG)
    # ──────────────────────────────────────────

    def _handle_generate(self, user_message: str) -> dict:
        # Step 1: Retrieve from vector database
        retrieved_docs = self.rag_pipeline.retrieve(user_message)

        # Step 2: Check relevance of retrieved context
        is_relevant = self.rag_pipeline.check_relevance(user_message, retrieved_docs)

        if is_relevant and retrieved_docs:
            # Relevant context found → generate with RAG
            return self._generate_with_context(
                user_message, retrieved_docs, is_relevant
            )
        else:
            # No relevant context → ask user for help
            return {
                "intent": "GENERATE",
                "response": (
                    "🔍 I couldn't find sufficiently relevant knowledge in my "
                    "database for your request.\n\n"
                    "**Could you provide the correct solution or a reference "
                    "implementation?** I'll learn it and use it for future "
                    "interactions.\n\n"
                    "Simply paste the code or describe the approach, and I'll "
                    "store it in my knowledge base."
                ),
                "retrieved_docs": retrieved_docs,
                "is_relevant": False,
                "needs_feedback": True,
            }

    def _generate_with_context(
        self, user_message: str, retrieved_docs: list, is_relevant: bool
    ) -> dict:
        rag_context = self.rag_pipeline.get_context_string(retrieved_docs)

        system_prompt = """You are an expert software engineer and code generator.
Your task is to generate high-quality, production-ready code based on the user's request.

You have access to relevant reference code from a knowledge base. Use it as inspiration
and guidance, but always generate original, well-structured code.

Guidelines:
- Write complete, runnable source code
- Include detailed comments explaining the logic
- Follow best practices and design patterns
- Add proper error handling
- Include type hints where applicable
- Format code in Markdown code blocks with the correct language tag
- If relevant, explain your approach briefly after the code
- Consider edge cases and input validation
"""

        # Build the full prompt
        context = self.memory.get_context_string()
        preferences = self.memory.get_preferences_string()

        full_prompt = (
            f"Conversation History:\n{context}\n\n"
            f"User Preferences:\n{preferences}\n\n"
            f"Retrieved Knowledge Base Context:\n{rag_context}\n\n"
            f"User Request:\n{user_message}"
        )

        try:
            llm = _get_llm()

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{full_prompt}"),
            ])

            chain = prompt | llm | StrOutputParser()
            response_text = chain.invoke({"full_prompt": full_prompt})

            return {
                "intent": "GENERATE",
                "response": response_text,
                "retrieved_docs": retrieved_docs,
                "is_relevant": is_relevant,
                "needs_feedback": False,
            }

        except Exception as e:
            return {
                "intent": "GENERATE",
                "response": f"⚠️ An error occurred during generation: {str(e)}",
                "retrieved_docs": retrieved_docs,
                "is_relevant": is_relevant,
                "needs_feedback": False,
            }

    # ──────────────────────────────────────────
    # Route 3: User Feedback Learning
    # ──────────────────────────────────────────

    def _handle_feedback(self, user_message: str) -> dict:
        try:
            # Store the user's solution in the vector database
            metadata = {
                "source": "user_feedback",
                "type": "learned_solution",
            }

            doc_id = self.vector_db.add_document(
                content=user_message,
                metadata=metadata,
            )

            return {
                "intent": "FEEDBACK",
                "response": (
                    "✅ **Thank you!** I've learned from your input.\n\n"
                    f"📚 **Document stored** with ID: `{doc_id}`\n\n"
                    "I've:\n"
                    "1. Generated embeddings for your solution\n"
                    "2. Created metadata tags\n"
                    "3. Stored it in my knowledge base\n\n"
                    "I'll use this knowledge in future interactions. "
                    "Feel free to ask me anything else!"
                ),
                "retrieved_docs": [],
                "is_relevant": None,
                "needs_feedback": False,
            }

        except Exception as e:
            return {
                "intent": "FEEDBACK",
                "response": (
                    f"⚠️ I encountered an error while learning: {str(e)}\n\n"
                    "Please try again."
                ),
                "retrieved_docs": [],
                "is_relevant": None,
                "needs_feedback": False,
            }
