from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
)

CLASSIFIER_PROMPT = """You are an intent classifier for a coding assistant.
Your ONLY job is to classify the user's message into exactly one of two categories:

1. **EXPLAIN** — The user wants to understand, analyze, or get an explanation of existing code.
   Examples:
   - "Explain this code"
   - "What does this function do?"
   - "Why is this loop incorrect?"
   - "Explain line by line"
   - "What is the time complexity of this?"
   - "Can you break down this algorithm?"

2. **GENERATE** — The user wants new code to be written, created, or generated.
   Examples:
   - "Write Python code to sort a list"
   - "Generate a CNN model"
   - "Build a Flask API"
   - "Create a sorting algorithm"
   - "Write a function that calculates factorial"
   - "Implement binary search"

3. **FEEDBACK** — The user is providing a correct solution or reference code for the assistant to learn.
   Examples:
   - "Here is the correct implementation..."
   - "The right way to do it is..."
   - "Learn this: ..."
   - "Store this solution..."

Rules:
- Respond with ONLY one word: EXPLAIN, GENERATE, or FEEDBACK.
- Do NOT add any explanation or extra text.
- If the message is ambiguous, lean toward GENERATE.
- If the user provides code AND asks a question about it, classify as EXPLAIN.
- If the user provides code as a correction/solution to learn from, classify as FEEDBACK.
"""


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0,           # Deterministic for classification
        max_tokens=10,           # Only need a single word
    )


def classify_intent(user_message: str) -> str:
    try:
        llm = _get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system", CLASSIFIER_PROMPT),
            ("human", "{user_message}"),
        ])

        classifier_chain = prompt | llm | StrOutputParser()
        classification = classifier_chain.invoke({"user_message": user_message}).strip().upper()

        # Validate that the response is one of the expected labels
        if classification in ("EXPLAIN", "GENERATE", "FEEDBACK"):
            return classification

        # Fallback: try to find the keyword in the response
        if "EXPLAIN" in classification:
            return "EXPLAIN"
        elif "FEEDBACK" in classification:
            return "FEEDBACK"
        else:
            return "GENERATE"

    except Exception as e:
        print(f"[Classifier Error] {e}")
        return "GENERATE"
