from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from config.settings import MAX_MEMORY_TURNS


class ConversationMemory:

    def __init__(self):
        self.history: list[dict] = []
        self.preferences: dict = {
            "language": None,
            "framework": None,
        }

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

        # Trim history if it exceeds the maximum window
        if len(self.history) > MAX_MEMORY_TURNS:
            self.history = self.history[-MAX_MEMORY_TURNS:]

        # Attempt to detect user preferences from user messages
        if role == "user":
            self._detect_preferences(content)

    def _detect_preferences(self, message: str) -> None:
        message_lower = message.lower()

        # Language detection
        languages = {
            "python": "Python",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "java": "Java",
            "c++": "C++",
            "cpp": "C++",
            "c#": "C#",
            "csharp": "C#",
            "rust": "Rust",
            "go": "Go",
            "ruby": "Ruby",
            "php": "PHP",
            "swift": "Swift",
            "kotlin": "Kotlin",
        }

        for keyword, language in languages.items():
            if keyword in message_lower:
                self.preferences["language"] = language
                break

        # Framework detection
        frameworks = {
            "flask": "Flask",
            "django": "Django",
            "fastapi": "FastAPI",
            "react": "React",
            "vue": "Vue",
            "angular": "Angular",
            "express": "Express",
            "spring": "Spring",
            "pytorch": "PyTorch",
            "tensorflow": "TensorFlow",
            "keras": "Keras",
            "streamlit": "Streamlit",
            "pandas": "Pandas",
            "numpy": "NumPy",
        }

        for keyword, framework in frameworks.items():
            if keyword in message_lower:
                self.preferences["framework"] = framework
                break

    def get_context_string(self) -> str:
        if not self.history:
            return "No previous conversation history."

        lines = []
        for turn in self.history:
            role_label = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {turn['content']}")

        return "\n".join(lines)

    def get_langchain_messages(self) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        for turn in self.history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
        return messages

    def get_preferences_string(self) -> str:
        parts = []
        if self.preferences["language"]:
            parts.append(f"Preferred Language: {self.preferences['language']}")
        if self.preferences["framework"]:
            parts.append(f"Preferred Framework: {self.preferences['framework']}")

        return "\n".join(parts) if parts else "No specific preferences detected."

    def clear(self) -> None:
        self.history.clear()
        self.preferences = {"language": None, "framework": None}

    def get_summary(self) -> dict:
        summary = {
            "total_turns": len(self.history),
            "preferences": self.preferences.copy(),
            "last_exchange": None,
        }

        if len(self.history) >= 2:
            summary["last_exchange"] = {
                "user": self.history[-2]["content"][:100] + "..."
                if len(self.history[-2]["content"]) > 100
                else self.history[-2]["content"],
                "assistant": self.history[-1]["content"][:100] + "..."
                if len(self.history[-1]["content"]) > 100
                else self.history[-1]["content"],
            }

        return summary

