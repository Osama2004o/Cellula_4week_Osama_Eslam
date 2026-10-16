# ============================================
# AI Coding Assistant — Streamlit Application
# ============================================
# The main entry point for the application.
# Provides a rich, interactive chat interface
# with code generation, explanation, execution,
# and knowledge-base learning capabilities.
# ============================================

import streamlit as st
import time

from config.settings import OPENROUTER_API_KEY
from core.classifier import classify_intent
from core.router import Router
from core.memory import ConversationMemory
from database.vector_db import VectorDatabase
from tools.code_executor import execute_code, extract_code_blocks


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Coding Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Custom CSS for Premium Look & Feel
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
    /* ── Global Styles ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header Gradient ── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.85);
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* ── Intent Badge ── */
    .intent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .intent-explain {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: #1a3a2a;
    }
    .intent-generate {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .intent-feedback {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }

    /* ── Status Cards ── */
    .status-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }

    /* ── Execution Result Box ── */
    .exec-success {
        background: #f0fff4;
        border: 1px solid #c6f6d5;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .exec-error {
        background: #fff5f5;
        border: 1px solid #fed7d7;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* ── Sidebar Styling ── */
    .sidebar-stat {
        background: rgba(102, 126, 234, 0.1);
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    .sidebar-stat h4 {
        margin: 0;
        color: #667eea;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .sidebar-stat p {
        margin: 0.25rem 0 0 0;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* ── Pipeline Visualization ── */
    .pipeline-step {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        font-size: 0.85rem;
    }
    .pipeline-step .step-icon {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 0.5rem;
        font-size: 0.7rem;
        font-weight: 700;
        color: white;
    }
    .step-active { background: #667eea; }
    .step-done { background: #43e97b; }
    .step-pending { background: #cbd5e0; }

    /* ── Divider ── */
    .styled-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 1rem 0;
        border: none;
    }

    /* ── Scrollable Chat ── */
    [data-testid="stChatMessageContent"] {
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stChatMessageContent"] code {
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    if "router" not in st.session_state:
        st.session_state.router = None

    if "db_seeded" not in st.session_state:
        st.session_state.db_seeded = False

    if "processing" not in st.session_state:
        st.session_state.processing = False

    if "last_intent" not in st.session_state:
        st.session_state.last_intent = None

    if "pipeline_log" not in st.session_state:
        st.session_state.pipeline_log = []


init_session_state()


# ──────────────────────────────────────────────
# System Initialization
# ──────────────────────────────────────────────
@st.cache_resource
def initialize_system():
    vdb = VectorDatabase()
    memory = ConversationMemory()
    router = Router(vector_db=vdb, memory=memory)
    return vdb, router


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(
    """
<div class="main-header">
    <h1>🤖 AI Coding Assistant</h1>
    <p>Intelligent code generation &amp; explanation powered by RAG + LangChain + OpenRouter</p>
</div>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # API Key status
    if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_openrouter_api_key_here":
        st.success("🔑 API Key Configured")
        api_ready = True
    else:
        st.error("🔑 API Key Missing")
        api_key_input = st.text_input(
            "Enter your OpenRouter API Key:",
            type="password",
            help="Get your key from https://openrouter.ai/keys",
        )
        if api_key_input:
            import os
            os.environ["OPENROUTER_API_KEY"] = api_key_input
            # Update the settings module so all components pick up the key
            import config.settings as _cfg
            _cfg.OPENROUTER_API_KEY = api_key_input
            st.success("🔑 API Key Set!")
            api_ready = True
        else:
            api_ready = False

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Database controls
    st.markdown("### 📚 Knowledge Base")

    if api_ready:
        # Initialize system
        if not st.session_state.initialized:
            with st.spinner("🔧 Initializing system..."):
                vdb, router = initialize_system()
                st.session_state.vector_db = vdb
                st.session_state.router = router
                st.session_state.memory = router.memory
                st.session_state.initialized = True

        # Show DB stats
        if st.session_state.vector_db:
            stats = st.session_state.vector_db.get_stats()
            st.markdown(
                f"""
            <div class="sidebar-stat">
                <h4>📊 Documents</h4>
                <p>{stats['document_count']}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Seed button
            if stats["document_count"] == 0:
                if st.button("🌱 Seed Knowledge Base", use_container_width=True):
                    with st.spinner("📥 Loading HumanEval dataset..."):
                        count = st.session_state.vector_db.seed_from_humaneval()
                        st.session_state.db_seeded = True
                    st.success(f"✅ Loaded {count} documents!")
                    st.rerun()
            else:
                st.session_state.db_seeded = True
                st.markdown("✅ Knowledge base ready")

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Memory visualization
    st.markdown("### 🧠 Memory")
    if st.session_state.memory:
        mem_summary = st.session_state.memory.get_summary()

        st.markdown(
            f"""
        <div class="sidebar-stat">
            <h4>💬 Conversation Turns</h4>
            <p>{mem_summary['total_turns']}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        prefs = mem_summary["preferences"]
        if prefs.get("language") or prefs.get("framework"):
            pref_text = ""
            if prefs.get("language"):
                pref_text += f"🔤 Language: **{prefs['language']}**\n\n"
            if prefs.get("framework"):
                pref_text += f"🏗️ Framework: **{prefs['framework']}**"
            st.markdown(pref_text)

        if st.button("🗑️ Clear Memory", use_container_width=True):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.session_state.pipeline_log = []
            st.rerun()

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Pipeline Log
    st.markdown("### 🔄 Pipeline Log")
    if st.session_state.pipeline_log:
        for log_entry in st.session_state.pipeline_log[-5:]:
            intent_class = f"intent-{log_entry['intent'].lower()}"
            st.markdown(
                f'<span class="intent-badge {intent_class}">'
                f"{log_entry['intent']}</span> "
                f"<small>{log_entry['query'][:40]}...</small>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("_No pipeline activity yet._")

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # File upload (optional)
    st.markdown("### 📎 Upload Code File")
    uploaded_file = st.file_uploader(
        "Upload a code file to analyze",
        type=["py", "js", "ts", "java", "cpp", "c", "rb", "go", "rs"],
        help="Upload a source code file to get explanations or improvements.",
    )
    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        st.code(file_content[:500] + ("..." if len(file_content) > 500 else ""), language="python")
        if st.button("📤 Analyze This File", use_container_width=True):
            # Add file content as a user message
            analysis_msg = f"Please explain this code:\n\n```python\n{file_content}\n```"
            st.session_state.messages.append(
                {"role": "user", "content": analysis_msg}
            )
            st.rerun()


# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "intent" in msg:
            intent_class = f"intent-{msg['intent'].lower()}"
            st.markdown(
                f'<span class="intent-badge {intent_class}">'
                f"{msg['intent']}</span>",
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])

        # Show code execution buttons for assistant messages
        if msg["role"] == "assistant" and "code_blocks" in msg and msg["code_blocks"]:
            for i, code_block in enumerate(msg["code_blocks"]):
                exec_key = f"exec_{msg.get('timestamp', '')}_{i}"
                with st.expander(f"📝 Code Block {i + 1}", expanded=False):
                    st.code(code_block, language="python")
                    if st.button(f"▶️ Run Code", key=exec_key):
                        with st.spinner("⚙️ Executing code..."):
                            result = execute_code(code_block)
                        if result["success"]:
                            st.markdown(
                                f'<div class="exec-success">'
                                f"<strong>✅ Output:</strong><br>"
                                f"<pre>{result['stdout']}</pre>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            error_msg = result["error"] or result["stderr"]
                            st.markdown(
                                f'<div class="exec-error">'
                                f"<strong>❌ Error:</strong><br>"
                                f"<pre>{error_msg}</pre>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

# Welcome message
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            """
👋 **Welcome to the AI Coding Assistant!**

I can help you with:

| Feature | Description |
|---------|-------------|
| 💡 **Explain Code** | Paste any code and I'll break it down step by step |
| ⚡ **Generate Code** | Describe what you need and I'll write it for you |
| ▶️ **Execute Code** | Run generated code directly in the browser |
| 📚 **Learn & Grow** | Teach me new solutions to expand my knowledge base |

**How it works:**
1. 🔍 I **classify** your intent (explain vs. generate)
2. 🔀 I **route** your request to the right pipeline
3. 📖 For code generation, I **search** my knowledge base (RAG)
4. ✅ I **verify** the relevance of retrieved context
5. 🎯 I generate a **high-quality** response

_Try asking me to write a function or explain some code!_
"""
        )


# ──────────────────────────────────────────────
# Chat Input Handler
# ──────────────────────────────────────────────
if prompt := st.chat_input("Ask me to write code, explain code, or teach me something..."):
    if not api_ready:
        st.error("Please configure your OpenRouter API key in the sidebar first.")
    elif not st.session_state.initialized:
        st.error("System is still initializing. Please wait...")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process through the router
        with st.chat_message("assistant"):
            # Show pipeline progress
            status_container = st.empty()

            # Step 1: Classification
            status_container.markdown("🔍 **Classifying intent...**")
            time.sleep(0.3)

            # Step 2: Route and process
            with st.spinner(""):
                # Use the router's memory instead of session state memory
                st.session_state.router.memory = st.session_state.memory
                result = st.session_state.router.process(prompt)

            # Update pipeline log
            st.session_state.pipeline_log.append(
                {
                    "intent": result["intent"],
                    "query": prompt,
                    "is_relevant": result.get("is_relevant"),
                    "timestamp": time.time(),
                }
            )

            # Clear the status
            status_container.empty()

            # Show intent badge
            intent_class = f"intent-{result['intent'].lower()}"
            st.markdown(
                f'<span class="intent-badge {intent_class}">'
                f"{result['intent']}</span>",
                unsafe_allow_html=True,
            )

            # Show RAG info for GENERATE intent
            if result["intent"] == "GENERATE" and result.get("retrieved_docs"):
                with st.expander("📖 Retrieved Knowledge Base Context", expanded=False):
                    for i, doc in enumerate(result["retrieved_docs"], 1):
                        source = doc["metadata"].get("source", "Unknown")
                        task_id = doc["metadata"].get("task_id", "N/A")
                        distance = doc.get("distance", "N/A")
                        st.markdown(
                            f"**Doc {i}** | Source: `{source}` | "
                            f"Task: `{task_id}` | "
                            f"Relevance Score: `{1 - distance:.2f}` "
                            if isinstance(distance, (int, float))
                            else f"**Doc {i}** | Source: `{source}`"
                        )
                        st.code(doc["document"][:300], language="python")

                relevance_status = (
                    "✅ Context is **relevant**"
                    if result.get("is_relevant")
                    else "❌ Context is **not relevant**"
                )
                st.markdown(relevance_status)

            # Stream the response
            response_text = result["response"]
            st.markdown(response_text)

            # Extract and store code blocks for execution
            code_blocks = extract_code_blocks(response_text)

            # Store the message
            msg_data = {
                "role": "assistant",
                "content": response_text,
                "intent": result["intent"],
                "code_blocks": code_blocks,
                "timestamp": str(time.time()),
            }
            st.session_state.messages.append(msg_data)

            # Show execution buttons for code blocks
            if code_blocks:
                st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
                st.markdown("**▶️ Run Generated Code:**")
                for i, code_block in enumerate(code_blocks):
                    with st.expander(f"📝 Code Block {i + 1}", expanded=True):
                        st.code(code_block, language="python")
                        if st.button(f"▶️ Execute", key=f"run_{time.time()}_{i}"):
                            with st.spinner("⚙️ Running..."):
                                exec_result = execute_code(code_block)
                            if exec_result["success"]:
                                st.success("✅ Code executed successfully!")
                                if exec_result["stdout"]:
                                    st.markdown("**Output:**")
                                    st.code(exec_result["stdout"], language="text")
                            else:
                                st.error("❌ Execution failed!")
                                error_msg = exec_result["error"] or exec_result["stderr"]
                                st.code(error_msg, language="text")

        # Rerun to update sidebar stats
        st.rerun()
