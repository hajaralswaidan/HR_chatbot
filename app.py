import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------- Project imports --------------------
from src.llm.sql_agent import answer_with_sql
from src.llm.cloud_groq import CloudGroq
from src.llm.local_qwen import LocalQwen


# -------------------- Page config --------------------
st.set_page_config(
    page_title="HR Analytics Chatbot",
    layout="wide",
)

# -------------------- CSS --------------------
def load_css(path="styles.css"):
    try:
        with open(path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()


# -------------------- Session State --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_mode" not in st.session_state:
    st.session_state.model_mode = "Cloud"  # Local / Cloud

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False


# -------------------- Cached models --------------------
@st.cache_resource
def get_cloud_llm():
    return CloudGroq()

@st.cache_resource
def get_local_llm():
    # Local Qwen 1.5B 
    return LocalQwen()  


def llm_call(prompt: str) -> str:
    """
    Unified LLM router for the SQL agent:
    - Cloud -> Groq
    - Local -> Qwen (1.5B)
    """
    if st.session_state.model_mode == "Cloud":
        return get_cloud_llm().generate(prompt)
    return get_local_llm().generate(prompt)


# -------------------- Helpers --------------------
def clear_chat():
    st.session_state.messages = []
    st.session_state.is_generating = False


def history_as_text(max_turns: int = 10) -> str:
    msgs = st.session_state.messages[-max_turns:]
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)


# -------------------- Sidebar --------------------
with st.sidebar:
      
     # Logo centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", width=110)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.header("Chat Settings")

    st.session_state.model_mode = st.radio(
        "Model",
        ["Local", "Cloud"],
        horizontal=True,
        index=0 if st.session_state.model_mode == "Local" else 1,
    )

    if st.session_state.model_mode == "Cloud":
        st.success("Cloud ready (Groq)")
    else:
        st.info("Local ready (Qwen 1.5B)")

    if st.button("Clear Chat", use_container_width=True):
        clear_chat()
        st.rerun()


    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#777;'> Developed by Hajar</p>",
        unsafe_allow_html=True,
    )


# -------------------- Main UI --------------------
st.title("HR Analytics Chatbot")


if not st.session_state.messages:
    st.markdown("""
    <div style="color:#666; font-size:16px; line-height:1.7; max-width:800px; border-left: 4px solid #7c3aed; padding-left: 15px;">
        <strong>Welcome! I am your AI HR Assistant.</strong><br>
        Ask me questions about employee statistics, attrition rates, or departmental insights in plain English or Arabic.<br>
        <hr style="margin: 10px 0; border: 0; border-top: 1px solid #eee;">
        <strong>مرحباً بك! أنا مساعدك الذكي لتحليل الموارد البشرية.</strong><br>
        يمكنك سؤالي عن إحصائيات الموظفين، نسب الاستقالات، أو تفاصيل الأقسام باللغة العربية أو الإنجليزية.
    </div>
    """, unsafe_allow_html=True)


# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("اسأل سؤال / Ask a question...")


# -------------------- Chat Logic --------------------
if user_input and not st.session_state.is_generating:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.is_generating = True

    with st.chat_message("assistant"):
        with st.spinner("جاري التحليل..."):
            history_text = history_as_text()

            answer, evidence, error = answer_with_sql(
                question=user_input,
                history_text=history_text,
                llm_call=llm_call,
                max_rows=20,
            )

        if error:
            st.error(error)
            assistant_text = error
        else:
            assistant_text = answer or "I can’t answer this from the HR dataset."

        st.markdown(assistant_text)

        # Evidence Expander: show SQL + first rows
        if evidence and isinstance(evidence, dict):
            with st.expander("Show SQL (Evidence)"):
                sql_txt = evidence.get("sql", "")
                if sql_txt:
                    st.code(sql_txt, language="sql")
                rows = evidence.get("rows", [])
                if rows:
                    st.write(rows)

        st.session_state.messages.append({"role": "assistant", "content": assistant_text})

    st.session_state.is_generating = False
    st.rerun()
