import streamlit as st

# Page ------------------------
st.set_page_config(
    page_title="HR Analytics Chatbot",
    layout="wide"
)

# Initialize session state (added)
if "model_mode" not in st.session_state:
    st.session_state.model_mode = "Local"

# Load CSS-----------------------
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")


# ---  Welcome messages (Gray) ---
WELCOME_MSG = """
<div style="color:#999; font-size:14px; line-height:1.6;">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
    <span style="font-size:18px;"></span>
    <b style="color:#777;">Welcome to the HR Analytics Chatbot.</b>
  </div>

  You can ask questions about the HR dataset, such as:
  <ul style="margin:8px 0 0 18px;">
    <li>headcount by department</li>
    <li>attrition trends</li>
    <li>salary and performance insights</li>
  </ul>
</div>
"""

CLEAR_MSG = """
<div style="color:#999; font-size:14px; line-height:1.6;">
  Chat cleared. Ask a new question about the HR dataset.
</div>
"""

def init_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MSG, "type": "system"}
        ]

def new_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MSG, "type": "system"}
    ]

def clear_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": CLEAR_MSG, "type": "system"}
    ]

def remove_system_if_chat_started():
    """
    Remove system messages (welcome/clear) as soon as user starts the conversation.
    """
    if any(m.get("role") == "user" for m in st.session_state.messages):
        st.session_state.messages = [
            m for m in st.session_state.messages if m.get("type") != "system"
        ]

init_chat()


# Sidebar----------------
with st.sidebar:

    st.header("Chat Settings")

    # Model mode switcher 
    st.subheader("Model Mode")
    st.session_state.model_mode = st.radio(
        "Select model type",
        ["Local", "Cloud"],
        index=0 if st.session_state.model_mode == "Local" else 1,
        horizontal=True
    )
    st.caption(f"Current mode: {st.session_state.model_mode}")

    st.write("Messages")
    st.subheader(len(st.session_state.messages))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("New chat", use_container_width=True):
            new_chat()
            st.rerun()

    with col2:
        if st.button("Clear chat", use_container_width=True):
            clear_chat()
            st.rerun()

    # Footer---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#666; font-weight:400; text-align:center;'>⭐ Developed by Hajar</p>",
        unsafe_allow_html=True
    )


# Main -------------------------
st.title("HR Analytics Chatbot")
st.caption("Ask questions about HR data.")

# Ensure  messages disappear after conversation starts
remove_system_if_chat_started()

# Show messages ------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Use markdown to allow gray messages
        st.markdown(msg["content"], unsafe_allow_html=True)


# Input  -------------------------
user_input = st.chat_input("Ask a question about the HR data...")

if user_input:
    # If first user message, remove any message (welcome/clear) before adding chat
    st.session_state.messages = [
        m for m in st.session_state.messages if m.get("type") != "system"
    ]

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # Temporary response------------
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "**HR Chatbot**\n\n"
                "Got it\n\n"
            )
        }
    )

    st.rerun()
