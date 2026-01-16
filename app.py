import os
import sys
import asyncio
import streamlit as st

from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage, MessageRole

# -------------------------------------------------
# 🔒 Async safety (Streamlit health-check safe)
# -------------------------------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -------------------------------------------------
# 🖥️ PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="GitaGPT",
    page_icon="🕉️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #FFF9F0; }
    h1 { color: #FF9933; text-align: center; font-family: 'Georgia', serif; }
    .stButton>button { background-color: #FF9933; color: white; width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🕉️ GitaGPT: Divine Guidance")

# -------------------------------------------------
# 🔍 DEBUG
# -------------------------------------------------
st.sidebar.write("Python version:")
st.sidebar.code(sys.version)

# -------------------------------------------------
# 🔐 GROQ API KEY
# -------------------------------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# -------------------------------------------------
# 🤖 LLM (NO EMBEDDINGS)
# -------------------------------------------------
llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
)

SYSTEM_PROMPT = (
    "You are Lord Krishna. "
    "Respond with calm, wisdom, clarity, and compassion. "
    "Answer in one short paragraph."
)

# -------------------------------------------------
# 💬 CHAT STATE
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# Render messages
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# -------------------------------------------------
# 💭 USER INPUT
# -------------------------------------------------
user_prompt = st.chat_input("Ask Lord Krishna…")

if user_prompt:
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Convert to Groq messages
    chat_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM if m["role"] == "system" else MessageRole.USER
            if m["role"] == "user"
            else MessageRole.ASSISTANT,
            content=m["content"],
        )
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        response = llm.chat(chat_messages)
        st.markdown(response.message.content)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.message.content}
    )
