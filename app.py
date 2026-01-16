import os
import sys
import asyncio
import streamlit as st

from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage, MessageRole

# -------------------------------------------------
# Async safety
# -------------------------------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="GitaGPT",
    page_icon="🕉️",
)

st.title("🕉️ GitaGPT: Divine Guidance")

st.sidebar.write("Python version:")
st.sidebar.code(sys.version)

# -------------------------------------------------
# API Key
# -------------------------------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY missing in Streamlit secrets")
    st.stop()

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# -------------------------------------------------
# LLM
# -------------------------------------------------
llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
)

SYSTEM_PROMPT = (
    "You are Lord Krishna. "
    "Respond with calm wisdom, clarity, and compassion. "
    "Keep answers concise and meaningful."
)

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# Render history
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# -------------------------------------------------
# User input
# -------------------------------------------------
prompt = st.chat_input("Ask Lord Krishna…")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert to Groq messages
    chat_msgs = []
    for m in st.session_state.messages:
        role = (
            MessageRole.SYSTEM if m["role"] == "system"
            else MessageRole.USER if m["role"] == "user"
            else MessageRole.ASSISTANT
        )
        chat_msgs.append(ChatMessage(role=role, content=m["content"]))

    # -------------------------------------------------
    # STREAMING RESPONSE (KEY FIX)
    # -------------------------------------------------
    with st.chat_message("assistant"):
        response_text = ""
        response = llm.stream_chat(chat_msgs)

        for token in response:
            if token.delta:
                response_text += token.delta
                st.write(response_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )
