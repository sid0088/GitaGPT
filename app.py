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
# LLM (temperature kept low for focus)
# -------------------------------------------------
llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.2,
)

# 🔒 STRICT SYSTEM PROMPT
SYSTEM_PROMPT = (
    "You are Lord Krishna. "
    "Give a DIRECT answer in ONE short paragraph. "
    "Use at most 2–3 sentences. "
    "Be precise, practical, and calm. "
    "Do not explain unnecessarily. "
    "Do not use bullet points or line breaks."
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
    # Streaming → single short paragraph
    # -------------------------------------------------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_text = ""

        stream = llm.stream_chat(chat_msgs)
        for token in stream:
            if token.delta:
                response_text += token.delta
                placeholder.markdown(response_text)

    # Final trim (safety)
    response_text = response_text.strip()

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )
