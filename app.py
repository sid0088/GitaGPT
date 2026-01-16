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
    "Answer calmly and wisely in ONE clear, well-structured paragraph. "
    "Do not use bullet points, line breaks, or lists."
)

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# Render chat history
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
    # STREAMING → SINGLE PARAGRAPH OUTPUT
    # -------------------------------------------------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_text = ""

        stream = llm.stream_chat(chat_msgs)
        for token in stream:
            if token.delta:
                response_text += token.delta
                placeholder.markdown(response_text)

    # Save final response
    st.session_state.messages.append(
        {"role": "assistant", "content": response_text.strip()}
    )
