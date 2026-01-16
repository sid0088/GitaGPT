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

st.title("🕉️ GitaGPT — Bhagavad Gita Only")

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
# LLM (very low temperature to reduce invention)
# -------------------------------------------------
llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.1,
)

# -------------------------------------------------
# STRICT GITA-ONLY SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = (
    "You are Krishna in the Bhagavad Gita. "
    "Answer ONLY using teachings that explicitly exist in the Bhagavad Gita. "
    "Base responses strictly on dharma, karma yoga, jnana, bhakti, detachment from results, "
    "self-control of mind and senses, and equanimity as taught in the Gita. "
    "DO NOT add ideas from modern psychology, self-help, or other scriptures. "
    "DO NOT invent verses, examples, or explanations not present in the Gita. "
    "If the Bhagavad Gita does not directly address the question, say clearly: "
    "'The Bhagavad Gita does not give direct guidance on this.' "
    "Do not guess or extrapolate beyond the text. "
    "Do not mention chapter or verse numbers. "
    "Respond in ONE short paragraph of at most 2 sentences. "
    "Be precise, literal, and faithful to the Gita."
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
prompt = st.chat_input("Ask a question related to the Bhagavad Gita…")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert messages
    chat_msgs = []
    for m in st.session_state.messages:
        role = (
            MessageRole.SYSTEM if m["role"] == "system"
            else MessageRole.USER if m["role"] == "user"
            else MessageRole.ASSISTANT
        )
        chat_msgs.append(ChatMessage(role=role, content=m["content"]))

    # -------------------------------------------------
    # Streaming with single clean paragraph
    # -------------------------------------------------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_text = ""

        stream = llm.stream_chat(chat_msgs)
        for token in stream:
            if token.delta:
                response_text += token.delta
                placeholder.markdown(response_text)

    response_text = response_text.strip()

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )
