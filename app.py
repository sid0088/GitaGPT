import os
import sys
import asyncio
import streamlit as st

from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage, MessageRole

# -------------------------------------------------
# Async safety (Streamlit health-check safe)
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

st.title("🕉️ GitaGPT: Guidance from the Bhagavad Gita")

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
# LLM (low temperature = clarity, not verbosity)
# -------------------------------------------------
llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.2,
)

# -------------------------------------------------
# GITA-FOCUSED SYSTEM PROMPT (CRITICAL PART)
# -------------------------------------------------
SYSTEM_PROMPT = (
    "You are Lord Krishna speaking to Arjuna. "
    "Answer every question strictly using the wisdom of the Bhagavad Gita. "
    "Base your guidance on dharma (right duty), karma yoga (selfless action), "
    "detachment from results, self-mastery, equanimity, and inner discipline. "
    "Do NOT give modern self-help advice. "
    "Do NOT mention verse numbers or chapters. "
    "Paraphrase Gita teachings in simple, practical language. "
    "Respond in ONE short paragraph of 2–3 sentences only. "
    "Be calm, direct, and compassionate. "
    "No bullet points, no line breaks, no unnecessary explanations."
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
    # Streaming → single concise Gita-based paragraph
    # -------------------------------------------------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_text = ""

        stream = llm.stream_chat(chat_msgs)
        for token in stream:
            if token.delta:
                response_text += token.delta
                placeholder.markdown(response_text)

    # Final trim
    response_text = response_text.strip()

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )
