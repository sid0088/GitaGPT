import os
import sys
import asyncio
import streamlit as st
import qdrant_client

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ChatPromptTemplate,
)
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.llms import ChatMessage, MessageRole

# 🔑 OpenAI-compatible embeddings via Groq
from llama_index.embeddings.openai import OpenAIEmbedding

# -------------------------------------------------
# 🔒 Async safety for Streamlit health checks
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
# 🔍 DEBUG (remove later if you want)
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
# 🧠 LAZY INITIALIZATION (CACHED)
# -------------------------------------------------
@st.cache_resource(show_spinner="📿 Awakening the wisdom of the Gita...")
def initialize_index():
    # --- LLM ---
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
    )

    # --- EMBEDDINGS (Groq OpenAI-compatible) ---
    embed_model = OpenAILikeEmbedding(
        model="text-embedding-3-small",
        api_base="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    # --- LOAD DOCUMENTS ---
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    documents = SimpleDirectoryReader(data_dir).load_data()

    if not documents:
        raise RuntimeError(
            "❌ No documents found in ./data. "
            "Please add Bhagavad Gita PDFs or text files."
        )

    # --- VECTOR STORE (IN-MEMORY, STREAMLIT SAFE) ---
    qdrant = qdrant_client.QdrantClient(":memory:")

    vector_store = QdrantVectorStore(
        client=qdrant,
        collection_name="gita_collection",
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

# -------------------------------------------------
# 🚀 STARTUP GATE (HEALTH-CHECK SAFE)
# -------------------------------------------------
if "index" not in st.session_state:
    st.info("🙏 Click below to begin the sacred discourse.")
    if st.button("Begin Discourse"):
        try:
            st.session_state.index = initialize_index()
            st.success("✨ GitaGPT is ready")
            st.rerun()
        except Exception as e:
            st.error(str(e))
    st.stop()

index = st.session_state.index

# -------------------------------------------------
# 🧾 QUERY ENGINE
# -------------------------------------------------
qa_prompt = ChatPromptTemplate(
    [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are Lord Krishna. "
                "Give calm, clear, compassionate guidance. "
                "Answer in one short paragraph."
            ),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="Context:\n{context_str}\n---\nQuestion: {query_str}",
        ),
    ]
)

query_engine = index.as_query_engine(
    text_qa_template=qa_prompt,
    streaming=True,
)

# -------------------------------------------------
# 💬 CHAT UI
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("Ask Lord Krishna…")

if user_prompt:
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response = query_engine.query(user_prompt)
        full_response = st.write_stream(response.response_gen)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
