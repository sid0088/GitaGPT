import streamlit as st
import os
import qdrant_client

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ChatPromptTemplate,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.llms import ChatMessage, MessageRole

# -------------------------------------------------
# 1. PAGE CONFIG
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
# 2. AUTHENTICATION (STREAMLIT SAFE)
# -------------------------------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("❌ Please add GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# -------------------------------------------------
# 3. INITIALIZATION FUNCTION (CACHED)
# -------------------------------------------------
@st.cache_resource(show_spinner="📿 Awakening the Gita’s wisdom...")
def initialize_gita_index():
    # LLM
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
    )

    # Embeddings
    embed_model = FastEmbedEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    # Ensure data exists
    data_path = "./data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    documents = SimpleDirectoryReader(data_path).load_data()

    if len(documents) == 0:
        raise RuntimeError(
            "❌ No documents found in ./data. "
            "Please upload Gita text files before deploying."
        )

    # ✅ Streamlit-safe in-memory Qdrant
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
# 4. LAZY START (NO HEALTH-CHECK FAILURE)
# -------------------------------------------------
if "index" not in st.session_state:
    st.info("🙏 Click below to begin the sacred discourse.")
    if st.button("Begin Discourse"):
        try:
            st.session_state.index = initialize_gita_index()
            st.success("✨ GitaGPT is ready")
            st.rerun()
        except Exception as e:
            st.error(str(e))
    st.stop()

index = st.session_state.index

# -------------------------------------------------
# 5. QUERY ENGINE
# -------------------------------------------------
qa_prompt = ChatPromptTemplate(
    [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are Lord Krishna. "
                "Answer with calm wisdom, clarity, and compassion. "
                "Respond in one short paragraph."
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
# 6. CHAT UI
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
        full_reply = st.write_stream(response.response_gen)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_reply}
    )
