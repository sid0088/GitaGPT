import streamlit as st
import os
import qdrant_client
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext, ChatPromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.llms import ChatMessage, MessageRole

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(page_title="GitaGPT", page_icon="🕉️", layout="centered")

# Custom Saffron & Cream Spiritual Theme
st.markdown("""
    <style>
    .stApp { background-color: #FFF9F0; }
    h1 { color: #FF9933; text-align: center; font-family: 'Georgia', serif; }
    .stChatMessage { border-radius: 15px; border: 1px solid #FF9933; }
    </style>
    """, unsafe_allow_html=True)

st.title("🕉️ GitaGPT: Divine Guidance")
st.markdown("---")

# --- 2. AUTHENTICATION & DIRECTORY ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Please add GROQ_API_KEY to Streamlit Secrets dashboard.")
    st.stop()

# Ensure data directory exists
if not os.path.exists("./data"):
    os.makedirs("./data")
    st.info("Created 'data' folder. Please ensure your PDF is uploaded there on GitHub.")

# --- 3. INITIALIZE MODELS (Lazy Loading) ---
@st.cache_resource(show_spinner="Connecting to the Divine...")
def load_gita_index():
    # Model Setup
    llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Load and Index Data
    documents = SimpleDirectoryReader("./data").load_data()
    
    # Using local path for Qdrant to improve stability over :memory:
    client = qdrant_client.QdrantClient(path="./qdrant_db")
    vector_store = QdrantVectorStore(client=client, collection_name="gita_collection")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Only start indexing when the app is ready
try:
    index = load_gita_index()
except Exception as e:
    st.error(f"Error initializing AI: {e}")
    st.stop()

# --- 4. DEFINE PERSONA ---
qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM, 
        content="You are Lord Krishna. Provide brief, high-impact wisdom in one short paragraph. If it is not in the Gita, say so."
    ),
    ChatMessage(
        role=MessageRole.USER, 
        content="Context:\n{context_str}\n---\nStructure: [Insight] + [Short Explanation] + [Verse Citation].\nQuery: {query_str}"
    )
]
qa_prompt = ChatPromptTemplate(qa_msgs)
query_engine = index.as_query_engine(text_qa_template=qa_prompt)

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Namaste. I am Krishna. How may I guide you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about life, duty, or peace..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = query_engine.query(prompt)
        answer = response.response
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
