import streamlit as st
import os
import qdrant_client
from llama_index.core import (
    Settings, 
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    ChatPromptTemplate
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.llms import ChatMessage, MessageRole

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="GitaGPT", page_icon="🕉️")

# Spiritual Saffron Styling
st.markdown("""
    <style>
    .stApp { background-color: #FFF9F0; }
    h1 { color: #FF9933; text-align: center; font-family: 'Georgia', serif; }
    .stButton>button { background-color: #FF9933; color: white; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.title("🕉️ GitaGPT: Divine Guidance")

# --- 2. AUTHENTICATION ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Please add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

# --- 3. LAZY INITIALIZATION FUNCTION ---
@st.cache_resource(show_spinner="Connecting to the Divine Wisdom... (This may take a minute)")
def initialize_gita():
    # Setup Models
    llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Load Data
    if not os.path.exists("./data"):
        os.makedirs("./data")
    documents = SimpleDirectoryReader("./data").load_data()
    
    # Persistent Vector Store (More stable for Cloud)
    client = qdrant_client.QdrantClient(path="./qdrant_db")
    vector_store = QdrantVectorStore(client=client, collection_name="gita_collection")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# --- 4. STARTUP LOGIC (Prevents Health Check Timeout) ---
if "initialized" not in st.session_state:
    st.info("Welcome! Please click below to initialize the Gita Wisdom engine.")
    if st.button("🙏 Begin Discourse"):
        st.session_state.index = initialize_gita()
        st.session_state.initialized = True
        st.rerun()
    st.stop() # Stops execution here until button is clicked

# --- 5. CHAT ENGINE SETUP ---
index = st.session_state.index
qa_msgs = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are Lord Krishna. Provide wise, direct guidance in one short paragraph."),
    ChatMessage(role=MessageRole.USER, content="Context:\n{context_str}\n---\nQuery: {query_str}")
]
qa_prompt = ChatPromptTemplate(qa_msgs)
query_engine = index.as_query_engine(text_qa_template=qa_prompt, streaming=True)

# --- 6. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Lord Krishna..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_stream = query_engine.query(prompt)
        # Typewriter effect for a better user experience
        full_response = st.write_stream(response_stream.response_gen)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
