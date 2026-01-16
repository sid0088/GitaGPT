import streamlit as st
import os
import qdrant_client
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext, ChatPromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.llms import ChatMessage, MessageRole

# --- PAGE CONFIG ---
st.set_page_config(page_title="GitaGPT", page_icon="🕉️")
st.title("🕉️ GitaGPT: Divine Wisdom")

# --- 1. AUTHENTICATION (SECURE) ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Please add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

# --- 2. INITIALIZE MODELS & DATA ---
@st.cache_resource # This prevents the app from re-loading the PDF every time you chat
def initialize_gita():
    # Setup LLM & Embeddings
    llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Load Data (Ensure your PDF is in a folder named 'data')
    documents = SimpleDirectoryReader("./data").load_data()
    
    # Setup Vector Store
    client = qdrant_client.QdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(client=client, collection_name="gita_collection")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

index = initialize_gita()

# --- 3. DEFINE PERSONA ---
qa_msgs = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are Lord Krishna. Provide brief, high-impact wisdom (max 3 sentences)."),
    ChatMessage(role=MessageRole.USER, content="Context:\n{context_str}\n---\nStructure: [Insight] + [Explanation] + [Verse].\nQuery: {query_str}")
]
qa_prompt = ChatPromptTemplate(qa_msgs)
query_engine = index.as_query_engine(text_qa_template=qa_prompt)

# --- 4. CHAT INTERFACE ---
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
        response = query_engine.query(prompt)
        st.markdown(response.response)
        st.session_state.messages.append({"role": "assistant", "content": response.response})
