#!/usr/bin/env python3
"""
RAG Application with Streamlit UI
Supports Ollama (local) and cloud LLMs (OpenAI, Gemini, Claude)
"""

import streamlit as st
import sys
import os
import subprocess

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from retriever import Retriever
from llm_handler import LLMHandler


# ============================================================
# INITIALIZATION FUNCTIONS
# ============================================================

def ensure_embeddings_exist():
    """Generate embeddings if they don't exist"""
    embedding_file = "data/embeddings.json"
    chunks_file = "data/chunks.json"
    
    if not os.path.exists(embedding_file) and os.path.exists(chunks_file):
        st.warning("⏳ Generating embeddings for first time... This may take 1-2 minutes")
        try:
            subprocess.run(["python", "scripts/generate_embeddings.py"], check=True)
            st.success("✅ Embeddings generated!")
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")


@st.cache_resource
def initialize_retriever():
    """Initialize retriever once"""
    return Retriever("config.yaml")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Assistant - Apple Organization",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main Streamlit app"""
    
    # Ensure embeddings exist
    ensure_embeddings_exist()
    
    st.title("🚀 RAG Assistant")
    st.markdown("**Retrieval-Augmented Generation with PDF Knowledge Base**")
    st.markdown("---")
    
    # Sidebar for LLM configuration
    st.sidebar.title("⚙️ Configuration")
    
    llm_type = st.sidebar.selectbox(
        "Select LLM Provider:",
        ["Ollama (Local)", "OpenAI", "Google Gemini", "Claude"],
        key="llm_type"
    )
    
    # Map UI names to config names
    llm_type_map = {
        "Ollama (Local)": "ollama",
        "OpenAI": "openai",
        "Google Gemini": "gemini",
        "Claude": "claude"
    }
    
    llm_key = llm_type_map[llm_type]
    api_key = None
    
    # Get API key if not Ollama
    if llm_key != "ollama":
        st.sidebar.markdown("### API Key")
        api_key = st.sidebar.text_input(
            f"Enter your {llm_type} API key:",
            type="password"
        )
        
        if not api_key:
            st.sidebar.warning(f"⚠️ Please enter your {llm_type} API key")
    
    else:
        st.sidebar.markdown("### Local LLM")
        st.sidebar.info("✅ Using Ollama (running locally)")
    
    # Initialize retriever
    try:
        retriever = initialize_retriever()
        st.sidebar.success("✅ Knowledge base loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading knowledge base: {str(e)}")
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This RAG system answers questions based on an Apple organizational structure article."
    )
    
    # Main content area
    st.markdown("### 💬 Ask a Question")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., How is Apple organized?"
    )
    
    if st.button("🔍 Search", use_container_width=True):
        
        if not query:
            st.warning("Please enter a question!")
            return
        
        if llm_key != "ollama" and not api_key:
            st.error(f"Please provide your {llm_type} API key!")
            return
        
        try:
            # Initialize LLM handler
            with st.spinner("🔄 Initializing LLM..."):
                llm_handler = LLMHandler(
                    llm_type=llm_key,
                    api_key=api_key,
                    config_path="config.yaml"
                )
                
                # Check connection
                if not llm_handler.check_connection():
                    if llm_key == "ollama":
                        st.error("❌ Cannot connect to Ollama. Make sure it's running (`ollama serve`)")
                    else:
                        st.error(f"❌ Invalid {llm_type} API key")
                    return
            
            # Retrieve chunks
            with st.spinner("🔍 Retrieving relevant documents..."):
                retrieved_chunks = retriever.retrieve(query)
            
            if not retrieved_chunks:
                st.warning("No relevant documents found!")
                return
            
            # Create prompt
            context = "\n\n".join([
                f"[Chunk {chunk['chunk_id']}]\n{chunk['text']}"
                for chunk in retrieved_chunks
            ])
            
            prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate answer
            with st.spinner("🤖 Generating answer..."):
                answer = llm_handler.generate_answer(prompt)
            
            if not answer:
                st.error("❌ Failed to generate answer!")
                return
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📚 Retrieved Documents")
                st.markdown(f"Found **{len(retrieved_chunks)} relevant chunks**")
                
                for chunk in retrieved_chunks:
                    with st.expander(
                        f"Chunk {chunk['chunk_id']} (Similarity: {chunk['similarity']})"
                    ):
                        st.write(chunk['text'])
            
            with col2:
                st.markdown("#### 💬 Answer")
                st.info(answer)
            
            # Display query info
            st.markdown("---")
            st.markdown("### ℹ️ Query Info")
            col1, col2, col3 = st.columns(3)
            col1.metric("LLM Provider", llm_type)
            col2.metric("Retrieved Chunks", len(retrieved_chunks))
            col3.metric("Top Similarity", retrieved_chunks[0]['similarity'])
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()