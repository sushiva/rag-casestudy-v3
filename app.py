#!/usr/bin/env python3
"""
Healthcare RAG Assistant v3 - LangChain + PineCone
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Healthcare RAG v3",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# IMPORTS
# ============================================================

from rag_chain import get_rag

# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main app"""
    
    st.title("🏥 Healthcare RAG Assistant v3")
    st.markdown("**LangChain + Chroma - Medical Knowledge Base**")
    st.markdown("---")
    
    # Initialize RAG
    rag = get_rag()
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        llm_provider = st.selectbox(
            "Select LLM",
            ["OpenAI", "Claude"]
        )
    
    with col2:
        api_key = st.text_input(f"{llm_provider} API Key", type="password")
    
    if not api_key:
        st.warning("⚠️ Please provide API key")
        st.stop()
    
    # ============================================================
    # QUESTION INPUT
    # ============================================================
    
    st.subheader("❓ Ask a Medical Question")
    
    sample_questions = [
        "What are the symptoms of diabetes?",
        "How is hypertension diagnosed?",
        "What causes heart disease?",
        "What are treatment options for asthma?",
    ]
    
    with st.expander("💡 Sample Questions"):
        for q in sample_questions:
            st.write(f"- {q}")
    
    question = st.text_input(
        "Enter your question:",
        placeholder=sample_questions[0]
    )
    
    # ============================================================
    # GENERATE ANSWER
    # ============================================================
    
    if question:
        if st.button("🔄 Generate Answer"):
            with st.spinner("⏳ Searching medical knowledge base..."):
                try:
                    # Get answer from RAG
                    answer = rag.query(question, llm_provider, api_key)
                    
                    # Display
                    st.success("✅ Answer generated!")
                    st.markdown(answer)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
