#!/usr/bin/env python3
"""
RAG Application v2 - Multi-Domain (Apple + Healthcare)
Supports searching across multiple domains
"""

# ============================================================
# 1. PAGE CONFIG (FIRST!)
# ============================================================
import streamlit as st

st.set_page_config(
    page_title="RAG Assistant v2 - Multi-Domain",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. IMPORTS (After set_page_config)
# ============================================================
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from retriever import MultiDomainRetriever

# ============================================================
# 3. INITIALIZATION FUNCTIONS
# ============================================================

def ensure_embeddings_exist():
    """Check if embeddings exist"""
    apple_emb = "data/apple/embeddings.json"
    healthcare_emb = "data/healthcare/embeddings.json"
    
    if not os.path.exists(apple_emb):
        st.warning("⚠️ Apple embeddings missing")
        return False
    if not os.path.exists(healthcare_emb):
        st.warning("⚠️ Healthcare embeddings missing")
        return False
    
    return True

@st.cache_resource
def initialize_retriever():
    """Initialize multi-domain retriever"""
    return MultiDomainRetriever()

# ============================================================
# 4. MAIN APP
# ============================================================

def main():
    """Main Streamlit app"""
    
    # Check embeddings
    if not ensure_embeddings_exist():
        st.error("❌ Embeddings not found. Please generate them first.")
        return
    
    st.title("🚀 RAG Assistant v2")
    st.markdown("**Multi-Domain Search: Select a domain to get started**")
    st.markdown("---")
    
    # Initialize retriever
    retriever = initialize_retriever()
    
    # Show domains info in sidebar
    st.sidebar.write("📊 **Available Domains:**")
    domain_info = retriever.get_domain_info()
    for domain, info in domain_info.items():
        st.sidebar.write(f"- {domain.upper()}: {info['chunks']} chunks")
    
    # ============================================================
    # STEP 1: DOMAIN SELECTION
    # ============================================================
    
    st.subheader("1️⃣ Select a Domain")
    
    domain_options = {
        "🍎 Apple Organization": "apple",
        "🏥 Healthcare & Medicine": "healthcare"
    }
    
    selected_domain_display = st.selectbox(
        "What would you like to search?",
        list(domain_options.keys()),
        key="domain_select"
    )
    
    selected_domain = domain_options[selected_domain_display]
    
    # Sample questions for each domain
    sample_questions = {
        "apple": [
            "How is Apple organized?",
            "What are Apple's three leadership characteristics?",
            "Explain the iPhone portrait mode development process",
            "Why did Steve Jobs change Apple's organization?"
        ],
        "healthcare": [
            "What are the common symptoms of diabetes?",
            "How is hypertension diagnosed?",
            "What risk factors contribute to heart disease?",
            "What are the treatment options for asthma?",
            "Describe the diagnostic criteria for depression"
        ]
    }
    
    st.info(f"📌 Selected: **{selected_domain_display}**")
    
    # ============================================================
    # STEP 2: API KEY CONFIGURATION
    # ============================================================
    
    st.subheader("2️⃣ Configure LLM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        llm_choice = st.selectbox(
            "Select LLM Provider",
            ["OpenAI", "Google Gemini", "Claude", "Ollama"],
            key="llm_select"
        )
    
    with col2:
        if llm_choice != "Ollama":
            api_key = st.text_input(f"{llm_choice} API Key", type="password", key="api_key_input")
        else:
            api_key = "ollama"
            st.write("✅ Using local Ollama")
    
    # Check if API key provided
    if llm_choice != "Ollama" and not api_key:
        st.warning("⚠️ Please provide API key to proceed")
        st.stop()
    
    # ============================================================
    # STEP 3: QUESTION INPUT
    # ============================================================
    
    st.subheader(f"3️⃣ Ask a Question ({selected_domain_display})")
    
    # Show sample questions
    with st.expander("💡 Sample Questions"):
        for sample in sample_questions[selected_domain]:
            st.write(f"- {sample}")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder=sample_questions[selected_domain][0],
        key="question_input"
    )
    
    # ============================================================
    # STEP 4: SEARCH & RETRIEVE (Hidden from user)
    # ============================================================
    
    if question:
        st.write("---")
        
        # Generate query embedding
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode(question)
        
        # Search only in selected domain
        domain_data = retriever.domains.get(selected_domain)
        
        results = []
        if domain_data:
            index = domain_data['index']
            chunks = domain_data['chunks']
            
            query_vec = np.array([query_embedding], dtype=np.float32)
            distances, indices = index.search(query_vec, min(5, len(chunks)))
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(chunks):
                    results.append({
                        'text': chunks[idx],
                        'domain': selected_domain,
                        'distance': float(dist),
                        'similarity': 1 / (1 + float(dist))
                    })
        
        # ============================================================
        # STEP 5: GENERATE ANSWER WITH LLM
        # ============================================================
        
        st.subheader("💬 Answer")
        
        if results:
            # Combine retrieved documents as context
            context = "\n\n".join([r['text'] for r in results])
            
            # Create RAG prompt
            prompt = f"""You are a helpful assistant. Based on the following documents, answer the user's question clearly and concisely.

Question: {question}

Documents:
{context}

Answer:"""
            
            # Generate answer button
            if st.button("🔄 Generate Answer", key="answer_button"):
                with st.spinner("⏳ Generating answer..."):
                    try:
                        if llm_choice == "OpenAI":
                            from openai import OpenAI
                            client = OpenAI(api_key=api_key)
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.7
                            )
                            answer = response.choices[0].message.content
                        
                        elif llm_choice == "Google Gemini":
                            import google.generativeai as genai
                            genai.configure(api_key=api_key)
                            model_gemini = genai.GenerativeModel('gemini-2.5-pro')
                            response = model_gemini.generate_content(prompt)
                            answer = response.text
                        
                        elif llm_choice == "Claude":
                            from anthropic import Anthropic
                            client = Anthropic(api_key=api_key)
                            response = client.messages.create(
                                model="claude-3-5-sonnet-20241022",
                                max_tokens=1024,
                                messages=[{"role": "user", "content": prompt}]
                            )
                            answer = response.content[0].text
                        
                        # Display answer
                        st.success("✅ Answer generated!")
                        st.markdown(answer)
                        
                        
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        else:
            st.warning("❌ No relevant documents found")

# ============================================================
# 5. RUN APP
# ============================================================

if __name__ == "__main__":
    main()