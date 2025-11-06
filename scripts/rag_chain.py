
#!/usr/bin/env python3
"""
LangChain RAG Pipeline for Healthcare
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from anthropic import Anthropic

class HealthcareRAG:
    """Healthcare RAG system using LangChain + Chroma"""
    
    def __init__(self, persist_directory: str = "data/chroma_db"):
        """Initialize RAG with Chroma DB"""
        print("🔄 Initializing Healthcare RAG...")
        
        # Load embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Load Chroma DB
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="healthcare"
        )
        
        print(f"✅ RAG initialized with Chroma DB")
    
    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve relevant documents"""
        results = self.db.similarity_search_with_score(query, k=top_k)
        return results
    
    def query_openai(self, question: str, api_key: str, top_k: int = 5):
        """Query using OpenAI"""
        # Retrieve context
        results = self.retrieve(question, top_k)
        context = "\n\n".join([doc.page_content for doc, _ in results])
        
        # Create prompt
        prompt = f"""You are a healthcare assistant. Based on the following medical information, answer the question clearly and accurately.

Medical Information:
{context}

Question: {question}

Answer:"""
        
        # Call OpenAI
        llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.7)
        response = llm.invoke(prompt)
        
        return response.content
    
    def query_gemini(self, question: str, api_key: str, top_k: int = 5):
        """Query using Google Gemini"""
        # Retrieve context
        results = self.retrieve(question, top_k)
        context = "\n\n".join([doc.page_content for doc, _ in results])
        
        # Create prompt
        prompt = f"""You are a healthcare assistant. Based on the following medical information, answer the question clearly and accurately.

Medical Information:
{context}

Question: {question}

Answer:"""
        
        # Call Gemini
        llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-2.5-flash", temperature=0.7)
        response = llm.invoke(prompt)
        
        return response.content
    
    def query_claude(self, question: str, api_key: str, top_k: int = 5):
        """Query using Claude"""
        # Retrieve context
        results = self.retrieve(question, top_k)
        context = "\n\n".join([doc.page_content for doc, _ in results])
        
        # Create prompt
        prompt = f"""You are a healthcare assistant. Based on the following medical information, answer the question clearly and accurately.

Medical Information:
{context}

Question: {question}

Answer:"""
        
        # Call Claude
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def query(self, question: str, llm_provider: str, api_key: str):
        """
        Query the RAG system
        
        Args:
            question: User's question
            llm_provider: "openai", "gemini", or "claude"
            api_key: API key for the provider
        
        Returns:
            Answer from LLM
        """
        if llm_provider == "OpenAI":
            return self.query_openai(question, api_key)
        elif llm_provider == "Google Gemini":
            return self.query_gemini(question, api_key)
        elif llm_provider == "Claude":
            return self.query_claude(question, api_key)
        else:
            raise ValueError(f"Unknown provider: {llm_provider}")

# Initialize globally
_rag = None

def get_rag():
    """Get or initialize RAG instance"""
    global _rag
    if _rag is None:
        _rag = HealthcareRAG()
    return _rag
