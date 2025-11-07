#!/usr/bin/env python3
"""LangChain RAG Pipeline with Pinecone (Serverless)"""

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from anthropic import Anthropic
import os

class HealthcareRAG:
    """Healthcare RAG using Pinecone (serverless)"""
    
    def __init__(self):
        """Initialize with Pinecone"""
        print("🔄 Initializing Healthcare RAG with Pinecone...")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index("healthcare")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create vector store
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings
        )
        
        print("✅ RAG initialized with Pinecone!")
    
    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve relevant documents"""
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return results
    
    def query_openai(self, question: str, api_key: str, top_k: int = 5):
        """Query using OpenAI"""
        results = self.retrieve(question, top_k)
        context = "\n\n".join([doc.page_content for doc, _ in results])
        
        prompt = f"""You are a healthcare assistant. Based on the following medical information, answer the question clearly and accurately.

Medical Information:
{context}

Question: {question}

Answer:"""
        
        llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.7)
        response = llm.invoke(prompt)
        return response.content
    
    def query_gemini(self, question: str, api_key: str, top_k: int = 5):
        """Query using Google Gemini"""
        results = self.retrieve(question, top_k)
        context = "\n\n".join([doc.page_content for doc, _ in results])
        
        prompt = f"""You are a healthcare assistant. Based on the following medical information, answer the question clearly and accurately.

Medical Information:
{context}

Question: {question}

Answer:"""
        
        llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-pro", temperature=0.7)
        response = llm.invoke(prompt)
        return response.content
    
    def query_claude(self, question: str, api_key: str, top_k: int = 5):
        """Query using Claude"""
        results = self.retrieve(question, top_k)
        context = "\n\n".join([doc.page_content for doc, _ in results])
        
        prompt = f"""You are a healthcare assistant. Based on the following medical information, answer the question clearly and accurately.

Medical Information:
{context}

Question: {question}

Answer:"""
        
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def query(self, question: str, llm_provider: str, api_key: str):
        """Query the RAG system"""
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
