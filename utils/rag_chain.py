#!/usr/bin/env python3
"""LangChain RAG Pipeline with Pinecone (Serverless)"""

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from anthropic import Anthropic
import google.generativeai as genai
import os

class HealthcareRAG:
    """Healthcare RAG using Pinecone (serverless)"""
    
    def __init__(self):
        """Initialize with Pinecone (no API key needed)"""
        print("🔄 Initializing Healthcare RAG with Pinecone...")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index("healthcare")
        
        self.vectorstore = None
        print("✅ RAG initialized with Pinecone!")
    
    def _ensure_vectorstore(self, openai_api_key: str):
        """Lazily initialize vectorstore with OpenAI API key"""
        if self.vectorstore is None:
            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            self.vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=embeddings
            )
    
    def retrieve(self, query: str, openai_api_key: str, top_k: int = 5):
        """Retrieve relevant documents"""
        self._ensure_vectorstore(openai_api_key)
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return results
    
    def query_openai(self, question: str, api_key: str, top_k: int = 5):
        """Query using OpenAI"""
        results = self.retrieve(question, api_key, top_k)
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
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        results = self.retrieve(question, openai_key, top_k)
        context = "\n\n".join([doc.page_content for doc, _ in results])
        
        prompt = f"""You are a healthcare assistant. Based on the following medical information, answer the question clearly and accurately.

Medical Information:
{context}

Question: {question}

Answer:"""
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        return response.text
    
    def query_claude(self, question: str, api_key: str, top_k: int = 5):
        """Query using Claude"""
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        results = self.retrieve(question, openai_key, top_k)
        context = "\n\n".join([doc.page_content for doc, _ in results])
        
        prompt = f"""You are a healthcare assistant. Based on the following medical information, answer the question clearly and accurately.

Medical Information:
{context}

Question: {question}

Answer:"""
        
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-5",
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
