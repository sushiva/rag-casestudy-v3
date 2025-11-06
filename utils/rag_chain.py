#!/usr/bin/env python3
"""
RAG Chain Module
Combines Retriever + LLM for complete RAG pipeline
"""

import os
import yaml
from retriever import Retriever
from llm_handler import LLMHandler


class RAGChain:
    """Complete RAG pipeline"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize RAG chain"""
        
        print("="*60)
        print("Initializing RAG Chain")
        print("="*60)
        
        # Initialize retriever
        print("\n📚 Initializing Retriever...")
        self.retriever = Retriever(config_path)
        
        # Initialize LLM
        print("\n🤖 Initializing LLM Handler...")
        self.llm_handler = LLMHandler(config_path)
        
        # Check LLM connection
        if not self.llm_handler.check_connection():
            raise Exception("Cannot connect to LLM!")
        
        print("\n✅ RAG Chain initialized successfully!")
        print("="*60)
    
    def create_prompt(self, query, retrieved_chunks):
        """Create prompt for LLM with retrieved context"""
        
        # Format retrieved chunks
        context = "\n\n".join([
            f"[Chunk {chunk['chunk_id']}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        ])
        
        # Create prompt
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def query(self, query):
        """
        Execute complete RAG pipeline
        
        Args:
            query: User query
        
        Returns:
            Dictionary with retrieved chunks and LLM answer
        """
        
        print("\n" + "="*60)
        print(f"Query: {query}")
        print("="*60)
        
        # Step 1: Retrieve relevant chunks
        print("\n🔍 Retrieving relevant chunks...")
        retrieved_chunks = self.retriever.retrieve(query)
        
        print(f"✅ Retrieved {len(retrieved_chunks)} chunks")
        for chunk in retrieved_chunks:
            print(f"   Rank {chunk['rank']}: Similarity {chunk['similarity']}")
        
        # Step 2: Create prompt
        print("\n📝 Creating prompt...")
        prompt = self.create_prompt(query, retrieved_chunks)
        
        # Step 3: Generate answer from LLM
        print("\n🤖 Generating answer from LLM...")
        answer = self.llm_handler.generate_answer(prompt)
        
        if not answer:
            print("❌ Failed to generate answer!")
            return None
        
        print("✅ Answer generated!")
        
        # Return results
        result = {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer
        }
        
        return result


def main():
    """Test RAG chain with sample queries"""
    
    try:
        # Initialize RAG chain
        rag = RAGChain("config.yaml")
        
        # Test queries
        test_queries = [
            "How is Apple organized?",
            "What is a functional organization?",
            "Tell me about Apple's leadership model"
        ]
        
        for query in test_queries:
            result = rag.query(query)
            
            if result:
                print("\n" + "="*60)
                print("RESULT")
                print("="*60)
                print(f"\n📌 Query: {result['query']}")
                print(f"\n📚 Retrieved {len(result['retrieved_chunks'])} chunks:")
                for chunk in result['retrieved_chunks']:
                    print(f"   [{chunk['rank']}] Chunk {chunk['chunk_id']} (Similarity: {chunk['similarity']})")
                print(f"\n💬 Answer:\n{result['answer']}")
                print("="*60)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()