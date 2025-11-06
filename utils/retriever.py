#!/usr/bin/env python3
"""
Retriever Module
Retrieves similar chunks from FAISS index based on query
"""

import os
import json
import yaml
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    """Retriever class for semantic search"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize retriever with configuration"""
        
        self.config = self.load_config(config_path)
        
        if not self.config or 'retrieval' not in self.config:
            raise ValueError("Invalid configuration!")
        
        self.retrieval_config = self.config['retrieval']
        
        # Load components
        self.model = self.load_model()
        self.index = self.load_index()
        self.metadata = self.load_metadata()
    
    def load_config(self, config_path="config.yaml"):
        """Load configuration from YAML file"""
        
        if not os.path.exists(config_path):
            print(f"❌ Error: Config file not found at {config_path}")
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Config loaded")
            return config
        except Exception as e:
            print(f"❌ Error loading config: {str(e)}")
            return None
    
    def load_model(self):
        """Load embedding model"""
        
        model_name = self.retrieval_config['model_name']
        print(f"🤖 Loading embedding model: {model_name}")
        
        try:
            model = SentenceTransformer(model_name)
            print(f"✅ Model loaded")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return None
    
    def load_index(self):
        """Load FAISS index"""
        
        index_path = self.retrieval_config['index_path']
        
        if not os.path.exists(index_path):
            print(f"❌ Error: FAISS index not found at {index_path}")
            return None
        
        try:
            index = faiss.read_index(index_path)
            print(f"✅ FAISS index loaded ({index.ntotal} vectors)")
            return index
        except Exception as e:
            print(f"❌ Error loading index: {str(e)}")
            return None
    
    def load_metadata(self):
        """Load metadata (text chunks)"""
        
        metadata_path = self.retrieval_config['metadata_path']
        
        if not os.path.exists(metadata_path):
            print(f"❌ Error: Metadata not found at {metadata_path}")
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✅ Metadata loaded ({metadata['total_chunks']} chunks)")
            return metadata
        except Exception as e:
            print(f"❌ Error loading metadata: {str(e)}")
            return None
    
    def retrieve(self, query):
        """
        Retrieve similar chunks for a query
        
        Args:
            query: Query text
        
        Returns:
            List of similar chunks with scores
        """
        
        if not self.model or not self.index or not self.metadata:
            print("❌ Retriever not initialized properly!")
            return []
        
        top_k = self.retrieval_config['top_k']
        
        # Convert query to embedding
        query_embedding = self.model.encode(query).astype('float32')
        query_embedding = np.array([query_embedding])
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for idx, (distance, chunk_idx) in enumerate(zip(distances[0], indices[0])):
            chunk = self.metadata['chunks'][chunk_idx]
            
            # Convert L2 distance to similarity score (0-1)
            # Lower distance = higher similarity
            similarity = 1 / (1 + distance)
            
            result = {
                "rank": idx + 1,
                "chunk_id": chunk['id'],
                "text": chunk['text'],
                "similarity": round(similarity, 4),
                "distance": round(float(distance), 4)
            }
            results.append(result)
        
        return results


def main():
    """Test retriever with sample queries"""
    
    print("="*60)
    print("Retriever - Test")
    print("="*60)
    
    # Initialize retriever
    retriever = Retriever("config.yaml")
    
    if not retriever.index:
        print("❌ Failed to initialize retriever!")
        return
    
    print("\n✅ Retriever initialized successfully!")
    print("="*60)
    
    # Test queries
    test_queries = [
        "How is Apple organized?",
        "What is the functional organization?",
        "Tell me about innovation at Apple"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        print("-" * 60)
        
        results = retriever.retrieve(query)
        
        for result in results:
            print(f"\nRank {result['rank']} (Similarity: {result['similarity']})")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Text: {result['text'][:200]}...")  # First 200 chars
        
        print("-" * 60)


if __name__ == "__main__":
    main()