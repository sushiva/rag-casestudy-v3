#!/usr/bin/env python3
"""
Vector Database Script
Builds FAISS index from embeddings for fast semantic search
Reads configuration from YAML file
"""

import os
import json
import yaml
import numpy as np
import faiss


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found at {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {str(e)}")
        return None


def load_embeddings(embeddings_file):
    """Load embeddings from JSON file"""
    
    if not os.path.exists(embeddings_file):
        print(f"❌ Error: Embeddings file not found at {embeddings_file}")
        return None
    
    try:
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📖 Embeddings loaded: {embeddings_file}")
        print(f"📊 Total chunks: {data['total_chunks']}")
        print(f"📏 Embedding dimension: {data['embedding_dimension']}")
        
        return data
    except Exception as e:
        print(f"❌ Error loading embeddings: {str(e)}")
        return None


def build_faiss_index(embeddings_data):
    """
    Build FAISS index from embeddings
    
    Args:
        embeddings_data: Dictionary with chunks and embeddings
    
    Returns:
        FAISS index object
    """
    
    print(f"\n🔨 Building FAISS index...")
    
    # Extract embeddings as numpy array
    embeddings = np.array([chunk['embedding'] for chunk in embeddings_data['chunks']])
    embedding_dim = embeddings_data['embedding_dimension']
    
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Dimension: {embedding_dim}")
    
    # Create FAISS index
    # Using IndexFlatL2 for exact search (good for small datasets)
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Add embeddings to index
    index.add(embeddings.astype('float32'))
    
    print(f"✅ FAISS index created with {index.ntotal} vectors")
    
    return index


def save_faiss_index(index, index_path):
    """Save FAISS index to file"""
    
    try:
        faiss.write_index(index, index_path)
        print(f"✅ Index saved to: {index_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving index: {str(e)}")
        return False


def save_metadata(embeddings_data, metadata_path):
    """
    Save metadata (text and IDs) separately for retrieval
    FAISS only stores vectors, not text
    """
    
    try:
        metadata = {
            "total_chunks": embeddings_data['total_chunks'],
            "chunks": [
                {
                    "id": chunk['id'],
                    "text": chunk['text'],
                    "length": chunk['length']
                }
                for chunk in embeddings_data['chunks']
            ]
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Metadata saved to: {metadata_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving metadata: {str(e)}")
        return False


def main():
    """Main function"""
    
    print("="*50)
    print("Vector Database - FAISS Index Building")
    print("="*50)
    
    # Load configuration
    config = load_config("config.yaml")
    
    if not config or 'vector_db' not in config:
        print("❌ Invalid configuration!")
        return
    
    vdb_config = config['vector_db']
    embeddings_file = vdb_config['embeddings_file']
    index_path = vdb_config['index_path']
    metadata_path = vdb_config['metadata_path']
    
    print(f"Embeddings file: {embeddings_file}")
    print(f"Index output: {index_path}")
    print(f"Metadata output: {metadata_path}")
    print("="*50)
    
    # Load embeddings
    embeddings_data = load_embeddings(embeddings_file)
    
    if not embeddings_data:
        print("❌ Failed to load embeddings!")
        return
    
    # Build FAISS index
    index = build_faiss_index(embeddings_data)
    
    if not index:
        print("❌ Failed to build index!")
        return
    
    # Save index
    if not save_faiss_index(index, index_path):
        print("❌ Failed to save index!")
        return
    
    # Save metadata
    if not save_metadata(embeddings_data, metadata_path):
        print("❌ Failed to save metadata!")
        return
    
    print("\n✅ Vector database creation complete!")
    print(f"📊 Index ready for {index.ntotal} vectors")


if __name__ == "__main__":
    main()