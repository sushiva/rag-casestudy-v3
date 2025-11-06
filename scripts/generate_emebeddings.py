#!/usr/bin/env python3
"""
Embedding Generation Script
Converts text chunks into vector embeddings using sentence-transformers
Reads configuration from YAML file
"""

import os
import json
import yaml
from sentence_transformers import SentenceTransformer


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


def load_chunks(input_file):
    """Load chunks from JSON file"""
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Chunks file not found at {input_file}")
        return None
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"📖 Chunks loaded: {input_file}")
        print(f"📊 Total chunks: {data['total_chunks']}")
        return data['chunks']
    except Exception as e:
        print(f"❌ Error loading chunks: {str(e)}")
        return None


def generate_embeddings(chunks, model_name):
    """
    Generate embeddings for chunks using sentence-transformers
    
    Args:
        chunks: List of chunk dictionaries
        model_name: Name of the sentence-transformer model
    
    Returns:
        List of chunks with embeddings
    """
    
    print(f"\n🤖 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"✅ Model loaded!")
    
    print(f"\n🔄 Generating embeddings for {len(chunks)} chunks...")
    
    chunks_with_embeddings = []
    
    for idx, chunk in enumerate(chunks):
        # Generate embedding for this chunk
        embedding = model.encode(chunk['text']).tolist()
        
        # Add embedding to chunk
        chunk_with_embedding = {
            "id": chunk['id'],
            "text": chunk['text'],
            "length": chunk['length'],
            "embedding": embedding
        }
        
        chunks_with_embeddings.append(chunk_with_embedding)
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"  ✓ Processed {idx + 1}/{len(chunks)} chunks")
    
    print(f"✅ All embeddings generated!")
    return chunks_with_embeddings


def save_embeddings(chunks_with_embeddings, output_file):
    """Save embeddings to JSON file"""
    
    try:
        data = {
            "total_chunks": len(chunks_with_embeddings),
            "embedding_dimension": len(chunks_with_embeddings[0]['embedding']),
            "chunks": chunks_with_embeddings
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Embeddings saved to: {output_file}")
        print(f"📊 Total chunks: {len(chunks_with_embeddings)}")
        print(f"📏 Embedding dimension: {data['embedding_dimension']}")
        
        return True
    except Exception as e:
        print(f"❌ Error saving embeddings: {str(e)}")
        return False


def main():
    """Main function"""
    
    print("="*50)
    print("Embedding Generation")
    print("="*50)
    
    # Load configuration
    config = load_config("config.yaml")
    
    if not config or 'embeddings' not in config:
        print("❌ Invalid configuration!")
        return
    
    embedding_config = config['embeddings']
    input_file = embedding_config['input_file']
    output_file = embedding_config['output_file']
    model_name = embedding_config['model_name']
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Model: {model_name}")
    print("="*50)
    
    # Load chunks
    chunks = load_chunks(input_file)
    
    if not chunks:
        print("❌ Failed to load chunks!")
        return
    
    # Generate embeddings
    chunks_with_embeddings = generate_embeddings(chunks, model_name)
    
    if not chunks_with_embeddings:
        print("❌ Failed to generate embeddings!")
        return
    
    # Save embeddings
    save_embeddings(chunks_with_embeddings, output_file)
    print("\n✅ Embedding generation complete!")


if __name__ == "__main__":
    main()