#!/usr/bin/env python3
"""
Text Chunking Script
Splits extracted text into chunks for embeddings
Reads configuration from YAML file
"""

import os
import json
import yaml


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


def read_text_file(file_path):
    """Read text from file"""
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Text file not found at {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"📖 Text file loaded: {file_path}")
        print(f"📊 Total characters: {len(text)}")
        return text
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        return None


def chunk_text(text, chunk_size, chunk_overlap):
    """
    Split text into chunks with overlap
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunks
    """
    
    chunks = []
    step = chunk_size - chunk_overlap
    
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks


def save_chunks_to_json(chunks, output_path):
    """Save chunks to JSON file"""
    
    try:
        # Create metadata
        data = {
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "id": idx,
                    "text": chunk,
                    "length": len(chunk)
                }
                for idx, chunk in enumerate(chunks)
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Chunks saved to: {output_path}")
        print(f"📊 Total chunks: {len(chunks)}")
        
        # Print chunk statistics
        avg_length = sum(c["length"] for c in data["chunks"]) / len(data["chunks"])
        print(f"📈 Average chunk length: {avg_length:.0f} characters")
        
        return True
    except Exception as e:
        print(f"❌ Error saving chunks: {str(e)}")
        return False


def main():
    """Main function"""
    
    print("="*50)
    print("Text Chunking")
    print("="*50)
    
    # Load configuration
    config = load_config("config.yaml")
    
    if not config or 'chunking' not in config:
        print("❌ Invalid configuration!")
        return
    
    chunk_config = config['chunking']
    input_file = chunk_config['input_file']
    output_file = chunk_config['output_file']
    chunk_size = chunk_config['chunk_size']
    chunk_overlap = chunk_config['chunk_overlap']
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")
    print("="*50)
    
    # Read text
    text = read_text_file(input_file)
    
    if not text:
        print("❌ Failed to read text file!")
        return
    
    # Chunk text
    print(f"\n🔪 Chunking text...")
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    if not chunks:
        print("❌ Failed to create chunks!")
        return
    
    # Save chunks
    save_chunks_to_json(chunks, output_file)
    print("\n✅ Chunking complete!")


if __name__ == "__main__":
    main()