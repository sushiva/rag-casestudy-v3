#!/usr/bin/env python3
"""
Ingest healthcare PDF into Chroma DB using LangChain
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_pdf(pdf_path: str, persist_directory: str = "data/chroma_db"):
    """
    Ingest PDF into Chroma DB
    
    Args:
        pdf_path: Path to PDF file
        persist_directory: Where to store Chroma DB
    """
    
    print(f"📄 Loading PDF: {pdf_path}")
    
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")
    
    # 2. Split into chunks
    print("✂️ Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Filter out small/poor quality chunks
    print("🧹 Filtering low-quality chunks...")
    filtered_chunks = []
    for chunk in chunks:
        text = chunk.page_content.strip()
        # Skip very short chunks
        if len(text) < 100:
            continue
        # Skip if mostly numbers/formatting
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.3:
            continue
        filtered_chunks.append(chunk)
    
    print(f"✅ Filtered to {len(filtered_chunks)} quality chunks")
    
    # 3. Generate embeddings and create Chroma DB
    print("🧠 Generating embeddings with HuggingFace...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create or update Chroma DB
    print(f"💾 Creating Chroma DB at {persist_directory}")
    db = Chroma.from_documents(
        documents=filtered_chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="healthcare"
    )
    
    db.persist()
    print(f"✅ Chroma DB created with {len(filtered_chunks)} documents")
    
    return db

if __name__ == "__main__":
    pdf_path = "data/medical_diagnosis_manual.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        exit(1)
    
    ingest_pdf(pdf_path)
    print("\n✅ Ingestion complete!")
