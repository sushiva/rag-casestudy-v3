#!/usr/bin/env python3
"""
Multi-Domain RAG Retriever
Searches across Apple and Healthcare domains
"""

import faiss
import json
import numpy as np
import os

class MultiDomainRetriever:
    """Search across multiple domains (Apple, Healthcare, etc.)"""
    
    def __init__(self, domains_config=None):
        """Initialize retriever with multiple domains"""
        if domains_config is None:
            domains_config = {
                'apple': 'data/apple',
                'healthcare': 'data/healthcare'
            }
        
        self.domains = {}
        self.load_all_domains(domains_config)
    
    def load_domain(self, domain_name, domain_path):
        """Load a single domain's index and chunks"""
        print(f"📂 Loading domain: {domain_name}...")
        
        # Check if files exist
        emb_path = f"{domain_path}/embeddings.json"
        chunks_path = f"{domain_path}/chunks.json"
        
        if not os.path.exists(emb_path):
            print(f"⚠️ Embeddings not found for {domain_name}")
            return False
        
        if not os.path.exists(chunks_path):
            print(f"⚠️ Chunks not found for {domain_name}")
            return False
        
        try:
            # Load embeddings
            with open(emb_path, 'r') as f:
                emb_data = json.load(f)
            
            # Handle different embedding formats
            if 'chunks' in emb_data:
                embeddings = emb_data['chunks']  # Old format
            else:
                embeddings = emb_data.get('embeddings', [])  # New format
            
            if not embeddings:
                print(f"❌ No embeddings found for {domain_name}")
                return False
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Build FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            
            # Load chunks
            with open(chunks_path, 'r') as f:
                chunks_data = json.load(f)
            
            raw_chunks = chunks_data.get('chunks', [])
            
            # Convert chunks to strings (CRITICAL!)
            chunks = []
            for chunk in raw_chunks:
                if isinstance(chunk, dict):
                    # Extract text from dict
                    text = chunk.get('text', str(chunk))
                elif isinstance(chunk, list):
                    # Join list elements
                    text = ' '.join(str(item) for item in chunk)
                else:
                    # Already a string or other type
                    text = str(chunk)
                
                chunks.append(text)
            
            if not chunks:
                print(f"❌ No chunks found for {domain_name}")
                return False
            
            # Store domain
            self.domains[domain_name] = {
                'index': index,
                'chunks': chunks,
                'embeddings': embeddings_array,
                'path': domain_path
            }
            
            print(f"✅ {domain_name}: {len(chunks)} chunks loaded (all strings)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading {domain_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_all_domains(self, domains_config):
        """Load all configured domains"""
        loaded = 0
        failed = []
        
        for domain_name, domain_path in domains_config.items():
            if self.load_domain(domain_name, domain_path):
                loaded += 1
            else:
                failed.append(domain_name)
        
        print(f"\n📊 Total domains loaded: {loaded}/{len(domains_config)}")
        if failed:
            print(f"⚠️ Failed domains: {failed}\n")
        else:
            print()
    
    def search(self, query_embedding, top_k=5):
        """
        Search across all domains
        
        Args:
            query_embedding: Query vector
            top_k: Number of results per domain
        
        Returns:
            List of results with domain info
        """
        results = []
        
        for domain_name, domain_data in self.domains.items():
            index = domain_data['index']
            chunks = domain_data['chunks']
            
            try:
                # Search in this domain
                query_vec = np.array([query_embedding], dtype=np.float32)
                distances, indices = index.search(query_vec, min(top_k, len(chunks)))
                
                # Collect results with domain info
                for dist, idx in zip(distances[0], indices[0]):
                    if idx != -1 and idx < len(chunks):
                        chunk_text = chunks[idx]
                        
                        # Ensure it's a string
                        if not isinstance(chunk_text, str):
                            chunk_text = str(chunk_text)
                        
                        results.append({
                            'text': chunk_text,
                            'domain': domain_name,
                            'distance': float(dist),
                            'similarity': 1 / (1 + float(dist))
                        })
            except Exception as e:
                print(f"⚠️ Error searching {domain_name}: {e}")
        
        if not results:
            return []
        
        # Sort by similarity (lower distance = higher similarity)
        results = sorted(results, key=lambda x: x['distance'])[:top_k]
        return results
    
    def get_domain_info(self):
        """Get info about loaded domains"""
        info = {}
        for domain_name, domain_data in self.domains.items():
            info[domain_name] = {
                'chunks': len(domain_data['chunks']),
                'dimension': domain_data['embeddings'].shape[1]
            }
        return info


# For backward compatibility
Retriever = MultiDomainRetriever