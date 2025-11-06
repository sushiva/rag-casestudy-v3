#!/usr/bin/env python3
"""
Comprehensive RAG Evaluation Framework
Evaluates retrieval quality, LLM answer quality, and end-to-end performance
"""

import json
import sys
import os
from collections import defaultdict

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../utils'))

from retriever import Retriever
from llm_handler import LLMHandler


class RetrievalEvaluator:
    """Evaluates retrieval quality"""
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.results = []
    
    def precision_at_k(self, retrieved_ids, expected_ids, k=5):
        """Calculate Precision@K"""
        relevant = len(set(retrieved_ids[:k]) & set(expected_ids))
        return relevant / k if k > 0 else 0
    
    def recall_at_k(self, retrieved_ids, expected_ids, k=5):
        """Calculate Recall@K"""
        if not expected_ids:
            return 0
        relevant = len(set(retrieved_ids[:k]) & set(expected_ids))
        return relevant / len(expected_ids)
    
    def mrr(self, retrieved_ids, expected_ids):
        """Calculate Mean Reciprocal Rank"""
        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in expected_ids:
                return 1 / (i + 1)
        return 0
    
    def evaluate_query(self, query, expected_chunk_ids):
        """Evaluate single query"""
        
        # Retrieve chunks
        results = self.retriever.retrieve(query)
        retrieved_ids = [r['chunk_id'] for r in results]
        
        # Calculate metrics
        metrics = {
            "query": query,
            "precision@5": self.precision_at_k(retrieved_ids, expected_chunk_ids, 5),
            "recall@5": self.recall_at_k(retrieved_ids, expected_chunk_ids, 5),
            "mrr": self.mrr(retrieved_ids, expected_chunk_ids),
            "retrieved_chunks": retrieved_ids,
            "expected_chunks": expected_chunk_ids
        }
        
        self.results.append(metrics)
        return metrics
    
    def aggregate_results(self):
        """Aggregate all results"""
        if not self.results:
            return {}
        
        avg_precision = sum(r['precision@5'] for r in self.results) / len(self.results)
        avg_recall = sum(r['recall@5'] for r in self.results) / len(self.results)
        avg_mrr = sum(r['mrr'] for r in self.results) / len(self.results)
        
        return {
            "avg_precision@5": round(avg_precision, 4),
            "avg_recall@5": round(avg_recall, 4),
            "avg_mrr": round(avg_mrr, 4),
            "total_queries": len(self.results)
        }


class AnswerEvaluator:
    """Evaluates answer quality"""
    
    def __init__(self):
        self.results = []
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between texts"""
        try:
            from sentence_transformers import util
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            emb1 = model.encode(text1, convert_to_tensor=True)
            emb2 = model.encode(text2, convert_to_tensor=True)
            
            similarity = util.pytorch_cos_sim(emb1, emb2).item()
            return round(similarity, 4)
        except:
            return None
    
    def evaluate_answer(self, generated_answer, expected_answer):
        """Evaluate single answer"""
        
        similarity = self.semantic_similarity(generated_answer, expected_answer)
        
        metrics = {
            "generated_answer": generated_answer[:100] + "...",
            "expected_answer": expected_answer[:100] + "...",
            "semantic_similarity": similarity
        }
        
        self.results.append(metrics)
        return metrics
    
    def aggregate_results(self):
        """Aggregate all results"""
        if not self.results:
            return {}
        
        similarities = [r['semantic_similarity'] for r in self.results if r['semantic_similarity'] is not None]
        
        if not similarities:
            return {"error": "No similarities calculated"}
        
        return {
            "avg_semantic_similarity": round(sum(similarities) / len(similarities), 4),
            "min_similarity": round(min(similarities), 4),
            "max_similarity": round(max(similarities), 4),
            "total_answers": len(self.results)
        }


def load_test_dataset(filepath):
    """Load test dataset"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['test_cases']


def create_prompt_with_context(query, retrieved_chunks):
    """Create prompt with context"""
    context = "\n\n".join([
        f"[Chunk {chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])
    
    prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
    
    return prompt


def main():
    """Main evaluation function"""
    
    print("="*70)
    print("RAG EVALUATION FRAMEWORK")
    print("="*70)
    
    # Load test dataset
    test_cases = load_test_dataset("scripts/test_dataset.json")
    print(f"\n📊 Loaded {len(test_cases)} test cases")
    
    # Initialize components
    print("\n🔧 Initializing components...")
    retriever = Retriever("config.yaml")
    llm_handler = LLMHandler(llm_type="ollama", config_path="config.yaml")
    
    # Initialize evaluators
    retrieval_eval = RetrievalEvaluator(retriever)
    answer_eval = AnswerEvaluator()
    
    print("✅ Components initialized!")
    
    # Evaluate each test case
    print("\n" + "="*70)
    print("EVALUATING TEST CASES")
    print("="*70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Question: {test_case['question'][:60]}...")
        
        # Evaluate retrieval
        retrieval_metrics = retrieval_eval.evaluate_query(
            test_case['question'],
            test_case['expected_chunks']
        )
        
        print(f"  Retrieval - Precision: {retrieval_metrics['precision@5']}, Recall: {retrieval_metrics['recall@5']}")
        
        # Get retrieved chunks for answer generation
        retrieved_chunks = retriever.retrieve(test_case['question'])
        
        # Generate answer
        prompt = create_prompt_with_context(test_case['question'], retrieved_chunks)
        generated_answer = llm_handler.generate_answer(prompt)
        
        if generated_answer:
            # Evaluate answer
            answer_metrics = answer_eval.evaluate_answer(
                generated_answer,
                test_case['expected_answer']
            )
            print(f"  Answer - Similarity: {answer_metrics['semantic_similarity']}")
        else:
            print("  Answer - Failed to generate")
    
    # Aggregate and display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    retrieval_results = retrieval_eval.aggregate_results()
    answer_results = answer_eval.aggregate_results()
    
    print("\n📚 RETRIEVAL METRICS:")
    for key, value in retrieval_results.items():
        print(f"  {key}: {value}")
    
    print("\n💬 ANSWER QUALITY METRICS:")
    for key, value in answer_results.items():
        print(f"  {key}: {value}")
    
    # Save results
    evaluation_report = {
        "timestamp": str(__import__('datetime').datetime.now()),
        "retrieval_metrics": retrieval_results,
        "answer_metrics": answer_results,
        "test_cases_count": len(test_cases),
        "llm_provider": "ollama"
    }
    
    with open("evaluation_report.json", 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    print("\n✅ Evaluation complete! Report saved to evaluation_report.json")
    print("="*70)


if __name__ == "__main__":
    main()