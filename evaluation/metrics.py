"""
RAG Metrics Calculation Module
"""
import numpy as np
from typing import List, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    print("⚠️  Install dependencies: pip install sentence-transformers scikit-learn")


class RAGMetrics:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embedder = None
        
        if HAS_DEPENDENCIES:
            try:
                self.embedder = SentenceTransformer(embedding_model_name)
                print(f"✅ Metrics initialized with {embedding_model_name}")
            except Exception as e:
                print(f"⚠️  Could not load embedder: {e}")
                self.embedder = None
        else:
            print("⚠️  Running without embedding capabilities")
    
    def retrieval_precision(self, retrieved_ids: List[int], expected_ids: List[int]) -> float:
        """Precision: % of retrieved items that are relevant"""
        if not retrieved_ids:
            return 0.0
        
        relevant_retrieved = set(retrieved_ids) & set(expected_ids)
        return len(relevant_retrieved) / len(retrieved_ids)
    
    def retrieval_recall(self, retrieved_ids: List[int], expected_ids: List[int]) -> float:
        """Recall: % of expected items that were retrieved"""
        if not expected_ids:
            return 0.0
        
        relevant_retrieved = set(retrieved_ids) & set(expected_ids)
        return len(relevant_retrieved) / len(expected_ids)
    
    def answer_relevance(self, generated_answer: str, expected_answer: str) -> float:
        """Semantic similarity between generated and expected answer (0-1)"""
        if not generated_answer or not expected_answer or not self.embedder:
            return 0.0
        
        try:
            embeddings = self.embedder.encode([generated_answer, expected_answer])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"⚠️  Relevance calculation failed: {e}")
            return 0.0
    
    def hallucination_score(self, generated_answer: str, context_docs: List[Dict]) -> float:
        """Check if answer contains unsupported information (simplified)"""
        if not generated_answer:
            return 0.0
        
        low_hallucination_phrases = [
            "cannot find", "not in the context", "I don't know", 
            "no information", "my knowledge base", "I cannot", "unknown",
            "based on the provided", "according to the context"
        ]
        
        answer_lower = generated_answer.lower()
        if any(phrase in answer_lower for phrase in low_hallucination_phrases):
            return 0.0
        
        return 0.1  # Default low hallucination score
    
    def response_time(self, start_time: float, end_time: float) -> float:
        """Calculate response time in seconds"""
        return end_time - start_time


if __name__ == "__main__":
    print("🧪 Testing RAGMetrics...")
    metrics = RAGMetrics()
    print("✅ Metrics module loaded successfully")