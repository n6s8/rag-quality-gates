"""
RAG Evaluator - Automated evaluation of RAG systems
"""
import json
import time
from typing import Dict, List, Any
from pathlib import Path
from .metrics import RAGMetrics


class RAGEvaluator:
    def __init__(self, rag_pipeline, eval_data_path: str = "data/eval_dataset.json"):
        self.rag_pipeline = rag_pipeline
        self.metrics = RAGMetrics()
        self.eval_data = self._load_eval_data(eval_data_path)
    
    def _load_eval_data(self, path: str) -> List[Dict]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load eval data: {e}")
            return []
    
    def run_evaluation(self, top_k: int = 3) -> Dict[str, Any]:
        """Run evaluation on all test queries"""
        if not self.eval_data:
            print("❌ No evaluation data loaded")
            return {"test_results": [], "average_metrics": {}}
        
        results = []
        total_metrics = {
            "precision": [],
            "recall": [],
            "relevance": [],
            "hallucination": [],
            "response_time": []
        }
        
        print(f"\n📊 Evaluating {len(self.eval_data)} test queries...")
        
        for i, test_case in enumerate(self.eval_data):
            print(f"\n{'='*50}")
            print(f"Test #{i+1}: {test_case['question'][:50]}...")
            
            try:
                start_time = time.time()
                result = self.rag_pipeline.process_query(test_case['question'], top_k=top_k)
                end_time = time.time()
                
                retrieved_ids = [doc['id'] for doc in result.get('search_results', [])]
                
                precision = self.metrics.retrieval_precision(
                    retrieved_ids, test_case['expected_quote_ids']
                )
                recall = self.metrics.retrieval_recall(
                    retrieved_ids, test_case['expected_quote_ids']
                )
                relevance = self.metrics.answer_relevance(
                    result.get('answer', ''), test_case['expected_answer']
                )
                hallucination = self.metrics.hallucination_score(
                    result.get('answer', ''), result.get('search_results', [])
                )
                response_time = self.metrics.response_time(start_time, end_time)
                
                test_result = {
                    "question": test_case['question'],
                    "retrieved_ids": retrieved_ids,
                    "expected_ids": test_case['expected_quote_ids'],
                    "answer": result.get('answer', ''),
                    "expected_answer": test_case['expected_answer'],
                    "metrics": {
                        "precision": precision,
                        "recall": recall,
                        "relevance": relevance,
                        "hallucination": hallucination,
                        "response_time": response_time
                    }
                }
                
                results.append(test_result)
                
                for key in total_metrics:
                    total_metrics[key].append(test_result['metrics'][key])
                
                print(f"✅ Retrieved IDs: {retrieved_ids}")
                print(f"📊 Precision: {precision:.2f}, Recall: {recall:.2f}")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                continue
        
        avg_metrics = {}
        for key, values in total_metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
            else:
                avg_metrics[key] = 0.0
        
        return {
            "test_results": results,
            "average_metrics": avg_metrics,
            "config": {
                "top_k": top_k,
                "num_queries": len(self.eval_data),
                "successful_queries": len(results)
            }
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"📁 Results saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


if __name__ == "__main__":
    print("🧪 Testing RAGEvaluator...")
    print("✅ Evaluator module loaded successfully")