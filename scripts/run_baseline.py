#!/usr/bin/env python3
"""
Run baseline evaluation of the RAG system
"""
import sys
import os
import json
from pathlib import Path

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rag.rag_pipeline_rest import RAGPipeline
    print("✅ RAGPipeline import OK")
except ImportError as e:
    print(f"❌ RAGPipeline import failed: {e}")
    sys.exit(1)

try:
    from evaluation.evaluator import RAGEvaluator
    print("✅ RAGEvaluator import OK")
except ImportError as e:
    print(f"❌ RAGEvaluator import failed: {e}")
    sys.exit(1)


def main():
    print("🚀 Running Baseline RAG Evaluation")
    print("="*50)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Check DB connection
    stats = pipeline.get_database_stats()
    print(f"📊 Database Stats: {stats}")
    
    # Run evaluation
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.run_evaluation(top_k=3)
    
    # Display summary
    print("\n" + "="*50)
    print("📋 EVALUATION SUMMARY")
    print("="*50)
    
    avg = results['average_metrics']
    print(f"📈 Average Precision:  {avg.get('precision', 0):.3f}")
    print(f"📈 Average Recall:     {avg.get('recall', 0):.3f}")
    print(f"📝 Answer Relevance:   {avg.get('relevance', 0):.3f}")
    print(f"👻 Hallucination Score: {avg.get('hallucination', 0):.3f}")
    print(f"⏱️  Avg Response Time:  {avg.get('response_time', 0):.3f}s")
    
    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    evaluator.save_results(results, "evaluation/results/baseline.json")
    
    print("\n✅ Baseline evaluation complete!")
    print("📁 Results saved to evaluation/results/baseline.json")


if __name__ == "__main__":
    main()