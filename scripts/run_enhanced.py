#!/usr/bin/env python3
"""
Run evaluation with enhanced RAG
"""
import sys
import os
from pathlib import Path

# Fix path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

try:
    # Try to import
    from enhancements.simple_enhancement import EnhancedRAG
    from evaluation.evaluator import RAGEvaluator
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    print("\nTrying alternative import...")
    
    # Try direct import
    import importlib.util
    import sys
    
    # Import evaluator
    evaluator_path = project_root / "evaluation" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
    evaluator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator_module)
    RAGEvaluator = evaluator_module.RAGEvaluator
    print("✅ Evaluator imported directly")
    
    # Import enhanced RAG
    enhancer_path = project_root / "enhancements" / "simple_enhancement.py"
    spec = importlib.util.spec_from_file_location("enhancer", enhancer_path)
    enhancer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enhancer_module)
    EnhancedRAG = enhancer_module.EnhancedRAG
    print("✅ EnhancedRAG imported directly")


def main():
    print("\n" + "="*60)
    print("🚀 RUNNING ENHANCED RAG EVALUATION")
    print("="*60)
    
    # Initialize pipeline
    pipeline = EnhancedRAG()
    
    # Run evaluation
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.run_evaluation(top_k=3)
    
    # Display summary
    print("\n" + "="*60)
    print("📋 ENHANCED EVALUATION RESULTS")
    print("="*60)
    
    avg = results['average_metrics']
    print(f"📈 Average Precision:  {avg.get('precision', 0):.3f}")
    print(f"📈 Average Recall:     {avg.get('recall', 0):.3f}")
    print(f"📝 Answer Relevance:   {avg.get('relevance', 0):.3f}")
    print(f"👻 Hallucination:      {avg.get('hallucination', 0):.3f}")
    print(f"⏱️  Response Time:     {avg.get('response_time', 0):.3f}s")
    
    # Calculate improvement needed
    baseline_path = project_root / "evaluation" / "results" / "baseline.json"
    if baseline_path.exists():
        import json
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        baseline_precision = baseline['average_metrics'].get('precision', 0)
        
        improvement = ((avg.get('precision', 0) - baseline_precision) / baseline_precision * 100) if baseline_precision > 0 else 0
        print(f"📊 Precision Improvement: {improvement:+.1f}%")
        
        if improvement >= 30:
            print("🎉 ✅ TARGET ACHIEVED: ≥30% improvement!")
        else:
            print(f"⚠️  Target NOT achieved: Need ≥30%, got {improvement:.1f}%")
    
    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    output_path = "evaluation/results/enhanced_simple.json"
    evaluator.save_results(results, output_path)
    
    print(f"\n📁 Results saved to: {output_path}")
    print("✅ Enhanced evaluation complete!")


if __name__ == "__main__":
    main()