#!/usr/bin/env python3
"""
Enhanced RAG Evaluation Runner
Runs evaluation for enhanced RAG system and saves stable result files for reporting
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from evaluation.evaluator import RAGEvaluator
    from rag.rag_pipeline_rest import RAGPipeline
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from project root directory")
    IMPORTS_OK = False


def print_metrics_summary(results, title="Evaluation Results"):
    avg = results.get("average_metrics", {})

    print(f"\n📊 {title}")
    print("-" * 60)
    print(f"  Retrieval Precision:     {avg.get('precision', 0):.3f}")
    print(f"  Retrieval Recall:        {avg.get('recall', 0):.3f}")
    print(f"  Answer Relevance:        {avg.get('relevance', 0):.3f}")
    print(f"  Hallucination:           {avg.get('hallucination', 0):.3f}")
    print(f"  Response Time:           {avg.get('response_time', 0):.3f}s")

    if "interpretation_score" in avg:
        print("\n🧠 ANALYSIS METRICS:")
        print(f"  Interpretation Score:    {avg.get('interpretation_score', 0):.3f}")
        print(f"  Historical Context:      {avg.get('historical_context_score', 0):.3f}")
        print(f"  Explanation Depth:       {avg.get('explanation_depth', 0):.3f}")
        print(f"  Thematic Analysis:       {avg.get('thematic_analysis', 0):.3f}")
        print(f"  Interpretation Quality:  {avg.get('interpretation_quality', 0):.3f}")


def main():
    if not IMPORTS_OK:
        print("❌ Cannot run enhanced evaluation due to import errors")
        return

    print("🚀 Running Enhanced RAG Evaluation")
    print("=" * 70)

    os.makedirs("evaluation/results", exist_ok=True)

    try:
        print("🔧 Initializing Enhanced RAG Pipeline...")
        pipeline = RAGPipeline(use_enhanced=True)
        print("✅ Enhanced pipeline initialized")

        evaluator = RAGEvaluator(pipeline)

        print("\n📊 Running evaluation...")
        results = evaluator.run_evaluation(
            top_k=3,
            include_analysis_metrics=True,
            analysis_mode=None
        )

        print_metrics_summary(results, "Enhanced System Results")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        enhanced_timestamped = f"evaluation/results/enhanced_simple_{timestamp}.json"
        evaluator.save_results(results, enhanced_timestamped)

        enhanced_stable = "evaluation/results/enhanced_simple.json"
        evaluator.save_results(results, enhanced_stable)

        print("\n📁 Saved:")
        print(f"  {enhanced_timestamped}")
        print(f"  {enhanced_stable}")

        baseline_path = Path("evaluation/results/baseline.json")
        if baseline_path.exists():
            try:
                import json
                with open(baseline_path, "r", encoding="utf-8") as f:
                    baseline = json.load(f)
                base_avg = baseline.get("average_metrics", {})
                enh_avg = results.get("average_metrics", {})

                base_p = float(base_avg.get("precision", 0) or 0)
                enh_p = float(enh_avg.get("precision", 0) or 0)
                improvement = ((enh_p - base_p) / base_p * 100) if base_p != 0 else 0.0

                print("\n📈 Precision Improvement vs baseline.json:")
                print(f"  Baseline:  {base_p:.3f}")
                print(f"  Enhanced:  {enh_p:.3f}")
                print(f"  Change:    {improvement:+.1f}%")
            except Exception as e:
                print(f"⚠️ Could not compute improvement vs baseline.json: {e}")
        else:
            print("\nℹ️ baseline.json not found yet. Run scripts/run_baseline.py first for comparisons.")

        print("\n✅ Enhanced evaluation complete!")

    except Exception as e:
        print(f"❌ Enhanced evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
