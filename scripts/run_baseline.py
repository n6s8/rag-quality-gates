#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rag.rag_pipeline_rest import RAGPipeline, AnalysisMode
    from evaluation.evaluator import RAGEvaluator
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from project root directory")
    IMPORTS_OK = False


def print_metrics_summary(results, title="Evaluation Results"):
    avg = results.get('average_metrics', {})

    print(f"\n📊 {title}")
    print("-" * 50)

    print(f"  Retrieval Precision:     {avg.get('precision', 0):.3f}")
    print(f"  Retrieval Recall:        {avg.get('recall', 0):.3f}")
    print(f"  Answer Relevance:        {avg.get('relevance', 0):.3f}")
    print(f"  Response Time:           {avg.get('response_time', 0):.3f}s")

    if 'interpretation_score' in avg:
        print("\n🧠 ANALYSIS CAPABILITIES:")
        print(f"  Interpretation Score:    {avg.get('interpretation_score', 0):.3f}")
        print(f"  Historical Context:      {avg.get('historical_context_score', 0):.3f}")
        print(f"  Explanation Depth:       {avg.get('explanation_depth', 0):.3f}")
        print(f"  Thematic Analysis:       {avg.get('thematic_analysis', 0):.3f}")
        print(f"  Interpretation Quality:  {avg.get('interpretation_quality', 0):.3f}")


def run_standard_evaluation(pipeline, evaluator_class, analysis_mode="standard"):
    print("\n" + "="*60)
    print("📊 STANDARD BASELINE EVALUATION")
    print("="*60)

    evaluator = evaluator_class(pipeline)
    results = evaluator.run_evaluation(
        top_k=3,
        include_analysis_metrics=True,
        analysis_mode=None
    )

    print_metrics_summary(results, "Standard Mode Results")

    os.makedirs("evaluation/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"evaluation/results/baseline_standard_{timestamp}.json"
    evaluator.save_results(results, output_path)

    stable_path = "evaluation/results/baseline.json"
    evaluator.save_results(results, stable_path)

    return results, output_path


def run_comprehensive_analysis_evaluation(pipeline, evaluator_class):
    print("\n" + "="*60)
    print("🔍 COMPREHENSIVE ANALYSIS EVALUATION")
    print("="*60)

    evaluator = evaluator_class(pipeline)
    results = evaluator.run_evaluation(
        top_k=3,
        include_analysis_metrics=True,
        analysis_mode="comprehensive"
    )

    print_metrics_summary(results, "Comprehensive Analysis Mode Results")

    os.makedirs("evaluation/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"evaluation/results/baseline_comprehensive_{timestamp}.json"
    evaluator.save_results(results, output_path)

    stable_path = "evaluation/results/baseline_comprehensive.json"
    evaluator.save_results(results, stable_path)

    return results, output_path


def evaluate_interpretation_capabilities(pipeline, evaluator_class):
    print("\n" + "="*60)
    print("🧠 INTERPRETATION CAPABILITIES ASSESSMENT")
    print("="*60)

    evaluator = evaluator_class(pipeline)

    interpretation_questions = [
        {
            "question": "What does Roosevelt's 'fear itself' quote mean?",
            "expected_answer": "Roosevelt meant that fear can paralyze people more than actual danger. The quote encouraged facing the Great Depression with courage rather than panic.",
            "expected_quote_ids": [1],
            "expected_authors": ["Franklin D. Roosevelt"],
            "category": "interpretation"
        },
        {
            "question": "Explain the historical significance of Martin Luther King's 'I have a dream' speech",
            "expected_answer": "The speech galvanized the Civil Rights Movement, influenced public opinion, helped push civil rights legislation, and remains a defining moment in American history.",
            "expected_quote_ids": [2],
            "expected_authors": ["Martin Luther King Jr."],
            "category": "historical_context"
        },
        {
            "question": "What is Gandhi trying to say with 'Be the change you wish to see in the world'?",
            "expected_answer": "Gandhi emphasized that social transformation begins with individual action and personal responsibility. Each individual must embody the values they want to see in society.",
            "expected_quote_ids": [5],
            "expected_authors": ["Mahatma Gandhi"],
            "category": "interpretation"
        }
    ]

    assessment = evaluator.evaluate_interpretation_capabilities(interpretation_questions)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"evaluation/results/interpretation_assessment_{timestamp}.json"

    os.makedirs("evaluation/results", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(assessment, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Interpretation assessment saved to: {output_path}")

    return assessment, output_path


def run_comparative_analysis(pipeline_class, evaluator_class):
    print("\n" + "="*60)
    print("🔬 COMPARATIVE ANALYSIS: STANDARD vs COMPREHENSIVE")
    print("="*60)

    try:
        standard_pipeline = pipeline_class()
        comprehensive_pipeline = pipeline_class()

        pipeline_variants = [
            ("Standard Mode", standard_pipeline),
            ("Comprehensive Mode", comprehensive_pipeline)
        ]

        evaluator = evaluator_class(standard_pipeline)
        comparative_results = evaluator.run_comparative_evaluation(pipeline_variants, top_k=3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"evaluation/results/comparative_analysis_{timestamp}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparative_results, f, indent=2, ensure_ascii=False)

        print(f"\n📁 Comparative analysis saved to: {output_path}")

        return comparative_results, output_path

    except Exception as e:
        print(f"❌ Comparative analysis failed: {e}")
        return None, None


def generate_summary_report(all_results):
    summary = {
        "timestamp": datetime.now().isoformat(),
        "baseline_evaluation_summary": {},
        "interpretation_capabilities": {},
        "recommendations": []
    }

    if 'standard' in all_results:
        std_avg = all_results['standard']['average_metrics']
        summary['baseline_evaluation_summary']['standard_mode'] = {
            "precision": std_avg.get('precision', 0),
            "recall": std_avg.get('recall', 0),
            "relevance": std_avg.get('relevance', 0),
            "interpretation_score": std_avg.get('interpretation_score', 0),
            "historical_context_score": std_avg.get('historical_context_score', 0),
            "explanation_depth": std_avg.get('explanation_depth', 0)
        }

    if 'comprehensive' in all_results:
        comp_avg = all_results['comprehensive']['average_metrics']
        summary['baseline_evaluation_summary']['comprehensive_mode'] = {
            "precision": comp_avg.get('precision', 0),
            "recall": comp_avg.get('recall', 0),
            "relevance": comp_avg.get('relevance', 0),
            "interpretation_score": comp_avg.get('interpretation_score', 0),
            "historical_context_score": comp_avg.get('historical_context_score', 0),
            "explanation_depth": comp_avg.get('explanation_depth', 0)
        }

    if 'interpretation_assessment' in all_results:
        interp = all_results['interpretation_assessment']
        summary['interpretation_capabilities'] = {
            "capable": interp.get('capable', False),
            "average_score": interp.get('average_score', 0),
            "assessment_level": (
                "excellent" if interp.get('average_score', 0) >= 0.7 else
                "good" if interp.get('average_score', 0) >= 0.5 else
                "needs_improvement"
            )
        }

    recommendations = []

    if 'standard' in all_results:
        std_avg = all_results['standard']['average_metrics']

        if std_avg.get('precision', 0) < 0.5:
            recommendations.append("Consider improving retrieval precision through better keyword boosting or re-ranking")

        if std_avg.get('interpretation_score', 0) < 0.3:
            recommendations.append("Improve quote interpretation by adding more analysis data to the dataset")

        if std_avg.get('historical_context_score', 0) < 0.3:
            recommendations.append("Enhance historical context retrieval by adding historical significance fields")

    summary['recommendations'] = recommendations

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"evaluation/results/baseline_summary_{timestamp}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Summary report saved to: {output_path}")

    return summary


def main():
    if not IMPORTS_OK:
        print("❌ Cannot run evaluation due to import errors")
        return

    print("🚀 Running Enhanced Baseline Evaluation")
    print("="*60)

    try:
        print("🔧 Initializing RAG Pipeline...")
        pipeline = RAGPipeline()
        print("✅ Pipeline initialized")

        all_results = {}

        standard_results, standard_path = run_standard_evaluation(pipeline, RAGEvaluator)
        all_results['standard'] = standard_results
        all_results['standard_path'] = standard_path

        try:
            comprehensive_results, comprehensive_path = run_comprehensive_analysis_evaluation(pipeline, RAGEvaluator)
            all_results['comprehensive'] = comprehensive_results
            all_results['comprehensive_path'] = comprehensive_path
        except Exception as e:
            print(f"⚠️ Comprehensive analysis evaluation skipped: {e}")

        try:
            interpretation_assessment, interp_path = evaluate_interpretation_capabilities(pipeline, RAGEvaluator)
            all_results['interpretation_assessment'] = interpretation_assessment
            all_results['interpretation_path'] = interp_path
        except Exception as e:
            print(f"⚠️ Interpretation assessment skipped: {e}")

        try:
            comparative_results, comparative_path = run_comparative_analysis(RAGPipeline, RAGEvaluator)
            all_results['comparative'] = comparative_results
            all_results['comparative_path'] = comparative_path
        except Exception as e:
            print(f"⚠️ Comparative analysis skipped: {e}")

        summary = generate_summary_report(all_results)

        print("\n" + "="*60)
        print("✅ ENHANCED BASELINE EVALUATION COMPLETE!")
        print("="*60)

        print("\n📁 Results saved to:")
        for key, data in all_results.items():
            if key.endswith('_path') and data:
                print(f"  {key}: {data}")

        if 'interpretation_assessment' in all_results:
            interp = all_results['interpretation_assessment']
            print(f"\n🧠 Interpretation Capability: {'✅ CAPABLE' if interp.get('capable') else '❌ NEEDS IMPROVEMENT'}")
            print(f"   Average Score: {interp.get('average_score', 0):.3f}")

        print("\n📊 Next Steps:")
        print("  1. Review detailed results in evaluation/results/")
        print("  2. Check interpretation_assessment_*.json for interpretation capabilities")
        print("  3. Compare standard vs comprehensive modes in comparative_analysis_*.json")
        print("  4. Use insights to enhance the system for interpretation tasks")

        print("\n🔧 To test specific interpretation questions:")
        print("  from rag.rag_pipeline_rest import AnalysisMode")
        print("  pipeline.process_query('What does this quote mean?', analysis_mode=AnalysisMode.COMPREHENSIVE)")

    except Exception as e:
        print(f"❌ Baseline evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
