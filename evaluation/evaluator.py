import json
import time
from typing import Dict, List, Any, Optional, Tuple
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
    
    def _analyze_answer_quality(self, answer: str, question: str, docs: List[Dict]) -> Dict[str, float]:
        analysis_scores = {
            "interpretation_score": 0.0,
            "historical_context_score": 0.0,
            "explanation_depth": 0.0,
            "thematic_analysis": 0.0
        }
        
        if not answer or not docs:
            return analysis_scores
        
        answer_lower = answer.lower()
        
        interpretation_keywords = [
            "means", "meaning", "interpret", "signifies", "suggests", 
            "implies", "indicates", "represents", "symbolizes", "conveys"
        ]
        
        interpretation_count = sum(1 for keyword in interpretation_keywords if keyword in answer_lower)
        analysis_scores["interpretation_score"] = min(interpretation_count / 2, 1.0)
        
        context_keywords = [
            "historical", "context", "era", "period", "century", "year",
            "during", "when", "background", "circumstances", "situation",
            "event", "movement", "war", "depression", "rights", "struggle"
        ]
        
        context_count = sum(1 for keyword in context_keywords if keyword in answer_lower)
        analysis_scores["historical_context_score"] = min(context_count / 3, 1.0)
        
        explanation_indicators = [
            "because", "therefore", "thus", "hence", "consequently",
            "as a result", "in order to", "so that", "due to", "owing to",
            "explain", "analysis", "analyze", "discuss", "elaborate",
            "detail", "comprehensive", "thorough", "depth"
        ]
        
        explanation_count = sum(1 for indicator in explanation_indicators if indicator in answer_lower)
        analysis_scores["explanation_depth"] = min(explanation_count / 3, 1.0)
        
        thematic_keywords = [
            "theme", "thematic", "concept", "idea", "philosophy",
            "principle", "value", "belief", "message", "moral",
            "lesson", "insight", "perspective", "viewpoint", "approach"
        ]
        
        thematic_count = sum(1 for keyword in thematic_keywords if keyword in answer_lower)
        analysis_scores["thematic_analysis"] = min(thematic_count / 2, 1.0)
        
        doc_has_interpretation = any(doc.get('interpretation') for doc in docs)
        doc_has_context = any(doc.get('historical_significance') for doc in docs)
        
        if doc_has_interpretation and analysis_scores["interpretation_score"] > 0:
            analysis_scores["interpretation_score"] = min(analysis_scores["interpretation_score"] + 0.2, 1.0)
        
        if doc_has_context and analysis_scores["historical_context_score"] > 0:
            analysis_scores["historical_context_score"] = min(analysis_scores["historical_context_score"] + 0.2, 1.0)
        
        return analysis_scores
    
    def _evaluate_interpretation_specific(self, question: str, answer: str, expected_answer: str) -> Dict[str, Any]:
        interpretation_keywords = ["mean", "meaning", "interpret", "significance", "explain"]
        
        is_interpretation_question = any(keyword in question.lower() for keyword in interpretation_keywords)
        
        if not is_interpretation_question:
            return {
                "is_interpretation_question": False,
                "interpretation_quality": 0.0,
                "has_interpretation": False
            }
        
        answer_lower = answer.lower()
        
        interpretation_markers = [
            "means that", "this means", "signifies that", "indicates that",
            "suggests that", "represents", "symbolizes", "conveys that"
        ]
        
        has_interpretation_marker = any(marker in answer_lower for marker in interpretation_markers)
        
        answer_length_score = min(len(answer.split()) / 100, 1.0)
        
        has_because = "because" in answer_lower
        has_therefore = "therefore" in answer_lower or "thus" in answer_lower
        
        interpretation_quality = 0.0
        if has_interpretation_marker:
            interpretation_quality += 0.4
        if has_because or has_therefore:
            interpretation_quality += 0.3
        interpretation_quality += (answer_length_score * 0.3)
        
        return {
            "is_interpretation_question": True,
            "interpretation_quality": min(interpretation_quality, 1.0),
            "has_interpretation": has_interpretation_marker,
            "has_causal_explanation": has_because or has_therefore,
            "answer_length_words": len(answer.split())
        }
    
    def run_evaluation(
        self, 
        top_k: int = 3, 
        include_analysis_metrics: bool = True,
        analysis_mode: Optional[str] = None
    ) -> Dict[str, Any]:
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
        
        if include_analysis_metrics:
            analysis_metrics = {
                "interpretation_score": [],
                "historical_context_score": [],
                "explanation_depth": [],
                "thematic_analysis": [],
                "interpretation_quality": []
            }
        
        print(f"\n📊 Evaluating {len(self.eval_data)} test queries...")
        if analysis_mode:
            print(f"📋 Analysis Mode: {analysis_mode}")
        
        for i, test_case in enumerate(self.eval_data):
            print(f"\n{'='*50}")
            print(f"Test #{i+1}: {test_case['question'][:60]}...")
            
            try:
                start_time = time.time()
                
                if analysis_mode:
                    try:
                        from rag.rag_pipeline_rest import AnalysisMode
                        mode_map = {
                            "basic": AnalysisMode.BASIC,
                            "standard": AnalysisMode.STANDARD,
                            "comprehensive": AnalysisMode.COMPREHENSIVE,
                            "comparative": AnalysisMode.COMPARATIVE
                        }
                        mode = mode_map.get(analysis_mode.lower(), AnalysisMode.STANDARD)
                        result = self.rag_pipeline.process_query(
                            test_case['question'], 
                            top_k=top_k,
                            analysis_mode=mode
                        )
                    except:
                        result = self.rag_pipeline.process_query(test_case['question'], top_k=top_k)
                else:
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
                    },
                    "pipeline_info": {
                        "analysis_mode": result.get('analysis_mode', 'standard'),
                        "answer_type": result.get('answer_type', 'unknown'),
                        "has_interpretation_data": result.get('has_interpretation', False),
                        "has_historical_context_data": result.get('has_historical_context', False),
                        "includes_analysis": result.get('includes_analysis', False)
                    }
                }
                
                if include_analysis_metrics:
                    analysis_scores = self._analyze_answer_quality(
                        result.get('answer', ''),
                        test_case['question'],
                        result.get('search_results', [])
                    )
                    
                    interpretation_eval = self._evaluate_interpretation_specific(
                        test_case['question'],
                        result.get('answer', ''),
                        test_case['expected_answer']
                    )
                    
                    test_result["analysis_metrics"] = analysis_scores
                    test_result["interpretation_evaluation"] = interpretation_eval
                    
                    for metric_name, score in analysis_scores.items():
                        if metric_name in analysis_metrics:
                            analysis_metrics[metric_name].append(score)
                    
                    if interpretation_eval["is_interpretation_question"]:
                        analysis_metrics["interpretation_quality"].append(
                            interpretation_eval["interpretation_quality"]
                        )
                
                results.append(test_result)
                
                for key in total_metrics:
                    total_metrics[key].append(test_result['metrics'][key])
                
                print(f"✅ Retrieved IDs: {retrieved_ids}")
                print(f"📊 Precision: {precision:.2f}, Recall: {recall:.2f}")
                print(f"📝 Answer Relevance: {relevance:.2f}")
                
                if include_analysis_metrics and 'analysis_metrics' in test_result:
                    print(f"🔍 Analysis Scores: ", end="")
                    for metric, score in test_result['analysis_metrics'].items():
                        if score > 0:
                            print(f"{metric}: {score:.2f} ", end="")
                    print()
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_metrics = {}
        for key, values in total_metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
            else:
                avg_metrics[key] = 0.0
        
        if include_analysis_metrics:
            avg_analysis_metrics = {}
            for key, values in analysis_metrics.items():
                if values:
                    avg_analysis_metrics[key] = sum(values) / len(values)
                else:
                    avg_analysis_metrics[key] = 0.0
            
            avg_metrics.update(avg_analysis_metrics)
        
        return {
            "test_results": results,
            "average_metrics": avg_metrics,
            "config": {
                "top_k": top_k,
                "num_queries": len(self.eval_data),
                "successful_queries": len(results),
                "include_analysis_metrics": include_analysis_metrics,
                "analysis_mode": analysis_mode
            }
        }
    
    def run_comparative_evaluation(
        self, 
        pipeline_variants: List[Tuple[str, Any]], 
        top_k: int = 3
    ) -> Dict[str, Any]:
        print(f"\n🔬 Running Comparative Evaluation of {len(pipeline_variants)} pipeline variants")
        print("="*60)
        
        comparative_results = {}
        
        for pipeline_name, pipeline in pipeline_variants:
            print(f"\n📋 Evaluating: {pipeline_name}")
            print("-"*40)
            
            evaluator = RAGEvaluator(pipeline, "data/eval_dataset.json")
            results = evaluator.run_evaluation(top_k=top_k, include_analysis_metrics=True)
            
            comparative_results[pipeline_name] = {
                "average_metrics": results["average_metrics"],
                "config": results["config"]
            }
            
            avg = results["average_metrics"]
            print(f"  Precision: {avg.get('precision', 0):.3f}")
            print(f"  Interpretation Score: {avg.get('interpretation_score', 0):.3f}")
            print(f"  Historical Context: {avg.get('historical_context_score', 0):.3f}")
        
        return comparative_results
    
    def evaluate_interpretation_capabilities(self, interpretation_questions: List[Dict] = None) -> Dict[str, Any]:
        if interpretation_questions is None:
            interpretation_questions = [
                {
                    "question": "What does Roosevelt's 'fear itself' quote mean?",
                    "expected_answer": "Should include interpretation of the quote's meaning",
                    "category": "interpretation"
                },
                {
                    "question": "Explain the significance of Martin Luther King's 'I have a dream' speech",
                    "expected_answer": "Should include historical context and significance",
                    "category": "historical_context"
                },
                {
                    "question": "What is Gandhi trying to say with 'Be the change'?",
                    "expected_answer": "Should explain the quote's philosophical meaning",
                    "category": "interpretation"
                }
            ]
        
        print(f"\n🧠 Evaluating Interpretation Capabilities")
        print("="*60)
        
        interpretation_results = []
        interpretation_scores = []
        
        for i, question_data in enumerate(interpretation_questions):
            print(f"\nQuestion {i+1}: {question_data['question']}")
            
            try:
                result = self.rag_pipeline.process_query(
                    question_data['question'],
                    top_k=2,
                    analysis_mode="comprehensive"
                )
                
                interpretation_eval = self._evaluate_interpretation_specific(
                    question_data['question'],
                    result.get('answer', ''),
                    question_data['expected_answer']
                )
                
                analysis_scores = self._analyze_answer_quality(
                    result.get('answer', ''),
                    question_data['question'],
                    result.get('search_results', [])
                )
                
                question_result = {
                    "question": question_data['question'],
                    "category": question_data['category'],
                    "answer": result.get('answer', '')[:200] + "..." if len(result.get('answer', '')) > 200 else result.get('answer', ''),
                    "interpretation_evaluation": interpretation_eval,
                    "analysis_scores": analysis_scores,
                    "overall_interpretation_score": (
                        interpretation_eval["interpretation_quality"] * 0.5 +
                        analysis_scores["interpretation_score"] * 0.3 +
                        analysis_scores["historical_context_score"] * 0.2
                    )
                }
                
                interpretation_results.append(question_result)
                interpretation_scores.append(question_result["overall_interpretation_score"])
                
                print(f"  Interpretation Quality: {interpretation_eval['interpretation_quality']:.2f}")
                print(f"  Overall Score: {question_result['overall_interpretation_score']:.2f}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        avg_interpretation_score = sum(interpretation_scores) / len(interpretation_scores) if interpretation_scores else 0
        
        interpretation_assessment = {
            "capable": avg_interpretation_score >= 0.5,
            "average_score": avg_interpretation_score,
            "score_breakdown": {
                "excellent": avg_interpretation_score >= 0.7,
                "good": 0.5 <= avg_interpretation_score < 0.7,
                "needs_improvement": avg_interpretation_score < 0.5
            },
            "question_results": interpretation_results
        }
        
        print(f"\n📊 Overall Interpretation Capability: {avg_interpretation_score:.2f}")
        print(f"   Assessment: {'✅ CAPABLE' if interpretation_assessment['capable'] else '❌ NEEDS IMPROVEMENT'}")
        
        return interpretation_assessment
    
    def save_results(self, results: Dict, output_path: str):
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"📁 Results saved to: {output_path}")
            
            summary_path = output_path.replace('.json', '_summary.txt')
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("RAG EVALUATION SUMMARY\n")
                f.write("=" * 40 + "\n")
                
                f.write("\nBASIC METRICS:\n")
                basic_metrics = ["precision", "recall", "relevance", "hallucination", "response_time"]
                for metric in basic_metrics:
                    if metric in results['average_metrics']:
                        f.write(f"{metric:20} {results['average_metrics'][metric]:.3f}\n")
                
                f.write("\nANALYSIS METRICS:\n")
                analysis_metrics = ["interpretation_score", "historical_context_score", 
                                  "explanation_depth", "thematic_analysis", "interpretation_quality"]
                for metric in analysis_metrics:
                    if metric in results['average_metrics']:
                        f.write(f"{metric:20} {results['average_metrics'][metric]:.3f}\n")
                
                f.write(f"\nCONFIG:\n")
                for key, value in results['config'].items():
                    f.write(f"{key:20} {value}\n")
            
            print(f"📋 Summary saved to: {summary_path}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


if __name__ == "__main__":
    print("🧪 Testing Enhanced RAG Evaluator with Analysis Metrics")
    print("="*60)
    
    class MockPipeline:
        def process_query(self, question, top_k=3, analysis_mode=None):
            return {
                "answer": f"This is a comprehensive analysis of '{question}'. The quote means that we should face challenges courageously. Historically, this was said during difficult times to inspire people.",
                "search_results": [
                    {"id": 1, "author": "Test Author", "quote": "Test quote", "interpretation": "Test interpretation"}
                ],
                "analysis_mode": "comprehensive" if analysis_mode else "standard",
                "answer_type": "comprehensive_analysis",
                "has_interpretation": True,
                "has_historical_context": True,
                "includes_analysis": True
            }
    
    pipeline = MockPipeline()
    evaluator = RAGEvaluator(pipeline)
    
    test_answer = "The quote means that we should not be afraid of fear itself. Historically, Roosevelt said this during the Great Depression to inspire confidence."
    test_question = "What does Roosevelt's quote mean?"
    test_docs = [{"interpretation": "Test", "historical_significance": "Test"}]
    
    analysis_scores = evaluator._analyze_answer_quality(test_answer, test_question, test_docs)
    print(f"\nAnalysis Scores Test:")
    for metric, score in analysis_scores.items():
        print(f"  {metric}: {score:.2f}")
    
    interpretation_eval = evaluator._evaluate_interpretation_specific(
        "What does this quote mean?",
        test_answer,
        "Expected answer"
    )
    print(f"\nInterpretation Evaluation:")
    for key, value in interpretation_eval.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Enhanced Evaluator test complete!")
