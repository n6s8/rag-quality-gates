"""
Enhanced RAG Pipeline with Analysis Capabilities
Supports both basic retrieval and comprehensive historical analysis
"""
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import re

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.llm.llm_client import llm_client


class AnalysisMode(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    COMPARATIVE = "comparative"


class RAGPipeline:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "historical_quotes",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        default_analysis_mode: AnalysisMode = AnalysisMode.STANDARD,
        use_enhanced: bool = False,
        keyword_boost: float = 0.5,
        analysis_doc_boost: float = 0.3,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.default_analysis_mode = default_analysis_mode

        self.use_enhanced = use_enhanced
        self.keyword_boost = keyword_boost
        self.analysis_doc_boost = analysis_doc_boost

        self.client = QdrantClient(host=self.host, port=self.port)
        self.embedder = SentenceTransformer(self.embedding_model_name)

        print(f"✅ RAG Pipeline initialized (Analysis Mode: {default_analysis_mode.value})")

    def get_database_stats(self) -> Dict[str, Any]:
        try:
            count_result = self.client.count(
                collection_name=self.collection_name,
                exact=True,
            )

            try:
                collection_info = self.client.get_collection(self.collection_name)
                vectors_count = int(count_result.count)
                vector_size = collection_info.config.params.vectors.size
                return {
                    "collection_name": self.collection_name,
                    "vectors_count": vectors_count,
                    "vector_size": vector_size,
                    "status": "healthy"
                }
            except Exception:
                return {
                    "collection_name": self.collection_name,
                    "vectors_count": int(count_result.count),
                    "status": "accessible"
                }

        except Exception as e:
            return {
                "error": str(e),
                "collection_name": self.collection_name,
                "vectors_count": 0,
                "status": "error"
            }

    def _hit_score(self, hit: Any) -> float:
        if hasattr(hit, "score") and hit.score is not None:
            try:
                return float(hit.score)
            except Exception:
                return 0.0
        if isinstance(hit, dict) and "score" in hit:
            try:
                return float(hit["score"])
            except Exception:
                return 0.0
        return 0.0

    def _hit_payload(self, hit: Any) -> Dict[str, Any]:
        if hasattr(hit, "payload") and hit.payload is not None:
            return hit.payload
        if isinstance(hit, dict):
            return hit.get("payload", {}) or {}
        return {}

    def _build_search_payload(self, hit: Any, include_analysis_fields: bool = True) -> Dict[str, Any]:
        payload = self._hit_payload(hit)

        base_payload = {
            "id": payload.get("id"),
            "quote": payload.get("quote", ""),
            "author": payload.get("author", "Unknown"),
            "era": payload.get("era", ""),
            "topic": payload.get("topic", ""),
            "context": payload.get("context", ""),
            "source": payload.get("source", ""),
            "tags": payload.get("tags", []),
            "language": payload.get("language", "English"),
            "score": self._hit_score(hit),
        }

        if include_analysis_fields:
            analysis_fields = {
                "interpretation": payload.get("interpretation", ""),
                "historical_significance": payload.get("historical_significance", ""),
                "themes": payload.get("themes", ""),
                "key_phrases": payload.get("key_phrases", []),
                "modern_relevance": payload.get("modern_relevance", "")
            }
            base_payload.update(analysis_fields)

        return base_payload

    def _select_prompt_method(self, analysis_mode: AnalysisMode, docs: List[Dict]) -> str:
        has_analysis_fields = any(
            doc.get("interpretation") or doc.get("historical_significance")
            for doc in docs
        )

        if analysis_mode == AnalysisMode.COMPREHENSIVE and has_analysis_fields:
            return "format_rag_prompt_with_analysis"
        if analysis_mode == AnalysisMode.STANDARD:
            return "format_rag_prompt"
        if analysis_mode == AnalysisMode.BASIC:
            return "format_simple_prompt"
        if analysis_mode == AnalysisMode.COMPARATIVE and len(docs) >= 2:
            return "comparative"
        return "format_rag_prompt_with_analysis" if has_analysis_fields else "format_rag_prompt"

    def _generate_comparative_analysis(self, question: str, docs: List[Dict]) -> str:
        if len(docs) < 2:
            return "Need at least two quotes for comparative analysis."

        if hasattr(llm_client, "compare_quotes"):
            comparison = llm_client.compare_quotes(docs[0], docs[1])
            return f"Comparative Analysis of Two Key Quotes:\n\n{comparison}"

        prompt = llm_client.format_rag_prompt_with_analysis(
            f"Compare and contrast these quotes in response to: {question}",
            docs
        )
        return llm_client.generate_response(prompt)

    def _extract_keywords(self, question: str) -> List[str]:
        q = question.lower()
        stop = {
            "what", "who", "when", "where", "why", "how", "said", "say",
            "about", "some", "the", "a", "an", "and", "or", "but", "in",
            "on", "at", "to", "for", "of", "with", "by", "does", "did",
            "do", "can", "could", "would", "should", "there", "are", "is",
            "was", "were", "jr"
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", q)
        keywords = [w for w in words if w not in stop and not w.isdigit()]
        return list(dict.fromkeys(keywords))

    def _question_needs_analysis(self, question: str) -> bool:
        q = question.lower()
        triggers = [
            "mean", "meaning", "interpret", "explain", "significance",
            "historical", "context", "background", "analyze", "analysis",
            "what does", "why did", "how does"
        ]
        return any(t in q for t in triggers)

    def _select_effective_top_k(self, question: str, requested_top_k: int, mode: AnalysisMode) -> int:
        if not self.use_enhanced:
            return requested_top_k

        q = question.lower().strip()

        is_topic_search = ("quotes" in q) and (
            "about" in q or "are there" in q or "some" in q or "list" in q
        )

        is_exact_or_author = (
            ("who said" in q) or
            (("what did" in q or "what was" in q) and ("quotes" not in q)) or
            (("'" in question or '"' in question) and ("quotes" not in q)) or
            (("dream about" in q) and ("quotes" not in q))
        )

        if is_exact_or_author and not is_topic_search:
            return 1

        if is_topic_search:
            return requested_top_k

        if mode == AnalysisMode.COMPREHENSIVE and self._question_needs_analysis(question):
            return min(max(requested_top_k, 2), 3)

        return requested_top_k

    def _apply_enhanced_boosting(self, docs: List[Dict[str, Any]], question: str, mode: AnalysisMode) -> List[Dict[str, Any]]:
        if not docs:
            return docs

        keywords = self._extract_keywords(question)
        needs_analysis = (mode == AnalysisMode.COMPREHENSIVE) or self._question_needs_analysis(question)

        boosted = []
        for d in docs:
            s = float(d.get("score", 0.0))
            blob = " ".join([
                str(d.get("author", "")).lower(),
                str(d.get("topic", "")).lower(),
                str(d.get("quote", "")).lower(),
                " ".join([str(x).lower() for x in (d.get("tags") or [])]),
            ])

            if keywords:
                for kw in keywords:
                    if kw in blob:
                        s += self.keyword_boost

            has_analysis = bool(d.get("interpretation") or d.get("historical_significance"))
            if needs_analysis and has_analysis:
                s += self.analysis_doc_boost

            d2 = dict(d)
            d2["score"] = float(s)
            boosted.append(d2)

        boosted.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return boosted

    def _qdrant_search(self, vector: List[float], limit: int) -> List[Any]:
        if hasattr(self.client, "query_points"):
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            points = getattr(resp, "points", None)
            return points or []

        if hasattr(self.client, "search_points"):
            return self.client.search_points(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            ) or []

        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            ) or []

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=200,
            with_payload=True,
            with_vectors=False,
        )
        return points or []

    def search_quotes(
        self,
        question: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        try:
            vector = self.embedder.encode(question).tolist()

            effective_top_k = self._select_effective_top_k(question, top_k, self.default_analysis_mode)
            hits = self._qdrant_search(vector, limit=effective_top_k * 2)

            docs = [self._build_search_payload(hit) for hit in hits]

            unique_docs = []
            seen_quotes = set()
            for doc in docs:
                quote_key = (doc.get("quote", "")[:100]).strip()
                if quote_key and quote_key not in seen_quotes:
                    seen_quotes.add(quote_key)
                    unique_docs.append(doc)

            if self.use_enhanced:
                unique_docs = self._apply_enhanced_boosting(unique_docs, question, self.default_analysis_mode)

            return unique_docs[:effective_top_k]

        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    def process_query(
        self,
        question: str,
        top_k: int = 3,
        analysis_mode: Optional[Union[AnalysisMode, str]] = None,
        include_analysis: bool = True
    ) -> Dict[str, Any]:
        try:
            if analysis_mode is None:
                mode = self.default_analysis_mode
            elif isinstance(analysis_mode, str):
                mode_map = {
                    "basic": AnalysisMode.BASIC,
                    "standard": AnalysisMode.STANDARD,
                    "comprehensive": AnalysisMode.COMPREHENSIVE,
                    "comparative": AnalysisMode.COMPARATIVE
                }
                mode = mode_map.get(analysis_mode.lower(), self.default_analysis_mode)
            elif isinstance(analysis_mode, AnalysisMode):
                mode = analysis_mode
            else:
                mode = self.default_analysis_mode

            effective_top_k = self._select_effective_top_k(question, top_k, mode)

            print(f"🔍 Processing query ({mode.value} analysis): '{question}'")

            vector = self.embedder.encode(question).tolist()

            hits = self._qdrant_search(vector, limit=effective_top_k * 2)

            docs = []
            seen_ids = set()

            for hit in hits:
                doc = self._build_search_payload(hit, include_analysis_fields=include_analysis)
                if doc.get("id") is None:
                    continue
                if doc["id"] not in seen_ids:
                    seen_ids.add(doc["id"])
                    docs.append(doc)

            if self.use_enhanced:
                docs = self._apply_enhanced_boosting(docs, question, mode)

            docs = docs[:effective_top_k]

            print(f"✅ Found {len(docs)} relevant quotes")

            prompt_method = self._select_prompt_method(mode, docs)

            if prompt_method == "format_rag_prompt_with_analysis":
                prompt = llm_client.format_rag_prompt_with_analysis(question, docs)
                answer_type = "comprehensive_analysis"
            elif prompt_method == "format_simple_prompt":
                prompt = llm_client.format_simple_prompt(question, docs)
                answer_type = "basic_answer"
            elif prompt_method == "comparative":
                answer = self._generate_comparative_analysis(question, docs)
                answer_type = "comparative_analysis"
                return {
                    "answer": answer,
                    "retrieved_count": len(docs),
                    "search_results": docs,
                    "analysis_mode": mode.value,
                    "answer_type": answer_type,
                    "has_interpretation": any(d.get("interpretation") for d in docs),
                    "has_historical_context": any(d.get("historical_significance") for d in docs),
                    "includes_analysis": True,
                    "used_top_k": effective_top_k
                }
            else:
                prompt = llm_client.format_rag_prompt(question, docs)
                answer_type = "standard_answer"

            answer = llm_client.generate_response(
                prompt,
                max_tokens=400 if mode == AnalysisMode.COMPREHENSIVE else 200
            )

            has_analysis_in_answer = any(
                keyword in answer.lower()
                for keyword in ["means", "significance", "context", "interpret", "analyze", "themes"]
            )

            return {
                "answer": answer,
                "retrieved_count": len(docs),
                "search_results": docs,
                "analysis_mode": mode.value,
                "answer_type": answer_type,
                "has_interpretation": any(d.get("interpretation") for d in docs),
                "has_historical_context": any(d.get("historical_significance") for d in docs),
                "includes_analysis": has_analysis_in_answer,
                "prompt_method": prompt_method,
                "used_top_k": effective_top_k
            }

        except Exception as e:
            error_text = f"Error during RAG processing: {e}"
            print(f"❌ {error_text}")
            import traceback
            traceback.print_exc()
            return {
                "answer": error_text,
                "retrieved_count": 0,
                "search_results": [],
                "analysis_mode": "error",
                "error": True
            }

    def analyze_single_quote(self, quote_id: int) -> Dict[str, Any]:
        try:
            try:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter={
                        "must": [
                            {"key": "id", "match": {"value": quote_id}}
                        ]
                    },
                    limit=1,
                    with_payload=True
                )
            except Exception:
                points, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=200,
                    with_payload=True
                )
                found = None
                for p in points or []:
                    payload = getattr(p, "payload", {}) or {}
                    if payload.get("id") == quote_id:
                        found = p
                        break
                if found is None:
                    return {"error": f"Quote with ID {quote_id} not found"}
                scroll_result = ([found], None)

            points, _ = scroll_result
            if not points:
                return {"error": f"Quote with ID {quote_id} not found"}

            hit = points[0]
            doc = self._build_search_payload(hit, include_analysis_fields=True)

            if doc.get("interpretation"):
                analysis = {
                    "quote": doc["quote"],
                    "author": doc["author"],
                    "interpretation": doc.get("interpretation", ""),
                    "historical_significance": doc.get("historical_significance", ""),
                    "themes": doc.get("themes", ""),
                    "modern_relevance": doc.get("modern_relevance", ""),
                    "source": "dataset"
                }
            else:
                analysis_prompt = f"""Analyze this historical quote in detail:

Quote: "{doc['quote']}"
Author: {doc['author']}
Context: {doc.get('context', 'Not specified')}
Era: {doc.get('era', 'Not specified')}

Please provide:
1. Interpretation of what the quote means
2. Historical context and significance
3. Key themes or ideas
4. Modern relevance

Detailed Analysis:"""

                generated_analysis = llm_client.generate_response(analysis_prompt, max_tokens=300)
                analysis = {
                    "quote": doc["quote"],
                    "author": doc["author"],
                    "interpretation": generated_analysis,
                    "historical_significance": doc.get("context", ""),
                    "themes": doc.get("topic", "").split(", "),
                    "modern_relevance": "Generated by AI analysis",
                    "source": "generated"
                }

            return {
                "success": True,
                "quote_id": quote_id,
                "analysis": analysis,
                "metadata": {k: v for k, v in doc.items() if k not in ["quote", "author"]}
            }

        except Exception as e:
            return {"error": str(e), "quote_id": quote_id}

    def compare_quotes(self, quote_ids: List[int]) -> Dict[str, Any]:
        try:
            quotes_data = []
            for quote_id in quote_ids[:3]:
                result = self.analyze_single_quote(quote_id)
                if "analysis" in result:
                    quotes_data.append(result["analysis"])

            if len(quotes_data) < 2:
                return {"error": "Need at least 2 quotes for comparison"}

            comparison_prompt = "Compare and contrast these historical quotes:\n\n"
            for i, quote_data in enumerate(quotes_data, 1):
                comparison_prompt += f"""QUOTE {i}:
"{quote_data['quote']}"
- Author: {quote_data['author']}
- Interpretation: {str(quote_data.get('interpretation', 'Not available'))[:200]}...
- Themes: {quote_data.get('themes', 'Not specified')}

"""

            comparison_prompt += """Please analyze:
1. Similarities in themes, messages, or historical context
2. Differences in approach, philosophy, or historical circumstances
3. How they complement or contrast with each other
4. Their collective significance

Comparative Analysis:"""

            comparison = llm_client.generate_response(comparison_prompt, max_tokens=400)

            return {
                "success": True,
                "compared_quotes": [qd["quote"][:50] + "..." for qd in quotes_data],
                "comparison_analysis": comparison,
                "individual_analyses": quotes_data
            }

        except Exception as e:
            return {"error": str(e)}


rag_pipeline = RAGPipeline(default_analysis_mode=AnalysisMode.STANDARD)
analysis_pipeline = RAGPipeline(default_analysis_mode=AnalysisMode.COMPREHENSIVE)


if __name__ == "__main__":
    print("🧪 Testing Enhanced RAG Pipeline with Analysis Capabilities")
    print("="*60)

    pipeline = RAGPipeline()

    stats = pipeline.get_database_stats()
    print(f"📊 Database Stats: {stats}")

    test_questions = [
        ("What does Roosevelt's 'fear itself' quote mean?", AnalysisMode.COMPREHENSIVE),
        ("What did Martin Luther King Jr. dream about?", AnalysisMode.STANDARD),
        ("List some quotes about perseverance", AnalysisMode.BASIC),
    ]

    for question, mode in test_questions:
        print(f"\n{'='*60}")
        print(f"Test: {question}")
        print(f"Mode: {mode.value}")

        result = pipeline.process_query(question, top_k=2, analysis_mode=mode)

        print(f"\nRetrieved: {result.get('retrieved_count', 0)} documents")
        print(f"Answer type: {result.get('answer_type', 'unknown')}")
        print(f"Has interpretation data: {result.get('has_interpretation', False)}")

        if result.get("search_results"):
            print("\nRetrieved Documents:")
            for i, doc in enumerate(result["search_results"]):
                print(f"  {i+1}. ID: {doc.get('id')}, Author: {doc.get('author')}")
                if doc.get("interpretation"):
                    print(f"     Has interpretation: Yes ({len(doc['interpretation'])} chars)")

        print(f"\nAnswer preview: {result.get('answer', '')[:150]}...")

    print(f"\n{'='*60}")
    print("Testing single quote analysis...")
    analysis_result = pipeline.analyze_single_quote(1)
    if "analysis" in analysis_result:
        print(f"Quote analysis for ID 1: {analysis_result['analysis'].get('interpretation', '')[:150]}...")

    print("\n✅ Enhanced RAG Pipeline test complete!")
