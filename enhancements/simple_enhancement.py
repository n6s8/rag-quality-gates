"""
Enhanced RAG with Analysis-Oriented Retrieval and Interpretation Support
Combines keyword boosting with analysis-aware document selection
"""
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from src.llm.llm_client import llm_client
import numpy as np
from typing import List, Dict, Any, Optional
import re


class EnhancedRAG:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "historical_quotes",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        analysis_boost: float = 0.3  # Additional boost for documents with analysis data
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.analysis_boost = analysis_boost
        
        self.client = QdrantClient(host=self.host, port=self.port)
        self.embedder = SentenceTransformer(self.embedding_model_name)
        
        print(f"✅ Enhanced RAG initialized (Analysis-Aware Retrieval, Boost: {analysis_boost})")
    
    def _extract_keywords_and_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract keywords AND detect query intent (retrieval vs interpretation)
        """
        query_lower = query.lower()
        keywords = []
        
        # Detect query intent
        is_interpretation_query = any(word in query_lower for word in [
            "mean", "meaning", "interpret", "explain", "significance",
            "analyze", "analysis", "what does", "why did", "how does"
        ])
        
        is_historical_query = any(word in query_lower for word in [
            "historical", "history", "context", "background", "era",
            "period", "century", "when", "during", "time"
        ])
        
        is_comparison_query = any(word in query_lower for word in [
            "compare", "contrast", "difference", "similar", "versus",
            "vs", "between", "both", "each"
        ])
        
        # Author names and variations
        author_mappings = {
            "roosevelt": ["franklin", "fdr", "eleanor"],
            "martin luther king": ["mlk", "king jr", "martin luther", "dr king"],
            "einstein": ["albert"],
            "gandhi": ["mahatma"],
            "mandela": ["nelson"],
            "churchill": ["winston"],
            "lincoln": ["abraham"],
            "aristotle": [],
            "confucius": [],
            "curie": ["marie"],
            "da vinci": ["leonardo"],
            "newton": ["isaac"],
            "edison": ["thomas"],
            "armstrong": ["neil"]
        }
        
        # Check for author mentions
        for author, variations in author_mappings.items():
            if author in query_lower:
                keywords.append(author)
                keywords.extend(variations)
            for variation in variations:
                if variation in query_lower:
                    keywords.append(author)
        
        # Topic keywords with synonyms
        topic_mappings = {
            "fear": ["afraid", "scared", "frightened", "anxiety"],
            "dream": ["dreamed", "dreamt", "aspiration", "vision"],
            "change": ["transform", "different", "alter", "modify"],
            "perseverance": ["persist", "never give up", "keep going", "resilience", "determination"],
            "leadership": ["leader", "govern", "government", "authority", "management"],
            "imagination": ["imagine", "creative", "creativity", "innovation", "invention"],
            "science": ["scientific", "knowledge", "learn", "discovery", "research"],
            "courage": ["brave", "bravery", "bold", "fearless", "heroic"],
            "equality": ["equal", "rights", "civil rights", "justice", "fairness"],
            "freedom": ["free", "liberty", "independence", "autonomy"],
            "success": ["achieve", "achievement", "accomplish", "victory", "triumph"],
            "failure": ["fail", "mistake", "error", "defeat", "loss"],
            "hope": ["hopeful", "optimistic", "expectation", "aspiration"]
        }
        
        for topic, synonyms in topic_mappings.items():
            if topic in query_lower:
                keywords.append(topic)
                keywords.extend(synonyms)
            for synonym in synonyms:
                if synonym in query_lower:
                    keywords.append(topic)
        
        # Extract important nouns and concepts
        stop_words = {"what", "who", "when", "where", "why", "how", "said", 
                     "say", "about", "some", "the", "a", "an", "and", "or", 
                     "but", "in", "on", "at", "to", "for", "of", "with", "by",
                     "does", "did", "do", "can", "could", "would", "should"}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query_lower)
        for word in words:
            if (word not in stop_words and 
                word not in keywords and
                not word.isdigit()):
                keywords.append(word)
        
        return {
            "keywords": list(set(keywords)),
            "intent": {
                "interpretation": is_interpretation_query,
                "historical": is_historical_query,
                "comparison": is_comparison_query,
                "type": "interpretation" if is_interpretation_query else 
                       "historical" if is_historical_query else
                       "comparison" if is_comparison_query else "retrieval"
            }
        }
    
    def _build_search_payload(self, payload, score=0.0, include_analysis=True):
        """Build document payload with optional analysis fields"""
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
            "score": score,
        }
        
        if include_analysis:
            # Add analysis fields if available
            analysis_fields = {
                "interpretation": payload.get("interpretation", ""),
                "historical_significance": payload.get("historical_significance", ""),
                "themes": payload.get("themes", ""),
                "key_phrases": payload.get("key_phrases", []),
                "modern_relevance": payload.get("modern_relevance", ""),
                "has_analysis": bool(payload.get("interpretation") or payload.get("historical_significance"))
            }
            base_payload.update(analysis_fields)
        
        return base_payload
    
    def _calculate_document_score(self, payload, query_vector, keywords, intent, question):
        """
        Calculate comprehensive document score considering:
        1. Semantic similarity
        2. Keyword matching
        3. Analysis data availability
        4. Query intent alignment
        """
        # 1. Semantic similarity
        doc_text = f"{payload.get('quote', '')} {payload.get('author', '')} {payload.get('topic', '')}"
        doc_vector = self.embedder.encode(doc_text).tolist()
        
        semantic_score = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
        
        # 2. Keyword matching
        keyword_score = 0
        matched_keywords = []
        
        if keywords:
            doc_content = f"{payload.get('author', '').lower()} {payload.get('topic', '').lower()} {payload.get('quote', '').lower()}"
            
            for keyword in keywords:
                if keyword in doc_content:
                    keyword_score += 1
                    matched_keywords.append(keyword)
                    
                    # Extra weight for exact matches
                    if keyword in payload.get('author', '').lower():
                        keyword_score += 2
                    if keyword in payload.get('topic', '').lower():
                        keyword_score += 1
        
        # 3. Analysis data boost
        analysis_boost = 0
        has_analysis = bool(payload.get('interpretation') or payload.get('historical_significance'))
        
        if has_analysis:
            # Boost more for interpretation/historical queries
            if intent['interpretation'] or intent['historical']:
                analysis_boost = self.analysis_boost * 1.5
            else:
                analysis_boost = self.analysis_boost
        
        # 4. Intent alignment boost
        intent_boost = 0
        if intent['interpretation'] and payload.get('interpretation'):
            intent_boost += 0.2
        if intent['historical'] and payload.get('historical_significance'):
            intent_boost += 0.2
        if intent['comparison']:
            # For comparison, prefer quotes with clear themes
            if payload.get('themes'):
                intent_boost += 0.1
        
        # Combine scores with weights
        final_score = (
            semantic_score * 0.4 +  # Semantic similarity
            (keyword_score * 0.3) +  # Keyword matching
            analysis_boost +         # Analysis data boost
            intent_boost             # Intent alignment
        )
        
        return {
            "final_score": final_score,
            "semantic_score": float(semantic_score),
            "keyword_score": keyword_score,
            "analysis_boost": analysis_boost,
            "intent_boost": intent_boost,
            "matched_keywords": matched_keywords,
            "has_analysis": has_analysis
        }
    
    def _select_best_documents(self, scored_docs, top_k, intent):
        """
        Select best documents considering query intent
        """
        # Sort by final score
        scored_docs.sort(key=lambda x: x["score_details"]["final_score"], reverse=True)
        
        # For interpretation/historical queries, prioritize documents with analysis
        if intent['interpretation'] or intent['historical']:
            # Separate documents with and without analysis
            docs_with_analysis = [d for d in scored_docs if d["score_details"]["has_analysis"]]
            docs_without_analysis = [d for d in scored_docs if not d["score_details"]["has_analysis"]]
            
            # Take more from analysis docs if available
            if docs_with_analysis:
                selected = docs_with_analysis[:min(top_k, len(docs_with_analysis))]
                remaining = top_k - len(selected)
                if remaining > 0 and docs_without_analysis:
                    selected.extend(docs_without_analysis[:remaining])
                return selected[:top_k]
        
        # Default: just take top_k
        return scored_docs[:top_k]
    
    def process_query(
        self, 
        question: str, 
        top_k: int = 3,
        analysis_mode: str = "auto"
    ):
        """
        Process query with analysis-aware retrieval
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            analysis_mode: "auto" (detect from query), "basic", or "comprehensive"
        """
        try:
            print(f"🔍 ANALYSIS-AWARE ENHANCED search: '{question}'")
            
            # Extract keywords and detect intent
            analysis_result = self._extract_keywords_and_intent(question)
            keywords = analysis_result["keywords"]
            intent = analysis_result["intent"]
            
            if keywords:
                print(f"   Keywords: {', '.join(keywords[:6])}")
            print(f"   Detected intent: {intent['type']}")
            
            # Determine analysis mode
            if analysis_mode == "auto":
                use_comprehensive = intent['interpretation'] or intent['historical']
            else:
                use_comprehensive = analysis_mode == "comprehensive"
            
            # Get query vector
            query_vector = self.embedder.encode(question).tolist()
            
            # Get ALL documents for scoring
            all_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=150,  # Get more documents for better selection
                with_payload=True
            )
            
            # Score each document
            scored_docs = []
            for point in all_points:
                payload = point.payload or {}
                doc_id = payload.get("id")
                
                if not doc_id:
                    continue
                
                # Calculate comprehensive score
                score_details = self._calculate_document_score(
                    payload, query_vector, keywords, intent, question
                )
                
                # Build document with score details
                doc = self._build_search_payload(payload, score_details["final_score"], include_analysis=True)
                doc.update({
                    "score_details": score_details,
                    "query_intent": intent['type']
                })
                
                scored_docs.append(doc)
            
            # Select best documents based on intent
            selected_docs = self._select_best_documents(scored_docs, top_k, intent)
            
            # Debug info
            print(f"✅ Found {len(selected_docs)} quotes (analysis-aware selection)")
            if selected_docs:
                top_doc = selected_docs[0]
                print(f"   Top document: ID {top_doc['id']}, Author: {top_doc['author']}")
                print(f"   Final score: {top_doc['score']:.3f}")
                if top_doc.get('has_analysis'):
                    print(f"   Has analysis data: YES")
                if top_doc['score_details'].get('matched_keywords'):
                    print(f"   Matched keywords: {', '.join(top_doc['score_details']['matched_keywords'][:3])}")
            
            # Generate answer with appropriate prompt
            if use_comprehensive:
                prompt = llm_client.format_rag_prompt_with_analysis(question, selected_docs)
                answer_type = "comprehensive_analysis"
            else:
                prompt = llm_client.format_rag_prompt(question, selected_docs)
                answer_type = "standard_answer"
            
            answer = llm_client.generate_response(
                prompt, 
                max_tokens=400 if use_comprehensive else 200
            )
            
            # Analyze if answer contains interpretation elements
            answer_lower = answer.lower()
            contains_interpretation = any(word in answer_lower for word in 
                                         ["means", "meaning", "signifies", "interpret"])
            contains_context = any(word in answer_lower for word in 
                                  ["historical", "context", "era", "during"])
            
            return {
                "answer": answer,
                "retrieved_count": len(selected_docs),
                "search_results": selected_docs,
                "method": "analysis_aware_retrieval",
                "query_analysis": analysis_result,
                "answer_type": answer_type,
                "analysis_mode": "comprehensive" if use_comprehensive else "standard",
                "contains_interpretation": contains_interpretation,
                "contains_historical_context": contains_context,
                "intent_matched": intent['type']
            }
            
        except Exception as e:
            error_text = f"Error in analysis-aware RAG: {e}"
            print(f"❌ {error_text}")
            import traceback
            traceback.print_exc()
            return {
                "answer": error_text,
                "retrieved_count": 0,
                "search_results": [],
                "method": "analysis_aware_retrieval",
                "error": True
            }
    
    def analyze_and_explain(self, quote_id: int) -> Dict[str, Any]:
        """
        Special method for comprehensive quote analysis
        Returns interpretation, context, and significance
        """
        try:
            # Get the specific quote
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
            
            points, _ = scroll_result
            if not points:
                return {"error": f"Quote with ID {quote_id} not found"}
            
            payload = points[0].payload or {}
            
            # Build comprehensive analysis prompt
            analysis_prompt = f"""Provide comprehensive analysis of this historical quote:

QUOTE: "{payload.get('quote', '')}"
AUTHOR: {payload.get('author', 'Unknown')}
ERA: {payload.get('era', 'Unknown')}
TOPIC: {payload.get('topic', 'Unknown')}
CONTEXT: {payload.get('context', 'Not specified')}
SOURCE: {payload.get('source', 'Not specified')}

Please provide detailed analysis covering:

1. LITERAL MEANING:
   - What does the quote literally say?
   - Key phrases and their significance

2. HISTORICAL CONTEXT:
   - When and why was this said?
   - What historical events/situation prompted it?
   - How did people react at the time?

3. INTERPRETATION & SIGNIFICANCE:
   - Deeper meaning beyond literal words
   - Why this quote became famous
   - Its impact on history/thought

4. MODERN RELEVANCE:
   - How is it relevant today?
   - Applications in modern context
   - Lessons for contemporary issues

5. THEMATIC ANALYSIS:
   - Key themes and ideas
   - Philosophical/political implications
   - Connection to broader human experience

Comprehensive Analysis:"""
            
            analysis = llm_client.generate_response(analysis_prompt, max_tokens=500)
            
            # Check if we have existing analysis in dataset
            existing_interpretation = payload.get('interpretation', '')
            existing_significance = payload.get('historical_significance', '')
            
            return {
                "success": True,
                "quote_id": quote_id,
                "quote": payload.get('quote', ''),
                "author": payload.get('author', ''),
                "comprehensive_analysis": analysis,
                "existing_analysis": {
                    "interpretation": existing_interpretation,
                    "historical_significance": existing_significance,
                    "themes": payload.get('themes', ''),
                    "modern_relevance": payload.get('modern_relevance', '')
                },
                "metadata": {
                    "era": payload.get('era', ''),
                    "context": payload.get('context', ''),
                    "source": payload.get('source', ''),
                    "topic": payload.get('topic', '')
                },
                "analysis_source": "existing" if existing_interpretation else "generated"
            }
            
        except Exception as e:
            return {"error": str(e), "quote_id": quote_id}
    
    def compare_quotes_for_analysis(self, quote_ids: List[int]) -> Dict[str, Any]:
        """
        Compare multiple quotes for thematic and historical analysis
        """
        try:
            quotes_data = []
            for qid in quote_ids[:3]:  # Limit to 3 for practical analysis
                result = self.analyze_and_explain(qid)
                if 'error' not in result:
                    quotes_data.append({
                        "id": qid,
                        "quote": result["quote"],
                        "author": result["author"],
                        "analysis": result["comprehensive_analysis"][:300] + "..."  # Preview
                    })
            
            if len(quotes_data) < 2:
                return {"error": "Need at least 2 quotes for comparison"}
            
            # Build comparison prompt
            comparison_text = ""
            for i, qd in enumerate(quotes_data, 1):
                comparison_text += f"""QUOTE {i}:
"{qd['quote']}"
Author: {qd['author']}
Analysis: {qd['analysis']}

"""
            
            comparison_prompt = f"""Compare and contrast these historical quotes:

{comparison_text}
Please analyze:

1. COMMON THEMES:
   - What themes do they share?
   - Similar philosophical approaches?

2. HISTORICAL CONTRASTS:
   - Different historical contexts
   - Different challenges addressed

3. COMPLEMENTARY PERSPECTIVES:
   - How do they complement each other?
   - What unique insights does each provide?

4. COLLECTIVE SIGNIFICANCE:
   - What do they collectively say about human experience?
   - How do they represent different aspects of wisdom?

Comparative Analysis:"""
            
            comparison = llm_client.generate_response(comparison_prompt, max_tokens=600)
            
            return {
                "success": True,
                "compared_quotes": [f"{qd['author']}: '{qd['quote'][:50]}...'" for qd in quotes_data],
                "comparative_analysis": comparison,
                "individual_analyses": quotes_data
            }
            
        except Exception as e:
            return {"error": str(e)}


# Create instance
enhanced_rag = EnhancedRAG(analysis_boost=0.3)


# Test functionality
if __name__ == "__main__":
    print("🧪 Testing Analysis-Aware Enhanced RAG")
    print("="*60)
    
    pipeline = EnhancedRAG()
    
    # Test different query types
    test_queries = [
        ("What does Roosevelt's 'fear itself' quote mean?", "interpretation"),
        ("What was the historical context of MLK's dream speech?", "historical"),
        ("What quotes are there about perseverance?", "retrieval"),
        ("Compare quotes about leadership and courage", "comparison")
    ]
    
    for query, expected_intent in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Expected intent: {expected_intent}")
        
        # Analyze query
        analysis = pipeline._extract_keywords_and_intent(query)
        print(f"Detected intent: {analysis['intent']['type']}")
        print(f"Keywords: {', '.join(analysis['keywords'][:5])}")
        
        # Process query
        result = pipeline.process_query(query, top_k=2)
        
        print(f"\nRetrieved {result.get('retrieved_count', 0)} documents")
        print(f"Answer type: {result.get('answer_type', 'unknown')}")
        print(f"Contains interpretation: {result.get('contains_interpretation', False)}")
        print(f"Contains historical context: {result.get('contains_historical_context', False)}")
        
        if result.get('search_results'):
            for i, doc in enumerate(result['search_results'][:2]):
                print(f"  {i+1}. ID: {doc.get('id')}, Author: {doc.get('author')}")
                print(f"     Score: {doc.get('score'):.3f}")
                if doc.get('has_analysis'):
                    print(f"     Has analysis data: YES")
    
    # Test single quote analysis
    print(f"\n{'='*60}")
    print("Testing single quote comprehensive analysis...")
    analysis_result = pipeline.analyze_and_explain(1)
    if 'comprehensive_analysis' in analysis_result:
        print(f"Analysis preview: {analysis_result['comprehensive_analysis'][:150]}...")
    
    print("\n✅ Analysis-Aware Enhanced RAG test complete!")