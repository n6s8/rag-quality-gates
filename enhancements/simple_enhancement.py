"""
Enhanced RAG with Aggressive Keyword Boosting
Uses keyword matching to significantly boost relevant document scores
"""
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from src.llm.llm_client import llm_client


class EnhancedRAG:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "historical_quotes",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        self.client = QdrantClient(host=self.host, port=self.port)
        self.embedder = SentenceTransformer(self.embedding_model_name)
        
        print("✅ Enhanced RAG initialized (Aggressive Keyword Boosting)")
    
    def _extract_keywords(self, query: str):
        """Comprehensive keyword extraction for maximum matching"""
        keywords = []
        query_lower = query.lower()
        
        # Author names and variations
        author_mappings = {
            "roosevelt": ["franklin", "fdr"],
            "martin luther king": ["mlk", "king jr", "martin luther"],
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
            "fear": ["afraid", "scared", "frightened"],
            "dream": ["dreamed", "dreamt", "aspiration"],
            "change": ["transform", "different", "alter"],
            "perseverance": ["persist", "never give up", "keep going", "resilience"],
            "leadership": ["leader", "govern", "government", "authority"],
            "imagination": ["imagine", "creative", "creativity", "innovation"],
            "science": ["scientific", "knowledge", "learn", "discovery"],
            "courage": ["brave", "bravery", "bold", "fearless"],
            "equality": ["equal", "rights", "civil rights", "justice"],
            "freedom": ["free", "liberty", "independence"],
            "success": ["achieve", "achievement", "accomplish", "victory"],
            "failure": ["fail", "mistake", "error", "defeat"],
            "hope": ["hopeful", "optimistic", "expectation"]
        }
        
        for topic, synonyms in topic_mappings.items():
            if topic in query_lower:
                keywords.append(topic)
                keywords.extend(synonyms)
            for synonym in synonyms:
                if synonym in query_lower:
                    keywords.append(topic)
        
        # Extract important nouns
        stop_words = {"what", "who", "when", "where", "why", "how", "said", 
                     "say", "about", "some", "the", "a", "an", "and", "or", 
                     "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        words = query_lower.split()
        for word in words:
            # Remove punctuation
            word = word.strip('.,!?;"\'()')
            if (len(word) > 3 and 
                word not in stop_words and 
                word not in keywords and
                not word.isdigit()):
                keywords.append(word)
        
        return list(set(keywords))
    
    def _build_search_payload(self, payload, score=0.0):
        return {
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
    
    def _calculate_keyword_match_score(self, payload, keywords):
        """Calculate how well document matches keywords"""
        if not keywords:
            return 0
        
        doc_author = payload.get('author', '').lower()
        doc_topic = payload.get('topic', '').lower()
        doc_tags = ' '.join(payload.get('tags', [])).lower()
        doc_quote = payload.get('quote', '').lower()
        
        doc_text = f"{doc_author} {doc_topic} {doc_tags} {doc_quote}"
        
        match_score = 0
        matched_keywords = []
        
        for keyword in keywords:
            if keyword in doc_text:
                match_score += 1
                matched_keywords.append(keyword)
                
                # Extra points for author/topic matches
                if keyword in doc_author:
                    match_score += 2  # Author match is very important
                if keyword in doc_topic:
                    match_score += 1  # Topic match is important
        
        return match_score, matched_keywords
    
    def process_query(self, question: str, top_k: int = 3):
        try:
            print(f"🔍 AGGRESSIVE ENHANCED search: '{question}'")
            
            # Extract keywords
            keywords = self._extract_keywords(question)
            if keywords:
                print(f"   Keywords: {', '.join(keywords[:8])}")
            
            # Get ALL documents to apply keyword filtering
            all_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True
            )
            
            # Score each document
            scored_docs = []
            for point in all_points:
                payload = point.payload or {}
                doc_id = payload.get("id")
                
                if not doc_id:
                    continue
                
                # Get semantic score
                vector = self.embedder.encode(question).tolist()
                point_vector = self.embedder.encode(
                    f"{payload.get('quote', '')} {payload.get('author', '')}"
                ).tolist()
                
                # Calculate cosine similarity manually
                import numpy as np
                v1 = np.array(vector)
                v2 = np.array(point_vector)
                semantic_score = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                
                # Calculate keyword match score
                keyword_score, matched_keywords = self._calculate_keyword_match_score(payload, keywords)
                
                # AGGRESSIVE BOOSTING: Keyword matches dominate
                if keyword_score > 0:
                    # If we have keyword matches, heavily boost this document
                    final_score = semantic_score + (keyword_score * 0.5)  # Aggressive boost
                    boost_note = f"(+{keyword_score * 0.5:.2f} keyword boost)"
                else:
                    final_score = semantic_score
                    boost_note = ""
                
                scored_docs.append({
                    **self._build_search_payload(payload, final_score),
                    "original_score": float(semantic_score),
                    "keyword_score": keyword_score,
                    "matched_keywords": matched_keywords,
                    "boost_note": boost_note
                })
            
            # Sort by final score (keyword-boosted documents first)
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            
            # Remove duplicates by ID
            unique_docs = []
            seen_ids = set()
            for doc in scored_docs:
                if doc["id"] not in seen_ids:
                    seen_ids.add(doc["id"])
                    unique_docs.append(doc)
            
            # Take top_k
            docs = unique_docs[:top_k]
            
            # Debug info
            print(f"✅ Found {len(docs)} quotes")
            if docs and keywords:
                print(f"   Top document:")
                print(f"     ID: {docs[0]['id']}, Author: {docs[0]['author']}")
                print(f"     Score: {docs[0]['score']:.3f} {docs[0].get('boost_note', '')}")
                if docs[0].get('matched_keywords'):
                    print(f"     Matched keywords: {', '.join(docs[0]['matched_keywords'][:3])}")
            
            # Generate answer
            prompt = llm_client.format_rag_prompt(question, docs)
            answer = llm_client.generate_response(prompt)
            
            return {
                "answer": answer,
                "retrieved_count": len(docs),
                "search_results": docs,
                "method": "aggressive_keyword_boost",
                "keywords": keywords
            }
        except Exception as e:
            error_text = f"Error: {e}"
            print(f"❌ {error_text}")
            import traceback
            traceback.print_exc()
            return {
                "answer": error_text,
                "retrieved_count": 0,
                "search_results": [],
                "method": "aggressive_keyword_boost"
            }


# Create instance
enhanced_rag = EnhancedRAG()


if __name__ == "__main__":
    print("🧪 Testing Aggressive Enhanced RAG...")
    pipeline = EnhancedRAG()
    
    test_queries = [
        "What did Roosevelt say about fear?",
        "What did Martin Luther King Jr. dream about?",
        "What quotes are there about perseverance?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        result = pipeline.process_query(query, top_k=3)
        
        print(f"\nRetrieved {result.get('retrieved_count', 0)} documents:")
        for i, doc in enumerate(result.get('search_results', [])[:3]):
            print(f"  {i+1}. ID: {doc.get('id')}, Author: {doc.get('author')}")
            print(f"     Score: {doc.get('score'):.3f} (orig: {doc.get('original_score', 0):.3f})")
            if doc.get('matched_keywords'):
                print(f"     Keyword matches: {doc.get('matched_keywords')}")
    
    print("\n✅ Aggressive Enhanced RAG test complete!")