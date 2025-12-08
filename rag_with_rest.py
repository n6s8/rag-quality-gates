"""
RAG Pipeline using REST API for Qdrant
"""
import requests
import json
from sentence_transformers import SentenceTransformer
from llm.llm_client import llm_client

class RAGRESTPipeline:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = "historical_quotes"
        self.base_url = "http://localhost:6333"
    
    def search_quotes(self, query, top_k=3):
        """
        Search for quotes using REST API
        """
        query_embedding = self.embedding_model.encode(query).tolist()
        
        search_payload = {
            "vector": query_embedding,
            "limit": top_k,
            "with_payload": True,
            "with_vector": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/collections/{self.collection_name}/points/search",
                json=search_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json()
                
                if results.get("result"):
                    points = results["result"]
                    
                    formatted_results = []
                    for point in points:
                        payload = point.get("payload", {})
                        formatted_results.append({
                            "quote": payload.get("quote", ""),
                            "author": payload.get("author", ""),
                            "era": payload.get("era", ""),
                            "topic": payload.get("topic", ""),
                            "context": payload.get("context", ""),
                            "source": payload.get("source", ""),
                            "score": point.get("score", 0)
                        })
                    
                    return formatted_results
            
        except Exception as e:
            print(f"Search error: {e}")
        
        return []
    
    def generate_answer(self, question, context_docs):
        """
        Generate answer using LLM
        """
        if not context_docs:
            return "I couldn't find any relevant quotes to answer your question."
        
        prompt = llm_client.format_rag_prompt(question, context_docs)
        
        response = llm_client.generate_response(prompt, max_tokens=300)
        
        return response
    
    def process_query(self, query, top_k=3):
        """
        Complete RAG pipeline
        """
        print(f"🔍 Processing: '{query}'")
        
        search_results = self.search_quotes(query, top_k=top_k)
        print(f"✅ Found {len(search_results)} relevant quotes")
        
        answer = self.generate_answer(query, search_results)
        
        return {
            "query": query,
            "search_results": search_results,
            "answer": answer,
            "retrieved_count": len(search_results)
        }
    
    def get_database_stats(self):
        """
        Get database statistics
        """
        try:
            response = requests.get(
                f"{self.base_url}/collections/{self.collection_name}",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("result", {})
                return {
                    "collection_name": self.collection_name,
                    "points_count": result.get("points_count", 0),
                    "status": "connected"
                }
            
        except Exception as e:
            return {"error": str(e)}
        
        return {"error": "Could not get stats"}

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 TESTING RAG PIPELINE WITH REST API")
    print("=" * 70)
    
    pipeline = RAGRESTPipeline()
    
    stats = pipeline.get_database_stats()
    print(f"Database stats: {stats}")
    
    print("\n🔍 Testing search...")
    results = pipeline.search_quotes("fear", top_k=2)
    print(f"Found {len(results)} quotes for 'fear':")
    
    for i, quote in enumerate(results, 1):
        print(f"\n{i}. {quote['author']}: \"{quote['quote'][:50]}...\"")
        print(f"   Score: {quote['score']:.3f}")
    
    print("\n" + "=" * 60)
    print("Testing complete RAG query...")
    
    full_result = pipeline.process_query(
        "What did Roosevelt say about fear?",
        top_k=2
    )
    
    print(f"\nQuery: {full_result['query']}")
    print(f"Retrieved: {full_result['retrieved_count']} quotes")
    print(f"\n🤖 AI Answer:")
    print("-" * 50)
    print(full_result['answer'])
    print("-" * 50)
    
    print("\n" + "=" * 70)
    print("🎉 RAG PIPELINE WITH REST API WORKS!")
    print("=" * 70)