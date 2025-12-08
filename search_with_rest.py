"""
Search Qdrant using REST API (works with any client version)
"""
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("🔍 SEARCHING QDRANT USING REST API")
print("=" * 60)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Test query
test_queries = [
    "fear",
    "dream",
    "moon",
    "leadership",
    "change"
]

for query in test_queries:
    print(f"\nSearching for: '{query}'")
    
    query_embedding = model.encode(query).tolist()
    
    search_payload = {
        "vector": query_embedding,
        "limit": 3,
        "with_payload": True,
        "with_vector": False
    }
    
    try:
        response = requests.post(
            "http://localhost:6333/collections/historical_quotes/points/search",
            json=search_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            
            if results.get("result"):
                points = results["result"]
                print(f"  Found {len(points)} results:")
                
                for i, point in enumerate(points, 1):
                    score = point.get("score", 0)
                    payload = point.get("payload", {})
                    
                    quote = payload.get("quote", "No quote")
                    author = payload.get("author", "Unknown")
                    
                    print(f"  {i}. Score: {score:.3f}")
                    print(f"     \"{quote[:60]}...\"")
                    print(f"     👤 {author}")
            else:
                print("  No results found")
        else:
            print(f"  Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 60)
print("✅ REST API SEARCH COMPLETE")
print("=" * 60)