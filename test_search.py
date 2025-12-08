"""
Test searching in Qdrant
"""
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

test_queries = [
    "fear",
    "dream",
    "moon landing",
    "leadership",
    "change the world"
]

print("=" * 60)
print("🔍 TESTING SEARCH IN QDRANT")
print("=" * 60)

for query in test_queries:
    print(f"\nSearching for: '{query}'")
    
    query_embedding = model.encode(query).tolist()
    
    results = client.search(
        collection_name="historical_quotes",
        query_vector=query_embedding,
        limit=2
    )
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.score:.3f}")
            print(f"     \"{result.payload['quote'][:60]}...\"")
            print(f"     👤 {result.payload['author']}")
    else:
        print("  No results found")

print("\n" + "=" * 60)
print("✅ SEARCH TEST COMPLETE")
print("=" * 60)
