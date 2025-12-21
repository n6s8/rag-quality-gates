from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

try:
    collections = client.get_collections()
    print(f"✅ Qdrant connected! Collections: {collections}")
    
    collection_info = client.get_collection("historical_quotes")
    print(f"📊 Collection info: {collection_info}")
except Exception as e:
    print(f"❌ Qdrant connection failed: {e}")