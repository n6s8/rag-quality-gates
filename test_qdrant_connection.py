"""
Test Qdrant connection and basic operations
"""
import requests
from qdrant_client import QdrantClient

print("=" * 60)
print("🧪 TESTING QDRANT CONNECTION")
print("=" * 60)

print("1. Testing HTTP connection...")
try:
    response = requests.get("http://localhost:6333/health", timeout=5)
    print(f"   ✅ HTTP Status: {response.status_code}")
    print(f"   ✅ Response: {response.text}")
except Exception as e:
    print(f"   ❌ HTTP Failed: {e}")

print("\n2. Testing Qdrant client...")
try:
    client = QdrantClient(host="localhost", port=6333)
    
    collections = client.get_collections()
    print(f"   ✅ Connected to Qdrant")
    print(f"   ✅ Existing collections: {len(collections.collections)}")
    
    for collection in collections.collections:
        print(f"      - {collection.name}")
        
except Exception as e:
    print(f"   ❌ Qdrant client failed: {e}")

print("\n" + "=" * 60)
print("If both tests pass, Qdrant is working correctly!")
print("=" * 60)
