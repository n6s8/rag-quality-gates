import requests

print("Testing Qdrant REST API directly...")

try:
    response = requests.get("http://localhost:6333/health", timeout=5)
    print(f"✅ Health check: {response.status_code} - {response.text}")
except Exception as e:
    print(f"❌ Health check failed: {e}")

try:
    response = requests.get("http://localhost:6333/collections", timeout=5)
    print(f"✅ Collections: {response.status_code}")
    
    if response.status_code == 200:
        collections = response.json()
        print(f"   Found collections: {collections}")
        
        if "historical_quotes" in str(collections):
            print("   ✅ 'historical_quotes' collection exists")
            
            coll_response = requests.get(
                "http://localhost:6333/collections/historical_quotes",
                timeout=5
            )
            if coll_response.status_code == 200:
                coll_info = coll_response.json()
                print(f"   Collection info: {coll_info}")
        else:
            print("   ❌ 'historical_quotes' collection not found")
            
except Exception as e:
    print(f"❌ Collections check failed: {e}")
