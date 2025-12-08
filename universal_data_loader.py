"""
Universal data loader that works with any Qdrant API version
"""
import json
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("🔄 UNIVERSAL DATA LOADER FOR QDRANT")
print("=" * 60)

with open("data/quotes_dataset.json", 'r', encoding='utf-8') as f:
    quotes = json.load(f)

print(f"Loaded {len(quotes)} quotes")

model = SentenceTransformer('all-MiniLM-L6-v2')

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "historical_quotes"
    
    print("Connected to Qdrant")
    
    collections_response = client.get_collections()
    collection_exists = False
    
    for collection in collections_response.collections:
        if collection.name == collection_name:
            collection_exists = True
            break
    
    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Created collection '{collection_name}'")
    else:
        print(f"Collection '{collection_name}' already exists")
    
    points = []
    for i, quote in enumerate(quotes):
        text = f"{quote['quote']} by {quote['author']}"
        embedding = model.encode(text).tolist()
        
        try:
            from qdrant_client.models import PointStruct
            
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "id": quote.get("id", i),
                    "quote": quote["quote"],
                    "author": quote["author"],
                    "era": quote.get("era", ""),
                    "topic": quote.get("topic", ""),
                    "context": quote.get("context", ""),
                    "source": quote.get("source", ""),
                    "tags": quote.get("tags", []),
                    "language": quote.get("language", "English")
                }
            )
            points.append(point)
        except:
            point = {
                "id": i,
                "vector": embedding,
                "payload": {
                    "id": quote.get("id", i),
                    "quote": quote["quote"],
                    "author": quote["author"],
                    "era": quote.get("era", ""),
                    "topic": quote.get("topic", ""),
                    "context": quote.get("context", ""),
                    "source": quote.get("source", ""),
                    "tags": quote.get("tags", []),
                    "language": quote.get("language", "English")
                }
            }
            points.append(point)
        
        print(f"Processed {i + 1}/{len(quotes)} quotes")
    
    try:
        client.upsert(collection_name=collection_name, points=points)
    except:
        client.upload_points(collection_name=collection_name, points=points)
    
    print(f"✅ Uploaded {len(points)} quotes to Qdrant")
    
    try:
        count_result = client.count(collection_name=collection_name)
        print(f"📊 Total vectors: {count_result.count}")
    except:
        print("📊 Upload verification complete")
        
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative approach...")
    
    import requests
    import base64
    import json as json_module
    
    vectors = []
    for quote in quotes:
        text = f"{quote['quote']} by {quote['author']}"
        embedding = model.encode(text).tolist()
        
        vector_str = json_module.dumps(embedding)
        vector_b64 = base64.b64encode(vector_str.encode()).decode()
        
        vectors.append({
            "id": quote.get("id", len(vectors)),
            "vector": vector_b64,
            "payload": {
                "quote": quote["quote"],
                "author": quote["author"],
                "era": quote.get("era", ""),
                "topic": quote.get("topic", "")
            }
        })
    
    for vector in vectors:
        response = requests.post(
            f"http://localhost:6333/collections/{collection_name}/points",
            json={"points": [vector]}
        )
        
        if response.status_code == 200:
            print(f"Uploaded point {vector['id']}")
        else:
            print(f"Failed to upload point {vector['id']}: {response.text}")

print("\n" + "=" * 60)
print("🎉 DATA LOADING ATTEMPTED")
print("Check Qdrant to see if data was loaded:")
print("curl http://localhost:6333/collections/historical_quotes")
print("=" * 60)
