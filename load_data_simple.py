"""
Simple data loader for Qdrant (Updated API)
"""
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("📚 LOADING DATA INTO QDRANT")
print("=" * 60)

print("1. Loading quotes...")
with open("data/quotes_dataset.json", "r", encoding="utf-8") as f:
    quotes = json.load(f)

print(f"   ✅ Loaded {len(quotes)} quotes")

print("2. Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
vector_size = 384
print(f"   ✅ Model loaded, vector size: {vector_size}")

print("3. Connecting to Qdrant...")
client = QdrantClient(host="localhost", port=6333)
collection_name = "historical_quotes"
print("   ✅ Connected to Qdrant")

print("4. Creating/checking collection...")
try:
    existing_collections = client.get_collections()
    collection_exists = False
    for collection in existing_collections.collections:
        if collection.name == collection_name:
            collection_exists = True
            break
    if collection_exists:
        print(f"   ⚠️ Collection '{collection_name}' already exists")
        print("   Do you want to (R)ecreate or (A)ppend? [R/A]: ", end="")
        choice = input().strip().upper()
        if choice == "R":
            client.delete_collection(collection_name=collection_name)
            print("   ✅ Deleted existing collection")
            recreate = True
        else:
            recreate = False
            print("   ✅ Using existing collection")
    else:
        recreate = True
except Exception as e:
    print(f"   Note: {e}")
    recreate = True

if recreate:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"   ✅ Created collection '{collection_name}'")

print("5. Preparing data...")
points = []
for i, quote in enumerate(quotes):
    text = f"{quote['quote']} by {quote['author']}"
    embedding = model.encode(text).tolist()
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
            "language": quote.get("language", "English"),
        },
    )
    points.append(point)
    print(f"   Processed {i + 1}/{len(quotes)} quotes")

print("\n6. Uploading to Qdrant...")
operation_info = client.upsert(collection_name=collection_name, points=points)
print(f"   ✅ Uploaded {len(points)} quotes")

print("\n7. Verifying upload...")
try:
    count_result = client.count(collection_name=collection_name, exact=True)
    print(f"   📊 Vectors in collection: {count_result.count}")
except Exception as e:
    print(f"   ⚠️ Could not verify collection size: {e}")

print("\n" + "=" * 60)
print("🎉 DATA LOADING COMPLETE!")
print("=" * 60)
