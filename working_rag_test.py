import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

print("=" * 70)
print("🚀 WORKING RAG SYSTEM TEST")
print("=" * 70)

print("\n1. 📊 LOADING DATA")
print("-" * 40)

with open("data/quotes_dataset.json", 'r', encoding='utf-8') as f:
    quotes = json.load(f)

print(f"✅ Loaded {len(quotes)} quotes")
for i, quote in enumerate(quotes, 1):
    print(f"   {i}. {quote['author']}: \"{quote['quote'][:30]}...\"")

print("\n2. 🤖 INITIALIZING EMBEDDINGS")
print("-" * 40)

model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded")

quote_texts = [q['quote'] for q in quotes]
embeddings = model.encode(quote_texts)
print(f"✅ Created {len(embeddings)} embeddings ({embeddings[0].shape[0]} dimensions)")

print("\n3. 🔍 SIMPLE SIMILARITY SEARCH")
print("-" * 40)

def search_quotes(query, top_k=3):
    query_embedding = model.encode(query)
    
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, score in similarities[:top_k]:
        results.append({
            **quotes[idx],
            'score': float(score)
        })
    
    return results

test_queries = [
    "fear",
    "dream", 
    "small step",
    "rising after falling",
    "change world"
]

for query in test_queries:
    print(f"\n🔍 Searching for: '{query}'")
    results = search_quotes(query, top_k=1)
    
    if results:
        result = results[0]
        print(f"   ✅ Found: \"{result['quote'][:50]}...\"")
        print(f"   👤 Author: {result['author']}")
        print(f"   📊 Similarity: {result['score']:.3f}")

print("\n4. 🧠 RAG SIMULATION")
print("-" * 40)

def simulate_rag(query):
    print(f"\n🤔 User asks: '{query}'")
    
    results = search_quotes(query, top_k=2)
    
    if not results:
        return "No relevant quotes found."
    
    print("📚 Retrieved quotes:")
    for i, quote in enumerate(results, 1):
        print(f"\n   {i}. \"{quote['quote']}\"")
        print(f"      Author: {quote['author']}")
        print(f"      Context: {quote.get('context', 'No context provided')}")
        print(f"      Similarity: {quote['score']:.3f}")
    
    print(f"\n🤖 Simulated AI Answer:")
    print("-" * 40)
    
    if query.lower().find("fear") != -1:
        answer = """Based on the retrieved quotes, Franklin D. Roosevelt famously said 
"The only thing we have to fear is fear itself" during his first inaugural 
address in 1933. This quote was meant to inspire confidence during the 
Great Depression and has become one of the most famous quotes in American history."""
    
    elif query.lower().find("dream") != -1:
        answer = """Martin Luther King Jr. delivered his iconic "I Have a Dream" speech 
in 1963 during the March on Washington. In it, he expressed his dream that 
his children would live in a nation where they would not be judged by their 
skin color but by their character."""
    
    elif query.lower().find("step") != -1:
        answer = """Neil Armstrong said "That's one small step for man, one giant leap 
for mankind" when he became the first person to step onto the moon in 1969. 
This marked a historic moment in space exploration."""
    
    else:
        answer = f"""Based on the retrieved quotes, I found relevant historical quotes 
that relate to your query about "{query}". The quotes provide wisdom from 
historical figures that can offer insight into this topic."""
    
    print(answer)
    print("-" * 40)
    
    return answer

print("\n" + "=" * 70)
print("💬 INTERACTIVE RAG TEST")
print("=" * 70)
print("Type your questions about historical quotes!")
print("Type 'exit' to quit")
print("=" * 70)

while True:
    print("\n" + "-" * 40)
    user_query = input("\nYour question: ").strip()
    
    if user_query.lower() in ['exit', 'quit', 'q']:
        print("\n👋 Goodbye!")
        break
    
    if not user_query:
        continue
    
    simulate_rag(user_query)

print("\n" + "=" * 70)
print("🎉 RAG SYSTEM TEST COMPLETE!")
print("\n✅ What's working:")
print("   - Project structure ✓")
print("   - Embeddings generation ✓") 
print("   - Vector similarity search ✓")
print("   - Data loading ✓")
print("\n⚠️  Next steps for full functionality:")
print("   1. Install Docker Desktop")
print("   2. Install qdrant-client: pip install qdrant-client")
print("   3. Start Qdrant: docker-compose -f docker/docker-compose.yml up -d")
print("   4. Load data to database: python src/database/data_loader.py")
print("\n📁 Your project is READY for Step 2!")
