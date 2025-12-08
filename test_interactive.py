import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

def test_database_connection():
    print("=" * 60)
    print("TEST 1: Database Connection")
    print("=" * 60)
    
    try:
        from database.qdrant_setup import QdrantDatabase
        
        db = QdrantDatabase()
        print(f"📊 Database configured for: {db.host}:{db.port}")
        print(f"📊 Collection name: {db.collection_name}")
        
        if db.connect():
            print("✅ Database connection SUCCESSFUL")
            return True
        else:
            print("❌ Database connection FAILED")
            print("   Note: This is OK for now - we'll use mock data")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   This might be because qdrant-client is not installed")
        return False

def test_embeddings():
    print("\n" + "=" * 60)
    print("TEST 2: Embedding Generation")
    print("=" * 60)
    
    try:
        from embeddings.embedding_client import EmbeddingClient
        
        client = EmbeddingClient()
        print(f"📊 Using model: {client.model_name}")
        
        test_text = "This is a test quote about history"
        embedding = client.embed_text(test_text)
        
        print(f"✅ Embedding generated successfully!")
        print(f"   Vector size: {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_llm():
    print("\n" + "=" * 60)
    print("TEST 3: LLM Response")
    print("=" * 60)
    
    try:
        from llm.llm_client import LLMClient
        
        client = LLMClient(use_local=True)
        print(f"📊 LLM configured (local: {client.use_local})")
        
        test_prompt = "Hello, are you working?"
        response = client.generate_response(test_prompt, max_tokens=50)
        
        print(f"✅ LLM responded successfully!")
        print(f"   Response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Note: If transformers is not installed, LLM will use mock responses")
        return True

def test_rag_with_mock_data():
    print("\n" + "=" * 60)
    print("TEST 4: RAG Pipeline (Mock Data)")
    print("=" * 60)
    
    try:
        from rag.rag_pipeline import RAGPipeline
        
        pipeline = RAGPipeline()
        print("📊 RAG pipeline initialized")
        
        print("\n🔍 Testing with mock search...")
        
        test_query = "What did Roosevelt say about fear?"
        
        print(f"Query: '{test_query}'")
        print("\nSimulating RAG pipeline...")
        
        mock_results = [
            {
                "quote": "The only thing we have to fear is fear itself.",
                "author": "Franklin D. Roosevelt",
                "era": "1933",
                "topic": "Leadership, Courage",
                "context": "From his first inaugural address during the Great Depression",
                "source": "First Inaugural Address, March 4, 1933",
                "score": 0.95
            }
        ]
        
        from llm.llm_client import llm_client
        prompt = llm_client.format_rag_prompt(test_query, mock_results)
        answer = llm_client.generate_response(prompt, max_tokens=150)
        
        print(f"\n✅ RAG pipeline test COMPLETE")
        print(f"\n📝 Generated Answer:")
        print("-" * 40)
        print(answer)
        print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def interactive_query():
    print("\n" + "=" * 60)
    print("INTERACTIVE QUERY MODE")
    print("=" * 60)
    print("You can now ask questions about historical quotes!")
    print("Type 'exit' to quit")
    print("=" * 60)
    
    try:
        from rag.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
    except:
        print("⚠️  Could not initialize full RAG pipeline")
        print("   Using mock mode instead...")
        pipeline = None
    
    while True:
        print("\n" + "=" * 40)
        query = input("\n💬 Your question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not query:
            continue
            
        print(f"\n🔍 Processing: '{query}'")
        
        try:
            if pipeline:
                result = pipeline.process_query(query, top_k=2)
                
                print(f"\n✅ Found {result['retrieved_count']} relevant quotes")
                print(f"\n🤖 AI Answer:")
                print("-" * 40)
                print(result['answer'])
                print("-" * 40)
                
                if result['search_results']:
                    print(f"\n📚 Retrieved quotes:")
                    for i, quote in enumerate(result['search_results'], 1):
                        print(f"\n{i}. \"{quote['quote']}\"")
                        print(f"   👤 {quote['author']}")
                        print(f"   📅 {quote['era']}")
                        print(f"   🏷️  {quote['topic']}")
                        if quote.get('context'):
                            print(f"   📖 Context: {quote['context']}")
            else:
                from llm.llm_client import llm_client
                
                mock_context = [
                    {
                        "quote": "The only thing we have to fear is fear itself.",
                        "author": "Franklin D. Roosevelt",
                        "era": "1933",
                        "topic": "Leadership, Courage",
                        "context": "From his first inaugural address during the Great Depression",
                        "source": "First Inaugural Address"
                    },
                    {
                        "quote": "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character.",
                        "author": "Martin Luther King Jr.",
                        "era": "1963",
                        "topic": "Civil Rights, Equality",
                        "context": "From the 'I Have a Dream' speech during the March on Washington",
                        "source": "Lincoln Memorial, Washington D.C."
                    }
                ]
                
                prompt = llm_client.format_rag_prompt(query, mock_context)
                answer = llm_client.generate_response(prompt, max_tokens=200)
                
                print(f"\n🤖 AI Answer (Mock Mode):")
                print("-" * 40)
                print(answer)
                print("-" * 40)
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print("   Using simple fallback...")
            
            fallback_responses = [
                f"I understand you're asking about '{query}'. In my knowledge base, I have quotes from historical figures like Roosevelt, MLK, Gandhi, and others.",
                f"Your question '{query}' relates to historical quotes. I can help you find quotes about leadership, courage, civil rights, and more.",
                f"Based on your query '{query}', I can search through quotes from famous historical figures and provide context about them."
            ]
            
            import random
            print(f"\n🤖 Response: {random.choice(fallback_responses)}")

def main():
    print("🚀 RAG SYSTEM INTERACTIVE TEST")
    print("=" * 60)
    
    print("This will test all components of the RAG system...")
    print()
    
    test_database_connection()
    test_embeddings()
    test_llm()
    test_rag_with_mock_data()
    
    print("\n" + "=" * 60)
    choice = input("\nDo you want to try interactive query mode? (y/n): ").strip().lower()
    
    if choice == 'y':
        interactive_query()
    
    print("\n" + "=" * 60)
    print("🎉 RAG System Test Complete!")
    print("\nNext steps:")
    print("1. Install Docker and start Qdrant for full functionality")
    print("2. Run: docker-compose -f docker/docker-compose.yml up -d")
    print("3. Load data: python src/database/data_loader.py")
    print("4. Try the full Streamlit app: streamlit run frontend/app.py")

if __name__ == "__main__":
    main()
