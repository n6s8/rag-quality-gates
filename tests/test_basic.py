import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from database.qdrant_setup import QdrantDatabase
from embeddings.embedding_client import EmbeddingClient
from llm.llm_client import LLMClient

def test_database_connection():
    print("🧪 Testing database connection...")
    db = QdrantDatabase(host="localhost", port=6333)
    connected = db.connect()
    
    if connected:
        print("✅ Database connection test PASSED")
    else:
        print("❌ Database connection test FAILED")
    
    return connected

def test_embedding_generation():
    print("🧪 Testing embedding generation...")
    client = EmbeddingClient()
    
    try:
        embedding = client.embed_text("Test quote")
        
        if len(embedding) == 384:
            print(f"✅ Embedding generation test PASSED (Vector size: {len(embedding)})")
            return True
        else:
            print(f"❌ Unexpected vector size: {len(embedding)}")
            return False
    except Exception as e:
        print(f"❌ Embedding generation test FAILED: {e}")
        return False

def test_llm_response():
    print("🧪 Testing LLM response generation...")
    client = LLMClient(use_local=True)
    
    try:
        response = client.generate_response("Hello, are you working?")
        
        if response and len(response) > 0:
            print(f"✅ LLM response test PASSED (Response length: {len(response)})")
            print(f"   Sample: {response[:50]}...")
            return True
        else:
            print("❌ Empty response from LLM")
            return False
    except Exception as e:
        print(f"❌ LLM response test FAILED: {e}")
        return False

def test_rag_pipeline():
    print("🧪 Testing RAG pipeline...")
    
    try:
        from rag.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        
        result = pipeline.process_query("test", top_k=1)
        
        if isinstance(result, dict) and "query" in result:
            print(f"✅ RAG pipeline test PASSED")
            print(f"   Query processed: {result['query']}")
            print(f"   Retrieved: {result['retrieved_count']} quotes")
            return True
        else:
            print("❌ Invalid result format from RAG pipeline")
            return False
    except Exception as e:
        print(f"❌ RAG pipeline test FAILED: {e}")
        return False

def run_all_tests():
    print("=" * 60)
    print("🧪 RUNNING RAG SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Embedding Generation", test_embedding_generation),
        ("LLM Response", test_llm_response),
        ("RAG Pipeline", test_rag_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\n📈 Score: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()
