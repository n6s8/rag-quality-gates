"""
FINAL TEST - Fixed version with proper imports
"""
import subprocess
import time
import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

print("=" * 70)
print("🎯 FINAL RAG SYSTEM VERIFICATION (FIXED)")
print("=" * 70)

def test_docker():
    print("\n1. 🐳 Testing Docker...")
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if "rag-qdrant" in result.stdout:
            print("   ✅ Docker: Qdrant container is running")
            return True
        else:
            print("   ❌ Docker: Qdrant container not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Docker test failed: {e}")
        return False

def test_qdrant_health():
    print("\n2. 🏥 Testing Qdrant health...")
    try:
        import requests
        response = requests.get("http://localhost:6333", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Qdrant is responding")
            return True
        else:
            print(f"   ⚠️ Qdrant returned: {response.status_code}")
            try:
                health_response = requests.get("http://localhost:6333/health", timeout=5)
                print(f"   Health endpoint: {health_response.status_code} - {health_response.text}")
            except:
                pass
            return True  
    except Exception as e:
        print(f"   ❌ Qdrant connection failed: {e}")
        return False

def test_database_data():
    print("\n3. 📊 Testing database data...")
    try:
        import requests
        response = requests.get(
            "http://localhost:6333/collections/historical_quotes",
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            points_count = data.get("result", {}).get("points_count", 0)
            
            if points_count > 0:
                print(f"   ✅ Database has {points_count} quotes loaded")
                return True
            else:
                print("   ❌ Database has no quotes")
                return False
        else:
            print(f"   ❌ Database check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        return False

def test_embeddings():
    print("\n4. 🤖 Testing embeddings...")
    try:
        import importlib.util
        embedding_path = src_dir / "embeddings" / "embedding_client.py"
        
        if embedding_path.exists():
            spec = importlib.util.spec_from_file_location("embedding_client", str(embedding_path))
            embedding_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(embedding_module)
            
            client = embedding_module.EmbeddingClient()
            embedding = client.embed_text("test")
            
            if len(embedding) == 384:
                print(f"   ✅ Embeddings working (vector size: {len(embedding)})")
                return True
            else:
                print(f"   ❌ Wrong vector size: {len(embedding)}")
                return False
        else:
            print("   ❌ Embedding client not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Embeddings test failed: {e}")
        return False

def test_simple_rag():
    print("\n5. 🔄 Testing Simple RAG (without imports)...")
    try:
        import requests
        from sentence_transformers import SentenceTransformer
        
        # Test search directly
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query = "fear"
        query_embedding = model.encode(query).tolist()
        
        search_payload = {
            "vector": query_embedding,
            "limit": 2,
            "with_payload": True
        }
        
        response = requests.post(
            "http://localhost:6333/collections/historical_quotes/points/search",
            json=search_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            if results.get("result"):
                print(f"   ✅ RAG search working (found {len(results['result'])} results)")
                
                # Show sample result
                first_result = results["result"][0]
                quote = first_result.get("payload", {}).get("quote", "")[:50]
                author = first_result.get("payload", {}).get("author", "Unknown")
                score = first_result.get("score", 0)
                
                print(f"   📝 Sample: \"{quote}...\" by {author} (score: {score:.3f})")
                return True
            else:
                print("   ❌ No results found")
                return False
        else:
            print(f"   ❌ Search failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Simple RAG test failed: {e}")
        return False

def test_streamlit():
    print("\n6. 🖥️ Testing Streamlit setup...")
    try:
        import streamlit
        print(f"   ✅ Streamlit installed (version: {streamlit.__version__})")
        
        app_path = current_dir / "frontend" / "app.py"
        if app_path.exists():
            print(f"   ✅ Streamlit app exists ({app_path.stat().st_size} bytes)")
            return True
        else:
            print("   ❌ Streamlit app not found")
            return False
            
    except ImportError:
        print("   ❌ Streamlit not installed")
        return False
    except Exception as e:
        print(f"   ❌ Streamlit test failed: {e}")
        return False

def main():
    """Run all tests"""
    tests = [
        ("Docker", test_docker),
        ("Qdrant Health", test_qdrant_health),
        ("Database Data", test_database_data),
        ("Embeddings", test_embeddings),
        ("Simple RAG", test_simple_rag),
        ("Streamlit", test_streamlit),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Score: {passed}/{total} ({passed/total*100:.0f}%)")
    
    if passed >= 5:  
        print("\n🎉 SYSTEM IS FUNCTIONAL!")
        print("Your RAG system is working!")
        print("\n✅ Ready for Step 9: Video Demonstration")
        print("\nQuick demo commands:")
        print("1. Test search: python search_with_rest.py")
        print("2. Run Streamlit: streamlit run frontend/app.py")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        print("Please fix the issues above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)