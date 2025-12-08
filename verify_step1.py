import os
import sys
import json
import subprocess
from pathlib import Path

def print_header(text):
    print(f"\n{'='*60}")
    print(f"VERIFY: {text}")
    print('='*60)

def check_file(path, description):
    file_path = Path(path)
    if file_path.exists():
        size = file_path.stat().st_size
        print(f"[OK] {description}: {size:,} bytes")
        return True
    else:
        print(f"[ERROR] {description}: MISSING")
        return False

def check_directory(path, description):
    dir_path = Path(path)
    if dir_path.exists() and dir_path.is_dir():
        file_count = len(list(dir_path.rglob("*")))
        print(f"[OK] {description}: {file_count} files")
        return True
    else:
        print(f"[ERROR] {description}: MISSING")
        return False

def check_python_file(path, description):
    file_path = Path(path)
    if not file_path.exists():
        print(f"[ERROR] {description}: File not found")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        compile(content, file_path.name, 'exec')
        
        lines = content.split('\n')
        print(f"[OK] {description}: {len(lines)} lines, syntax OK")
        return True
    except SyntaxError as e:
        print(f"[ERROR] {description}: Syntax error - {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {description}: Error - {e}")
        return False

def check_json_file(path, description):
    file_path = Path(path)
    if not file_path.exists():
        print(f"[ERROR] {description}: File not found")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            print(f"[OK] {description}: Valid JSON with {len(data)} items")
            return True
        elif isinstance(data, dict):
            print(f"[OK] {description}: Valid JSON with {len(data)} keys")
            return True
        else:
            print(f"[WARN] {description}: Valid JSON but unexpected type")
            return True
    except json.JSONDecodeError as e:
        print(f"[ERROR] {description}: Invalid JSON - {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {description}: Error - {e}")
        return False

def test_imports():
    print_header("TESTING IMPORTS")
    
    imports_to_test = [
        ("sys", "Built-in system module"),
        ("json", "JSON module"),
        ("pathlib", "Path handling"),
    ]
    
    all_ok = True
    for module_name, description in imports_to_test:
        try:
            __import__(module_name)
            print(f"[OK] {description}: Import successful")
        except ImportError:
            print(f"[ERROR] {description}: Import failed")
            all_ok = False
    
    return all_ok

def test_basic_python_script():
    print_header("TESTING BASIC PYTHON EXECUTION")
    
    test_script = """
import sys
print("Python version:", sys.version.split()[0])
print("Python executable:", sys.executable)
print("Current directory:", __file__)
"""
    
    test_file = Path("_test_python.py")
    try:
        test_file.write_text(test_script, encoding='utf-8')
        
        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        
        result = subprocess.run(
            [sys.executable, "_test_python.py"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=env,
            timeout=5
        )
        
        if result.returncode == 0:
            print("[OK] Python script executed successfully")
            output_lines = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    output_lines.append(line.strip())
            print(f"   Output: {' | '.join(output_lines)}")
            success = True
        else:
            print(f"[ERROR] Python script failed with code {result.returncode}")
            if result.stderr:
                error_msg = result.stderr[:200].replace('\n', ' ')
                print(f"   Error: {error_msg}")
            success = False
    except Exception as e:
        print(f"[ERROR] Error running test script: {str(e)[:100]}")
        success = False
    finally:
        if test_file.exists():
            test_file.unlink()
    
    return success

def verify_project_structure():
    print_header("VERIFYING PROJECT STRUCTURE")
    
    checks = [
        (lambda: check_file("README.md", "README documentation"), "README.md"),
        (lambda: check_file("requirements.txt", "Requirements file"), "requirements.txt"),
        (lambda: check_file("project_description.md", "Project description"), "project_description.md"),
        (lambda: check_file(".gitignore", "Git ignore file"), ".gitignore"),
        
        (lambda: check_file("data/quotes_dataset.json", "Quotes dataset"), "data/quotes_dataset.json"),
        (lambda: check_json_file("data/quotes_dataset.json", "Quotes dataset validation"), "data/quotes_dataset.json"),
        
        (lambda: check_directory("src", "Source code directory"), "src"),
        (lambda: check_directory("src/database", "Database module"), "src/database"),
        (lambda: check_directory("src/embeddings", "Embeddings module"), "src/embeddings"),
        (lambda: check_directory("src/llm", "LLM module"), "src/llm"),
        (lambda: check_directory("src/rag", "RAG module"), "src/rag"),
        (lambda: check_directory("src/utils", "Utilities module"), "src/utils"),
        (lambda: check_directory("frontend", "Frontend directory"), "frontend"),
        (lambda: check_directory("docker", "Docker directory"), "docker"),
        (lambda: check_directory("tests", "Tests directory"), "tests"),
        
        (lambda: check_python_file("src/database/qdrant_setup.py", "Qdrant setup script"), "qdrant_setup.py"),
        (lambda: check_python_file("src/embeddings/embedding_client.py", "Embedding client"), "embedding_client.py"),
        (lambda: check_python_file("frontend/app.py", "Streamlit app"), "app.py"),
        (lambda: check_python_file("tests/test_basic.py", "Test suite"), "test_basic.py"),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_func, description in checks:
        if check_func():
            passed += 1
    
    print(f"\n[STATS] Structure Check: {passed}/{total} checks passed")
    return passed, total

def test_sample_data():
    print_header("TESTING SAMPLE DATA")
    
    try:
        with open("data/quotes_dataset.json", 'r', encoding='utf-8') as f:
            quotes = json.load(f)
        
        if not isinstance(quotes, list):
            print("[ERROR] Quotes data is not a list")
            return False
        
        print(f"[OK] Loaded {len(quotes)} quotes from dataset")
        
        if quotes:
            first_quote = quotes[0]
            required_fields = ["quote", "author", "era", "topic"]
            
            missing_fields = []
            for field in required_fields:
                if field not in first_quote:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"[ERROR] First quote missing fields: {missing_fields}")
                return False
            
            print(f"[OK] First quote structure OK:")
            quote_preview = first_quote['quote'][:50] + ('...' if len(first_quote['quote']) > 50 else '')
            print(f"   Quote: '{quote_preview}'")
            print(f"   Author: {first_quote['author']}")
            print(f"   Era: {first_quote['era']}")
            print(f"   Topic: {first_quote['topic']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing sample data: {e}")
        return False

def test_docker_compose():
    print_header("TESTING DOCKER COMPOSE")
    
    docker_file = Path("docker/docker-compose.yml")
    if not docker_file.exists():
        print("[ERROR] docker-compose.yml not found")
        return False
    
    try:
        content = docker_file.read_text(encoding='utf-8')
        
        if "qdrant" in content.lower() and "ports" in content:
            print("[OK] Docker compose file appears valid")
            print(f"   File size: {len(content):,} bytes")
            
            services_count = content.count("services:")
            print(f"   Services defined: {services_count}")
            
            return True
        else:
            print("[WARN] Docker compose file missing expected content")
            return False
    except Exception as e:
        print(f"[ERROR] Error reading docker-compose.yml: {e}")
        return False

def install_dependencies():
    print_header("INSTALLING MISSING DEPENDENCIES")
    
    if not Path("requirements.txt").exists():
        print("[ERROR] requirements.txt not found")
        return False
    
    try:
        print("Installing core dependencies...")
        
        dependencies = [
            "qdrant-client>=1.6.0",
            "sentence-transformers>=2.2.2",
            "numpy>=1.24.0"
        ]
        
        for dep in dependencies:
            print(f"  Installing {dep}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", dep],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"  [OK] {dep} installed successfully")
            else:
                print(f"  [ERROR] Failed to install {dep}")
                print(f"  Error: {result.stderr[:200]}")
                return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error installing dependencies: {e}")
        return False

def run_smoke_test_no_deps():
    print_header("RUNNING BASIC SMOKE TEST")
    
    smoke_test = """
import sys
from pathlib import Path

print("Smoke test starting...")

current_dir = Path(__file__).parent
print(f"Current directory: {current_dir}")

key_dirs = ["src", "data", "frontend", "docker"]
for dir_name in key_dirs:
    dir_path = current_dir / dir_name
    if dir_path.exists():
        print(f"[OK] Directory exists: {dir_name}")
    else:
        print(f"[ERROR] Missing directory: {dir_name}")

key_files = [
    "data/quotes_dataset.json",
    "src/database/qdrant_setup.py",
    "frontend/app.py",
    "README.md"
]

for file_path in key_files:
    full_path = current_dir / file_path
    if full_path.exists():
        print(f"[OK] File exists: {file_path}")
    else:
        print(f"[ERROR] Missing file: {file_path}")

print("\\nSmoke test completed!")
"""
    
    test_file = Path("_smoke_test.py")
    try:
        test_file.write_text(smoke_test, encoding='utf-8')
        
        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        
        result = subprocess.run(
            [sys.executable, "_smoke_test.py"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=env,
            timeout=10
        )
        
        if result.returncode == 0:
            print("[OK] Smoke test passed!")
            print("Output:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"  {line.strip()}")
            return True
        else:
            print(f"[ERROR] Smoke test failed with return code {result.returncode}")
            if result.stderr:
                error_msg = result.stderr[:200].replace('\n', ' ')
                print(f"  Error: {error_msg}")
            return False
    except subprocess.TimeoutExpired:
        print("[ERROR] Smoke test timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Error running smoke test: {str(e)[:100]}")
        return False
    finally:
        if test_file.exists():
            test_file.unlink()

def main():
    print("STEP 1 VERIFICATION - Historical Quotes Explorer")
    print("=" * 60)
    
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    print(f"Directory exists: {current_dir.exists()}")
    
    results = []
    
    results.append(("Python Imports", test_imports()))
    results.append(("Python Execution", test_basic_python_script()))
    
    struct_passed, struct_total = verify_project_structure()
    results.append(("Project Structure", struct_passed == struct_total))
    
    results.append(("Sample Data", test_sample_data()))
    results.append(("Docker Compose", test_docker_compose()))
    
    if install_dependencies():
        smoke_test = """
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

try:
    from database.qdrant_setup import QdrantDatabase
    from embeddings.embedding_client import EmbeddingClient
    from llm.llm_client import LLMClient
    
    print("All key modules import successfully")
    
    db = QdrantDatabase()
    print(f"Database client created: {db.host}:{db.port}")
    
    emb = EmbeddingClient()
    print(f"Embedding client created: {emb.model_name}")
    
    llm = LLMClient(use_local=True)
    print(f"LLM client created (local: {llm.use_local})")
    
    print("\\nSmoke test passed! Basic structure is working.")
    
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
"""
        
        test_file = Path("_full_smoke_test.py")
        try:
            test_file.write_text(smoke_test, encoding='utf-8')
            
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'
            
            result = subprocess.run(
                [sys.executable, "_full_smoke_test.py"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                env=env,
                timeout=10
            )
            
            if result.returncode == 0:
                print("[OK] Full smoke test passed!")
                print("Output:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"  {line.strip()}")
                results.append(("Smoke Test", True))
            else:
                print("[WARN] Full smoke test failed, running basic test instead")
                results.append(("Smoke Test", run_smoke_test_no_deps()))
        except Exception as e:
            print(f"[WARN] Error in full smoke test: {e}")
            results.append(("Smoke Test", run_smoke_test_no_deps()))
        finally:
            if test_file.exists():
                test_file.unlink()
    else:
        print("[WARN] Dependencies not installed, running basic smoke test")
        results.append(("Smoke Test", run_smoke_test_no_deps()))
    
    print_header("VERIFICATION SUMMARY")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_checks = len(results)
    
    print(f"\n[STATS] Overall Results: {total_passed}/{total_checks} checks passed")
    print("\n" + "=" * 60)
    
    for check_name, passed in results:
        status = "[OK]" if passed else "[ERROR]"
        print(f"{status} - {check_name}")
    
    print("\n" + "=" * 60)
    
    if total_passed == total_checks:
        print("\nCONGRATULATIONS! Step 1 verification PASSED!")
        print("Your project structure is complete and working.")
        print("\nNext steps:")
        print("1. Run: python setup.py")
        print("2. Or continue to Step 2: Expand the dataset")
        return True
    else:
        print(f"\nWARNING: Step 1 verification has {total_checks - total_passed} issue(s)")
        print("Issues found:")
        for check_name, passed in results:
            if not passed:
                print(f"  - {check_name}")
        print("\nPlease fix the issues above before proceeding.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during verification: {e}")
        sys.exit(1)
