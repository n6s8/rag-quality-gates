import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing imports...")

# Test 1: Check if file exists
enhancement_file = project_root / "enhancements" / "simple_enhancement.py"
print(f"File exists: {enhancement_file.exists()}")

# Test 2: Try to read file
try:
    with open(enhancement_file, 'r', encoding='utf-8') as f:
        content = f.read(100)
        print(f"First 100 chars: {content}")
        if '\x00' in content:
            print("❌ File contains NULL bytes!")
        else:
            print("✅ File looks clean")
except Exception as e:
    print(f"❌ Cannot read file: {e}")

# Test 3: Try import
try:
    from enhancements.simple_enhancement import EnhancedRAG
    print("✅ Import successful!")
except ValueError as e:
    print(f"❌ ValueError: {e}")
    print("File is corrupted with wrong encoding")
except Exception as e:
    print(f"❌ Other error: {e}")