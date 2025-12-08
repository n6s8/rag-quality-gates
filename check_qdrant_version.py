"""
Check Qdrant client version and API compatibility
"""
import pkg_resources

try:
    version = pkg_resources.get_distribution("qdrant-client").version
    print(f"Qdrant Client Version: {version}")
    
    if version.startswith("1."):
        print("API: Using newer API (query_points)")
    else:
        print("API: Using older API (search)")
        
except:
    print("Could not determine qdrant-client version")

from qdrant_client import QdrantClient

try:
    client = QdrantClient(host="localhost", port=6333)
    
    print("\nAvailable methods in QdrantClient:")
    methods = [method for method in dir(client) if not method.startswith('_')]
    for method in sorted(methods):
        print(f"  - {method}")
        
except Exception as e:
    print(f"Error: {e}")