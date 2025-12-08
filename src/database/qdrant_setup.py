"""
Qdrant database setup and connection management
"""
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models

class QdrantDatabase:
    def __init__(self, host="localhost", port=6333):
        self.host = host
        self.port = port
        self.client = None
        self.collection_name = "historical_quotes"
        
    def connect(self):
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            print(f"✅ Connected to Qdrant at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            return False
    
    def create_collection(self, vector_size=384):
        if not self.client:
            print("❌ Not connected to database")
            return False
        
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                print(f"✅ Collection '{self.collection_name}' already exists")
                return True
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create collection: {e}")
            return False
    
    def get_collection_info(self):
        if not self.client:
            print("❌ Not connected to database")
            return None
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info
        except Exception as e:
            print(f"❌ Failed to get collection info: {e}")
            return None
    
    def disconnect(self):
        if self.client:
            print("Disconnected from Qdrant")
            self.client = None

qdrant_db = QdrantDatabase()

if __name__ == "__main__":
    db = QdrantDatabase()
    if db.connect():
        db.create_collection()
        info = db.get_collection_info()
        if info:
            print(f"Collection info: {info}")
