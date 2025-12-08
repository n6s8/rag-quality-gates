import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from database.qdrant_setup import qdrant_db
from embeddings.embedding_client import EmbeddingClient

class DataLoader:
    def __init__(self):
        self.embedding_client = EmbeddingClient()
        self.db = qdrant_db
        
    def load_quotes_from_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                quotes = json.load(f)
            print(f"✅ Loaded {len(quotes)} quotes from {file_path}")
            return quotes
        except Exception as e:
            print(f"❌ Failed to load quotes from {file_path}: {e}")
            return []
    
    def prepare_quote_for_db(self, quote):
        text_for_embedding = f"{quote['quote']} by {quote['author']}"
        
        embedding = self.embedding_client.embed_text(text_for_embedding)
        
        payload = {
            "id": quote.get("id", 0),
            "quote": quote["quote"],
            "author": quote["author"],
            "era": quote.get("era", ""),
            "topic": quote.get("topic", ""),
            "context": quote.get("context", ""),
            "source": quote.get("source", ""),
            "tags": quote.get("tags", []),
            "language": quote.get("language", "English")
        }
        
        return {
            "id": quote.get("id", 0),
            "vector": embedding,
            "payload": payload
        }
    
    def insert_quotes_to_db(self, quotes):
        if not self.db.client:
            print("❌ Database not connected")
            return 0
        
        prepared_quotes = []
        
        for quote in quotes:
            try:
                prepared = self.prepare_quote_for_db(quote)
                prepared_quotes.append(prepared)
            except Exception as e:
                print(f"❌ Failed to prepare quote {quote.get('id', 'unknown')}: {e}")
        
        if not prepared_quotes:
            print("❌ No quotes prepared for insertion")
            return 0
        
        try:
            batch_size = 50
            inserted_count = 0
            
            for i in range(0, len(prepared_quotes), batch_size):
                batch = prepared_quotes[i:i + batch_size]
                
                points = []
                for item in batch:
                    point = {
                        "id": item["id"],
                        "vector": item["vector"],
                        "payload": item["payload"]
                    }
                    points.append(point)
                
                self.db.client.upsert(
                    collection_name=self.db.collection_name,
                    points=points
                )
                
                inserted_count += len(batch)
                print(f"✅ Inserted batch {i//batch_size + 1}: {len(batch)} quotes")
            
            print(f"✅ Total inserted: {inserted_count} quotes")
            return inserted_count
            
        except Exception as e:
            print(f"❌ Failed to insert quotes: {e}")
            return 0
    
    def get_data_directory(self):
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        data_dir = project_root / "data"
        return data_dir
    
    def load_all_data(self):
        data_dir = self.get_data_directory()
        
        if not self.db.connect():
            return False
        
        if not self.db.create_collection():
            return False
        
        quotes_file = data_dir / "quotes_dataset.json"
        if quotes_file.exists():
            quotes = self.load_quotes_from_file(quotes_file)
            if quotes:
                inserted = self.insert_quotes_to_db(quotes)
                return inserted > 0
        
        return False

if __name__ == "__main__":
    loader = DataLoader()
    success = loader.load_all_data()
    
    if success:
        print("✅ Data loading completed successfully!")
    else:
        print("❌ Data loading failed!")
