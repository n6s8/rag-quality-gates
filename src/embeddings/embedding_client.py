from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingClient:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.vector_size = 384
        
    def load_model(self):
        try:
            print(f"📥 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            test_embedding = self.model.encode("test")
            self.vector_size = len(test_embedding)
            print(f"✅ Model loaded. Vector size: {self.vector_size}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model {self.model_name}: {e}")
            return False
    
    def embed_text(self, text):
        if not self.model:
            if not self.load_model():
                raise Exception("Failed to load embedding model")
        
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Failed to embed text: {e}")
            return [0] * self.vector_size
    
    def embed_batch(self, texts):
        if not self.model:
            if not self.load_model():
                raise Exception("Failed to load embedding model")
        
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"❌ Failed to embed batch: {e}")
            return [[0] * self.vector_size] * len(texts)
    
    def get_vector_size(self):
        if not self.model:
            self.load_model()
        return self.vector_size

embedding_client = EmbeddingClient()

if __name__ == "__main__":
    client = EmbeddingClient()
    
    test_text = "This is a test quote for embedding"
    embedding = client.embed_text(test_text)
    
    print(f"Text: {test_text}")
    print(f"Embedding shape: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    test_texts = [
        "First test quote",
        "Second test quote with different content",
        "Third quote about history"
    ]
    embeddings = client.embed_batch(test_texts)
    
    print(f"\nBatch embeddings: {len(embeddings)} vectors")
    for i, emb in enumerate(embeddings):
        print(f"  Text {i+1}: {len(emb)} dimensions")
