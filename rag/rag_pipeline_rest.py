import os
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

from llm.llm_client import llm_client


class RAGPipeline:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "historical_quotes",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        self.client = QdrantClient(host=self.host, port=self.port)
        self.embedder = SentenceTransformer(self.embedding_model_name)

    def get_database_stats(self) -> Dict[str, Any]:
        try:
            count_result = self.client.count(
                collection_name=self.collection_name,
                exact=True,
            )
            return {
                "collection_name": self.collection_name,
                "vectors_count": int(count_result.count),
            }
        except Exception as e:
            return {
                "error": str(e),
                "collection_name": self.collection_name,
                "vectors_count": 0,
            }

    def _build_search_payload(self, hit) -> Dict[str, Any]:
        payload = hit.payload or {}
        return {
            "id": payload.get("id"),
            "quote": payload.get("quote", ""),
            "author": payload.get("author", "Unknown"),
            "era": payload.get("era", ""),
            "topic": payload.get("topic", ""),
            "context": payload.get("context", ""),
            "source": payload.get("source", ""),
            "tags": payload.get("tags", []),
            "language": payload.get("language", "English"),
            "score": float(hit.score),
        }

    def process_query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        try:
            print(f"🔍 Processing query: '{question}'")
            print(f"📥 Loading embedding model: {self.embedding_model_name}")
            vector = self.embedder.encode(question).tolist()
            print(f"✅ Model loaded. Vector size: {len(vector)}")

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            docs = [self._build_search_payload(hit) for hit in search_result]
            print(f"✅ Found {len(docs)} relevant quotes")

            prompt = llm_client.format_rag_prompt(question, docs)
            answer = llm_client.generate_response(prompt)

            return {
                "answer": answer,
                "retrieved_count": len(docs),
                "search_results": docs,
            }
        except Exception as e:
            error_text = f"Error during RAG processing: {e}"
            print(f"❌ {error_text}")
            return {
                "answer": error_text,
                "retrieved_count": 0,
                "search_results": [],
            }


rag_pipeline = RAGPipeline()
