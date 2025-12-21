from __future__ import annotations

import re
import numpy as np
from typing import List, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    print("⚠️  Install dependencies: pip install sentence-transformers scikit-learn")


_EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "so",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "i", "you", "we", "they", "he", "she", "them", "him", "her",
    "my", "your", "our", "their", "his", "hers", "ours", "theirs",
    "what", "who", "when", "where", "why", "how", "which",
    "do", "does", "did", "done", "can", "could", "would", "should", "may", "might",
    "not", "no", "yes", "very", "more", "most", "less", "least",
    "about", "into", "over", "under", "again", "also", "than"
}


def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize_words(s: str) -> List[str]:
    s = _clean_text(s).lower()
    words = re.findall(r"[a-zA-Z']+", s)
    return [w for w in words if len(w) >= 3 and w not in _EN_STOPWORDS]


def _extract_numbers(s: str) -> List[str]:
    s = _clean_text(s)
    return re.findall(r"\b\d{1,4}\b", s)


def _context_to_text(context_docs: List[Dict[str, Any]]) -> str:
    parts = []
    for d in context_docs or []:
        parts.append(str(d.get("quote", "")))
        parts.append(str(d.get("author", "")))
        parts.append(str(d.get("topic", "")))
        parts.append(str(d.get("era", "")))
        parts.append(str(d.get("context", "")))
        parts.append(str(d.get("source", "")))
        parts.append(str(d.get("interpretation", "")))
        parts.append(str(d.get("historical_significance", "")))
        parts.append(str(d.get("themes", "")))
        parts.append(str(d.get("modern_relevance", "")))
        tags = d.get("tags", [])
        if isinstance(tags, list):
            parts.append(" ".join([str(t) for t in tags]))
        else:
            parts.append(str(tags))
    return _clean_text(" ".join([p for p in parts if p]))


class RAGMetrics:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embedder = None

        if HAS_DEPENDENCIES:
            try:
                self.embedder = SentenceTransformer(embedding_model_name)
                print(f"✅ Metrics initialized with {embedding_model_name}")
            except Exception as e:
                print(f"⚠️  Could not load embedder: {e}")
                self.embedder = None
        else:
            print("⚠️  Running without embedding capabilities")

    def retrieval_precision(self, retrieved_ids: List[int], expected_ids: List[int]) -> float:
        if not retrieved_ids:
            return 0.0
        relevant_retrieved = set(retrieved_ids) & set(expected_ids)
        return float(len(relevant_retrieved) / len(retrieved_ids))

    def retrieval_recall(self, retrieved_ids: List[int], expected_ids: List[int]) -> float:
        if not expected_ids:
            return 0.0
        relevant_retrieved = set(retrieved_ids) & set(expected_ids)
        return float(len(relevant_retrieved) / len(expected_ids))

    def answer_relevance(self, generated_answer: str, expected_answer: str) -> float:
        """
        Semantic similarity between generated and expected answer (0..1).
        Uses cosine similarity of sentence embeddings and normalizes to [0, 1].
        """
        if not generated_answer or not expected_answer or not self.embedder:
            return 0.0

        try:
            ga = _clean_text(generated_answer)
            ea = _clean_text(expected_answer)

            emb = self.embedder.encode([ga, ea])
            sim = cosine_similarity([emb[0]], [emb[1]])[0][0]

            sim = float(sim)
            sim = max(-1.0, min(1.0, sim))
            sim01 = (sim + 1.0) / 2.0
            return float(max(0.0, min(1.0, sim01)))
        except Exception as e:
            print(f"⚠️  Relevance calculation failed: {e}")
            return 0.0

    def context_grounded_score(self, generated_answer: str, context_docs: List[Dict[str, Any]]) -> float:
        """
        How well the answer is supported by retrieved context (0..1).
        Embedding-based similarity answer <-> context_text.
        """
        if not generated_answer or not context_docs or not self.embedder:
            return 0.0
        try:
            context_text = _context_to_text(context_docs)
            if not context_text:
                return 0.0

            emb = self.embedder.encode([_clean_text(generated_answer), context_text])
            sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
            sim = float(sim)
            sim = max(-1.0, min(1.0, sim))
            sim01 = (sim + 1.0) / 2.0
            return float(max(0.0, min(1.0, sim01)))
        except Exception as e:
            print(f"⚠️  Groundedness calculation failed: {e}")
            return 0.0

    def hallucination_score(self, generated_answer: str, context_docs: List[Dict[str, Any]]) -> float:
        """
        Hallucination score (0..1), where 0 is best (no hallucination).
        Heuristic: penalize answers that are weakly grounded in context,
        with extra penalty for numbers (years/dates) not present in context.
        """
        if not generated_answer:
            return 0.0

        if not context_docs:
            return 0.35

        context_text = _context_to_text(context_docs)
        if not context_text:
            return 0.35

        grounded = self.context_grounded_score(generated_answer, context_docs)

        ans_nums = set(_extract_numbers(generated_answer))
        ctx_nums = set(_extract_numbers(context_text))
        num_penalty = 0.0
        if ans_nums:
            missing = [n for n in ans_nums if n not in ctx_nums]
            if missing:
                num_penalty = min(0.3, 0.1 * len(missing))

        ans_words = _tokenize_words(generated_answer)
        ctx_words = set(_tokenize_words(context_text))
        word_penalty = 0.0
        if ans_words:
            missing_words = [w for w in ans_words if w not in ctx_words]
            missing_ratio = len(missing_words) / max(1, len(ans_words))
            word_penalty = min(0.25, missing_ratio * 0.25)

        halluc = (1.0 - grounded) * 0.35 + num_penalty + word_penalty
        halluc = float(max(0.0, min(1.0, halluc)))

        if halluc < 0.05:
            halluc = 0.0

        return halluc

    def response_time(self, start_time: float, end_time: float) -> float:
        return float(end_time - start_time)


if __name__ == "__main__":
    print("🧪 Testing RAGMetrics...")
    metrics = RAGMetrics()
    print("✅ Metrics module loaded successfully")
