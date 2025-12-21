▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

      🛰️  REPOSITORY ORBIT: github.com/n6s8/rag-quality-gates
      📡 SIGNAL STRENGTH: ██████████ 100%

      🎬 TRANSMISSION FEED: youtu.be/HgSonhJaUoU
      📶 BANDWIDTH: ██████████ 100%

▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

# 📜 Historical Quotes Explorer — Advanced RAG + Evaluation & Enhancement

A complete **Retrieval-Augmented Generation (RAG)** system for exploring historical quotes with AI-powered context and explanations — plus an **automated evaluation pipeline** that measures RAG metrics, applies an enhancement, and generates a Markdown report.

You can ask:

- “What did Roosevelt say about fear?”
- “What did Martin Luther King Jr. dream about?”
- “Who said ‘Be the change you wish to see in the world’?”
- “Show me quotes about perseverance or leadership”

…and the system returns:

- an **LLM answer** grounded in the dataset
- the **exact retrieved quotes + metadata** used as context
- (for the Advanced task) **metrics + report** proving improvement after enhancement

---

## ✅ Assignment Alignment (Advanced RAG Practical Task)

This repository includes:

1) **Metrics definition for RAG**
- Valuable RAG metrics are implemented and measured automatically.
- We selected **Retrieval Precision** as the primary target metric (high business value: less irrelevant context → more trustworthy answers).

2) **Automated testing environment**
- Scripts measure metrics under evaluation queries and store machine-readable artifacts (`.json`).
- A report generator builds `docs/enhancement_report.md` from those artifacts.

3) **System enhancement**
- An enhancement was applied to improve the target metric.
- The report includes baseline vs enhanced comparison and trade-offs.

4) **Re-evaluation + appended reporting**
- Baseline and enhanced runs are repeated with the same evaluation setup.
- The report is updated with the new state.

✅ **Acceptance criterion met:** Retrieval Precision improved by **+128.6%** (>= +30%).

---

## 🧠 System Architecture (Core RAG)

### 1) Dataset
- Domain: historical quotes (Roosevelt, MLK, Gandhi, Mandela, etc.)
- Data files:
  - `data/quotes_dataset.json` — quote text + author + era + topic + tags + context + source
  - (optional) `data/historical_context.json` — author metadata / bios

### 2) Vector Database
- Vector DB: **Qdrant**
- Collection: `historical_quotes`
- Vector size: 384
- Distance: cosine

### 3) Embeddings
- SentenceTransformers: `all-MiniLM-L6-v2`
- Used for:
  - embedding quotes during ingestion
  - embedding user queries at runtime

### 4) Ingestion (Load data into Qdrant)
- Script: `src/database/data_loader.py`
  - reads JSON
  - creates embeddings
  - upserts into Qdrant

### 5) LLM Client
- Local HF model supported (example used in runs):
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (CPU)
- Generates answers from retrieved context.

### 6) UI (Optional)
- Streamlit interface: `frontend/app.py`
- Lets users ask questions and view retrieved quotes.

### 7) RAG Pipeline
- File: `rag/rag_pipeline_rest.py`
- Steps:
  - embed query
  - vector search in Qdrant
  - build prompt/context from retrieved docs
  - LLM generation
  - return answer + evidence

---

## 📏 Metrics (Advanced Task)

The evaluation pipeline tracks:

### Core retrieval metrics
- **Retrieval Precision** (TARGET) — how many retrieved docs are actually relevant
- Retrieval Recall — how many relevant docs were successfully retrieved

### Answer quality metrics (heuristics)
- Answer Relevance
- Hallucination Score
- Response Time

### Optional analysis/interpretation metrics (heuristics)
- Interpretation Score
- Historical Context Score
- Explanation Depth
- Thematic Analysis
- Interpretation Quality

> Note: interpretation metrics are heuristic and mainly useful for regression comparisons.

---

## 🚀 Quick Start (Run the App)

### ✅ Prerequisites
- Python 3.8+
- Git
- Docker Desktop / Docker Engine
- Internet (first run downloads models)

### 📦 1) Clone + Install
```bash
git clone <repository-url>
cd rag-historical-quotes
pip install -r requirements.txt
```

## 🚀 Quick Start (Run the App)

### 🧱 2) Start Qdrant
```bash
docker-compose -f docker/docker-compose.yml up -d
```
### 🩺 3) Check Qdrant Health
```bash
python -c "import requests; print(requests.get('http://localhost:6333/health').text)"
```
### 📥 4. Load Quotes into Qdrant
```bash
python src/database/data_loader.py
```
### 🔎 5. Optional: Quick Retrieval Test
```bash
python test_search.py
```
### ▶️ 6. Run UI (Streamlit)
```bash
streamlit run frontend/app.py
```
## 🧪 Advanced Task: Automated Evaluation + Report

### ✅ One-command full evaluation (baseline + enhanced + report)

```bash
python scripts/run_full_evaluation.py
```
### 📊 Current Results (from the latest run)

- **Target metric:** Retrieval Precision  
- **Baseline:** 0.389  
- **Enhanced:** 0.889  
- **Improvement:** **+128.6%** ✅ (>= +30%)
