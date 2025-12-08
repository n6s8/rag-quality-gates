▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

      🛰️  REPOSITORY ORBIT: github.com/n6s8/Full-RAG-Stack
      📡 SIGNAL STRENGTH: ██████████ 100%
      
      🎬 TRANSMISSION FEED: youtu.be/Eawfe7b_0OE
      📶 BANDWIDTH: ██████████ 100%

▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

# 📜 Historical Quotes Explorer – RAG System

A complete **Retrieval-Augmented Generation (RAG)** system for exploring historical quotes with AI-powered context and explanations.  

The app lets you ask questions like:

- “What did Roosevelt say about fear?”
- “What did Martin Luther King Jr. dream about?”
- “Show me quotes about perseverance or leadership”

and returns:

- an **LLM answer** grounded in your custom dataset, and  
- the **exact quotes** and metadata used as context.

---

## 🎯 Main Idea (for assignment)

This project implements all required RAG components:

1. **Idea & Dataset**

   - Domain: *historical quotes* from famous figures (Roosevelt, MLK, Gandhi, Mandela, Armstrong, etc.).
   - Dataset files (source data, prepared manually / with LLM help):
     - `data/quotes_dataset.json` – quote text + author + era + topic + tags + context + source.
     - `data/historical_context.json` – extra biographical info per author (lifespan, occupation, short bio).
   - The dataset is small but **representative and well-annotated** and can be easily extended.

2. **Database with Vector Search**

   - Vector DB: **Qdrant**, running locally via Docker (`qdrant/qdrant:latest`).
   - Collection: `historical_quotes`.
   - Vector size: 384, distance: **cosine**.

3. **Embeddings Client**

   - Model: `all-MiniLM-L6-v2` from **SentenceTransformers**.
   - Used to:
     - embed each quote (`quote + author`) when loading data,
     - embed user queries at runtime.
   - Implemented in:
     - `src/database/data_loader.py` (ingestion),
     - `rag/rag_pipeline_rest.py` (search).

4. **Filling the Database**

   - Script: `src/database/data_loader.py`  
     - Loads JSON dataset.  
     - Creates embeddings.  
     - Upserts points with payload into Qdrant.

5. **LLM Client**

   - File: `rag/llm_client.py`.
   - Supports:
     - local HF chat model (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`),
     - optional DIAL/OpenAI via API key (if configured).
   - Handles simple **request–response** calls with a formatted RAG prompt.

6. **UI**

   - Technology: **Streamlit** (`frontend/app.py`).
   - Features:
     - text input for user question,
     - “Search” button,
     - sidebar with DB status and `top_k` slider,
     - main area with AI answer and retrieved quotes.

7. **RAG Pipeline (Joining Everything)**

   - File: `rag/rag_pipeline_rest.py`.
   - Steps:
     - take user input from UI,
     - convert to embedding,
     - search vectors in Qdrant,
     - build context from top-K quotes,
     - send question + context to LLM,
     - display answer + quotes back in the UI.

8. **Video (1–3 minutes)**
    Link:
   ```text
   https://youtu.be/Eawfe7b_0OE

## 🚀 Quick Start

### ✅ Prerequisites

- Python 3.8+
- Git
- Docker Desktop (or Docker Engine)
- Internet connection for downloading models

---

### 📦 1. Clone the Repository and Create Environment

```bash
# Clone repository
git clone <repository-url>
cd rag-historical-quotes

# Install dependencies
pip install -r requirements.txt

# Start Qdrant via Docker
docker-compose -f docker/docker-compose.yml up -d

# Check that Qdrant container is running
docker ps

# Check Qdrant health endpoint
python -c "import requests; print(requests.get('http://localhost:6333/health').text)"

# Load quotes into 'historical_quotes' collection
python src/database/data_loader.py

python test_search.py

streamlit run frontend/app.py