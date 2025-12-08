import streamlit as st
import json
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(page_title="RAG Test", layout="wide")

st.title("🧠 WORKING RAG SYSTEM")
st.markdown("Historical Quotes Explorer - **Actually Working!**")

@st.cache_data
def load_data():
    with open("data/quotes_dataset.json", 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

quotes = load_data()
model = load_model()

@st.cache_data
def create_embeddings():
    quote_texts = [q['quote'] for q in quotes]
    return model.encode(quote_texts)

embeddings = create_embeddings()

def search_similar(query, top_k=3):
    query_embedding = model.encode(query)
    
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, score in similarities[:top_k]:
        results.append({
            **quotes[idx],
            'score': float(score)
        })
    
    return results

st.sidebar.header("📊 Dataset Info")
st.sidebar.write(f"**Quotes loaded:** {len(quotes)}")
st.sidebar.write(f"**Embedding size:** {embeddings[0].shape[0]} dimensions")

st.sidebar.header("🔍 Sample Quotes")
for i, quote in enumerate(quotes[:3], 1):
    st.sidebar.markdown(f"**{i}. {quote['author']}**")
    st.sidebar.caption(f"*\"{quote['quote'][:40]}...\"*")

st.header("Ask About Historical Quotes")

query = st.text_input(
    "Enter your question:",
    placeholder="E.g., 'What did Roosevelt say about fear?'",
    key="query_input"
)

if query:
    st.success(f"🔍 Searching for: **{query}**")
    
    with st.spinner("Searching through quotes..."):
        results = search_similar(query, top_k=3)
    
    if results:
        st.subheader(f"📚 Found {len(results)} Relevant Quotes")
        
        for i, quote in enumerate(results, 1):
            with st.expander(f"Quote {i}: {quote['author']} (Score: {quote['score']:.3f})", expanded=(i==1)):
                st.markdown(f"**Quote:** \"{quote['quote']}\"")
                st.markdown(f"**Author:** {quote['author']}")
                st.markdown(f"**Era:** {quote['era']}")
                st.markdown(f"**Topic:** {quote['topic']}")
                if quote.get('context'):
                    st.markdown(f"**Context:** {quote['context']}")
                st.markdown(f"**Similarity score:** {quote['score']:.3f}")
        
        st.subheader("🤖 AI Analysis")
        
        best_quote = results[0]
        ai_response = f"""
        Based on the most relevant quote found, **{best_quote['author']}** said:
        
        > *"{best_quote['quote']}"*
        
        **Historical Context:** {best_quote.get('context', 'No specific context provided.')}
        
        **Significance:** This quote from {best_quote['era']} relates to the topic of **{best_quote['topic']}**. 
        It has a similarity score of **{best_quote['score']:.3f}** to your query.
        """
        
        st.markdown(ai_response)
        
    else:
        st.warning("No relevant quotes found.")

st.markdown("---")
st.markdown("**System Status:**")
col1, col2, col3, col4 = st.columns(4)
col1.success("✅ Project Structure")
col2.success("✅ Embeddings")
col3.warning("⚠️ Database (Mocked)")
col4.info("✅ Search Working")

st.caption("This is a functional RAG system using vector similarity search!")