"""
Streamlit frontend for Historical Quotes Explorer
"""

import os
import sys
from pathlib import Path
import json

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

for p in {ROOT_DIR, SRC_DIR}:
    if str(p) not in sys.path:
        sys.path.append(str(p))

from rag.rag_pipeline_rest import rag_pipeline

st.set_page_config(
    page_title="Historical Quotes Explorer",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .quote-card {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .source-text {
        font-size: 0.9rem;
        color: #6B7280;
        font-style: italic;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #10B981;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #3B82F6;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "current_results" not in st.session_state:
        st.session_state.current_results = None
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""


def display_sidebar():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2237/2237746.png", width=100)

        st.markdown("### 📊 Database Status")
        stats = rag_pipeline.get_database_stats()

        if "error" in stats:
            st.error("Database not connected")
        else:
            st.success(f"Connected to: {stats.get('collection_name', 'N/A')}")
            st.info(f"Quotes in database: {stats.get('vectors_count', 0)}")

        st.markdown("---")

        st.markdown("### 🔍 Search Settings")

        answer_style = st.selectbox(
            "Answer style",
            [
                "Standard — balanced answer",
                "Quick — short direct answer",
                "In-depth — detailed analysis",
                "Compare — contrast multiple quotes",
            ],
            index=0,
            help="Choose how detailed you want the AI answer to be.",
        )

        style_to_mode = {
            "Standard — balanced answer": "standard",
            "Quick — short direct answer": "basic",
            "In-depth — detailed analysis": "comprehensive",
            "Compare — contrast multiple quotes": "comparative",
        }

        analysis_mode = style_to_mode[answer_style]

        top_k = st.slider(
            "Maximum number of quotes",
            1,
            10,
            3,
            help="Upper bound for how many supporting quotes can be shown.",
        )

        st.markdown("---")

        st.markdown("### 📖 About")
        st.markdown(
            """
        This is a RAG-based Historical Quotes Explorer.
        
        **Features:**
        - Search historical quotes by topic, author, or concept
        - Get AI-powered explanations
        - View source context and citations
        
        **Technology:**
        - Qdrant Vector Database
        - Sentence Transformers for embeddings
        - Local LLM (Gemma 2B) or DIAL API
        
        Built as a learning project for RAG systems.
        """
        )

        return top_k, analysis_mode


def display_main_content(top_k, analysis_mode):
    st.markdown(
        '<h1 class="main-header">📜 Historical Quotes Explorer</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Ask questions about famous quotes and get AI-powered answers with historical context</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Ask about historical quotes:",
            placeholder="E.g., 'What did Einstein say about imagination?' or 'Quotes about leadership'",
            value=st.session_state.get("query_input", ""),
        )

    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    st.session_state.query_input = query

    if search_button and query:
        with st.spinner("Searching through historical quotes..."):
            results = rag_pipeline.process_query(
                query,
                top_k=top_k,
                analysis_mode=analysis_mode,
            )
            st.session_state.current_results = results
            st.session_state.search_history.append(query)

    if st.session_state.current_results:
        results = st.session_state.current_results

        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.markdown(
                f"**Analysis mode:** `{results.get('analysis_mode', analysis_mode)}`"
            )
        with meta_cols[1]:
            st.markdown(
                f"**Answer type:** `{results.get('answer_type', 'standard_answer')}`"
            )
        with meta_cols[2]:
            used_top_k = results.get("used_top_k", top_k)
            st.markdown(f"**Quotes used:** `{used_top_k}`")

        st.markdown("### 🤖 AI Answer")
        with st.container():
            st.markdown(
                f'<div class="success-box">{results["answer"]}</div>',
                unsafe_allow_html=True,
            )

        if not results.get("search_results"):
            st.warning("No matching quotes were found for this query.")
            return

        st.markdown("---")

        st.markdown(f"### 📚 Found {results['retrieved_count']} relevant quotes")

        for i, quote in enumerate(results["search_results"]):
            with st.container():
                st.markdown('<div class="quote-card">', unsafe_allow_html=True)

                st.markdown(f'**"{quote["quote"]}"**')

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"**Author:** {quote['author']}")
                with col_b:
                    st.markdown(f"**Era:** {quote['era']}")
                with col_c:
                    st.markdown(f"**Topic:** {quote['topic']}")

                if quote.get("context"):
                    st.markdown(f"*Context:* {quote['context']}")

                if quote.get("tags"):
                    tags_str = ", ".join(str(t) for t in (quote.get("tags") or []))
                    st.markdown(f"*Tags:* {tags_str}")

                if quote.get("source"):
                    st.markdown(
                        f'<p class="source-text">Source: {quote["source"]}</p>',
                        unsafe_allow_html=True,
                    )

                st.markdown(f"*Relevance score: {quote['score']:.3f}*")

                has_extra_analysis = any(
                    quote.get(k)
                    for k in [
                        "interpretation",
                        "historical_significance",
                        "themes",
                        "modern_relevance",
                    ]
                )

                if has_extra_analysis:
                    with st.expander("View deeper analysis for this quote"):
                        if quote.get("interpretation"):
                            st.markdown("**Interpretation**")
                            st.write(quote["interpretation"])
                        if quote.get("historical_significance"):
                            st.markdown("**Historical significance**")
                            st.write(quote["historical_significance"])
                        if quote.get("themes"):
                            st.markdown("**Themes**")
                            st.write(quote["themes"])
                        if quote.get("modern_relevance"):
                            st.markdown("**Modern relevance**")
                            st.write(quote["modern_relevance"])

                st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📊 View raw response data (debug)"):
            st.json(results)

    if not st.session_state.current_results:
        st.markdown("---")
        st.markdown("### 💡 Try These Example Queries:")

        examples_cols = st.columns(3)

        example_queries = [
            "What did Roosevelt say about fear?",
            "Tell me about leadership quotes from historical figures",
            "Find quotes about science and discovery",
            "What did Martin Luther King Jr. dream about?",
            "Show me ancient philosophical quotes",
            "Quotes about perseverance and resilience",
        ]

        for i, example in enumerate(example_queries):
            col = examples_cols[i % 3]
            with col:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.query_input = example
                    st.session_state.current_results = None
                    # Streamlit 1.40+ uses st.rerun instead of experimental_rerun
                    try:
                        st.rerun()
                    except AttributeError:
                        # Fallback for very old Streamlit versions
                        if hasattr(st, "experimental_rerun"):
                            st.experimental_rerun()


def display_footer():
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Built with:**")
        st.markdown("- Streamlit")
        st.markdown("- Qdrant")
        st.markdown("- Sentence Transformers")

    with col2:
        st.markdown("**RAG Pipeline:**")
        st.markdown("1. Query → Embedding")
        st.markdown("2. Vector Search")
        st.markdown("3. Context + LLM")
        st.markdown("4. Answer + Sources")

    with col3:
        st.markdown("**Learning Project**")
        st.markdown("For GenAI RAG Course")
        st.markdown("Frontend Developer Edition")

    st.markdown("---")
    st.markdown(
        "<center>📜 Historical Quotes Explorer | RAG System Implementation</center>",
        unsafe_allow_html=True,
    )


def main():
    initialize_session_state()
    top_k, analysis_mode = display_sidebar()
    display_main_content(top_k, analysis_mode)
    display_footer()


if __name__ == "__main__":
    main()
