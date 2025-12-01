import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

class MedicalSearchEngine:
    def __init__(self, passage_model_name='sentence-transformers/all-MiniLM-L6-v2', question_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.passage_encoder = SentenceTransformer(passage_model_name)
        self.question_encoder = SentenceTransformer(question_model_name)
        self.corpus = []
        self.passage_embeddings = None
        self.index = None

    def ingest_and_index_corpus(self, medical_data_path=None, dummy_data=None):
        if medical_data_path:
            # In a real application, load from medical_data_path (e.g., CSV, database)
            # For this example, we'll use a dummy data if no path is provided
            pass
        
        if dummy_data is None:
            # Using a small, dummy medical literature corpus for demonstration
            dummy_data = [
                "The efficacy of remdesivir in treating severe COVID-19 has been demonstrated in several clinical trials.",
                "Aspirin is commonly prescribed for its antiplatelet effects, reducing the risk of cardiovascular events.",
                "Diabetes mellitus type 2 is characterized by insulin resistance and relative insulin deficiency.",
                "Immunotherapy has revolutionized cancer treatment, particularly for melanoma and lung cancer.",
                "Recent studies suggest a link between gut microbiota composition and various neurological disorders.",
                "Hypertension, or high blood pressure, is a major risk factor for heart disease and stroke.",
                "Gene editing techniques like CRISPR-Cas9 hold immense promise for curing genetic diseases.",
                "Vaccination remains the most effective strategy for preventing infectious diseases like influenza.",
                "Understanding pharmacokinetics and pharmacodynamics is crucial for safe drug administration.",
                "Neurodegenerative diseases, such as Alzheimer's and Parkinson's, present significant challenges in treatment development."
            ]
        self.corpus = dummy_data
        st.write(f"Loaded {len(self.corpus)} passages for indexing.")

        st.write("Generating passage embeddings...")
        self.passage_embeddings = self.passage_encoder.encode(self.corpus, show_progress_bar=True)
        st.write("Passage embeddings generated.")

        embedding_dim = self.passage_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)  # Using L2 distance for similarity
        self.index.add(np.array(self.passage_embeddings).astype('float32'))
        st.write(f"FAISS index created with {self.index.ntotal} passages.")

    def search(self, query, top_k=5):
        if self.index is None:
            st.error("Corpus not indexed. Please index the corpus first.")
            return [], []

        query_embedding = self.question_encoder.encode([query])[0]
        query_embedding = np.array([query_embedding]).astype('float32')

        distances, indices = self.index.search(query_embedding, top_k)

        retrieved_passages = [self.corpus[idx] for idx in indices[0]]
        # Optional: return similarity scores as well (1 - normalized_distance if using L2 for semantic sim)
        # For L2, lower distance means higher similarity. We can invert or just show distance.
        return retrieved_passages, distances[0]

# Streamlit UI
st.title("🔬 Medical Literature Semantic Search Engine")
st.markdown("Ask a natural language question and retrieve semantically relevant medical passages.")

# Initialize the search engine (and load/index corpus only once) with Streamlit caching
@st.cache_resource
def get_search_engine():
    engine = MedicalSearchEngine()
    engine.ingest_and_index_corpus() # Using dummy data for this demo
    return engine

search_engine = get_search_engine()

query = st.text_input("Enter your medical question:", "What are the treatments for type 2 diabetes?")

if st.button("Search"):
    if query:
        with st.spinner("Searching for relevant passages..."):
            retrieved_passages, distances = search_engine.search(query, top_k=5)
        
        st.subheader("Retrieved Passages:")
        if retrieved_passages:
            for i, (passage, dist) in enumerate(zip(retrieved_passages, distances)):
                st.write(f"**{i+1}.** {passage} (Distance: {dist:.4f})")
        else:
            st.info("No passages found for your query. Try a different question.")
    else:
        st.warning("Please enter a question to search.")

st.markdown("---")
st.info("Note: This is a demonstration using a small, pre-defined set of dummy medical passages and a generic sentence transformer model. For a real-world application, a much larger, domain-specific corpus and potentially fine-tuned DPR models would be used.")
