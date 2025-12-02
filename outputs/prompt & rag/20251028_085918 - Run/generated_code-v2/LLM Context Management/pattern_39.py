import streamlit as st
import numpy as np
import faiss
import os
import pickle
import shutil
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import pipeline

MODEL_NAME_EMBEDDING = "all-MiniLM-L6-v2"
MODEL_NAME_QA = "distilbert-base-uncased-distilled-squad"
INDEX_DIR = "faiss_indexes"
TOP_K_RETRIEVAL = 5

os.makedirs(INDEX_DIR, exist_ok=True)

def simulate_news_scraper():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        f"Breaking News Update {current_time}: Global markets react to new economic policies announced by major world powers. Inflation concerns are at an all-time high.",
        f"Sports News {current_time}: Local football team wins championship in a thrilling overtime match, celebrating with thousands of fans at the stadium.",
        f"Technology Review {current_time}: New AI ethics guidelines proposed to regulate advanced language models and ensure responsible development. Developers are discussing the implications.",
        f"Health Alert {current_time}: Researchers discover a new variant of a common virus, urging increased vigilance and public health measures. Vaccination drives are being intensified.",
        f"Political Insight {current_time}: Discussions around international climate agreements intensify ahead of the COP28 summit. Leaders are working on carbon emission targets.",
        f"Entertainment Gossip {current_time}: Famous actor announces retirement from the film industry, shocking fans worldwide. Tributes pour in from colleagues.",
        f"Science Discovery {current_time}: Breakthrough in fusion energy brings clean power closer to reality. Scientists hope for commercial reactors within decades.",
        f"Education Reform {current_time}: New curriculum changes introduced in schools to focus on digital literacy and critical thinking skills for students of all ages.",
        f"Financial Market {current_time}: Stock prices fluctuate amid persistent inflation concerns and anticipated interest rate hikes by central banks.",
        f"Cultural Event {current_time}: Annual film festival kicks off with a grand opening ceremony and celebrity appearances, showcasing international cinema.",
        f"Urban Development {current_time}: City council approves new infrastructure project to ease traffic congestion and improve public transportation networks. Construction begins next month.",
        f"Environmental News {current_time}: Conservation efforts lead to recovery of endangered species population in national parks. Ecologists report positive trends."
    ]

def preprocess_text(articles):
    return articles

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME_EMBEDDING)

def generate_embeddings(texts, model):
    if not texts:
        return np.array([])
    return model.encode(texts, convert_to_tensor=False)

def build_faiss_index(embeddings, texts):
    if embeddings.shape[0] == 0:
        return None, None, None

    d = embeddings.shape[1]
    
    nlist = min(1000, max(2, embeddings.shape[0] // 2))

    if embeddings.shape[0] < nlist and embeddings.shape[0] > 0:
        st.warning(f"Dataset size ({embeddings.shape[0]}) is too small for effective IVF training with nlist={nlist}. Using IndexFlatL2 as fallback.")
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
    elif embeddings.shape[0] == 0:
        return None, None, None # Should be caught by initial check
    else:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(embeddings)
        index.add(embeddings)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    index_filename = os.path.join(INDEX_DIR, f"news_index_{timestamp}.faiss")
    docs_filename = os.path.join(INDEX_DIR, f"news_docs_{timestamp}.pkl")

    faiss.write_index(index, index_filename)
    with open(docs_filename, "wb") as f:
        pickle.dump(texts, f)

    return index_filename, docs_filename, timestamp

def load_index_and_docs(index_path, docs_path):
    if not index_path or not docs_path:
        return None, None
    try:
        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
        return index, docs
    except Exception as e:
        st.error(f"Error loading index or documents: {e}")
        return None, None

def hotswap_active_index(new_index_path, new_docs_path, new_timestamp):
    st.session_state.active_index_path = new_index_path
    st.session_state.active_docs_path = new_docs_path
    st.session_state.last_updated_time = new_timestamp
    st.session_state.active_index, st.session_state.active_docs = load_index_and_docs(new_index_path, new_docs_path)

@st.cache_resource
def get_qa_pipeline():
    return pipeline("question-answering", model=MODEL_NAME_QA, tokenizer=MODEL_NAME_QA)

def retrieve_context(query_embedding, active_index, active_docs, k):
    if active_index is None or active_docs is None:
        return []
    
    query_embedding_2d = np.array([query_embedding]).astype('float32')

    D, I = active_index.search(query_embedding_2d, k)
    
    retrieved_docs = []
    for idx in I[0]:
        if idx != -1 and idx < len(active_docs):
            retrieved_docs.append(active_docs[idx])
    return retrieved_docs

def generate_answer(query, context, qa_pipeline):
    if not context:
        return "I'm sorry, I couldn't find relevant information in my knowledge base."
    
    combined_context = " ".join(context)
    
    try:
        result = qa_pipeline(question=query, context=combined_context)
        return result['answer']
    except Exception as e:
        st.warning(f"QA pipeline error, trying to provide context directly. Error: {e}")
        return f"Based on the most relevant information: {combined_context}"

st.set_page_config(layout="wide", page_title="Real-time News QA System")

st.title("📰 Real-time News QA System with Index Hotswapping")

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = get_embedding_model()
if "qa_pipeline" not in st.session_state:
    st.session_state.qa_pipeline = get_qa_pipeline()
if "active_index_path" not in st.session_state:
    st.session_state.active_index_path = None
if "active_docs_path" not in st.session_state:
    st.session_state.active_docs_path = None
if "last_updated_time" not in st.session_state:
    st.session_state.last_updated_time = "Never"
if "active_index" not in st.session_state:
    st.session_state.active_index = None
if "active_docs" not in st.session_state:
    st.session_state.active_docs = None

with st.sidebar:
    st.header("System Controls")
    st.info(f"Last Index Update: {st.session_state.last_updated_time}")

    if st.button("Generate & Hotswap New News Index"):
        with st.spinner("Generating new news articles and building index..."):
            new_articles = simulate_news_scraper()
            processed_articles = preprocess_text(new_articles)
            new_embeddings = generate_embeddings(processed_articles, st.session_state.embedding_model)
            
            if new_embeddings.shape[0] == 0:
                st.warning("No new articles generated to build an index. Please try again.")
            else:
                new_index_path, new_docs_path, timestamp = build_faiss_index(new_embeddings.astype('float32'), processed_articles)
                if new_index_path and new_docs_path and timestamp:
                    hotswap_active_index(new_index_path, new_docs_path, timestamp)
                    st.success(f"Index hotswapped successfully to version: {timestamp}")
                    st.rerun()

st.subheader("Ask a question about current events:")
user_query = st.text_input("Your question:", placeholder="What are the latest developments in AI ethics?")

if user_query:
    if st.session_state.active_index is None or st.session_state.active_docs is None:
        st.warning("No active knowledge base. Please generate an index first by clicking the button in the sidebar.")
    else:
        with st.spinner("Searching for answers..."):
            query_embedding = generate_embeddings([user_query], st.session_state.embedding_model)[0]
            retrieved_context = retrieve_context(
                query_embedding,
                st.session_state.active_index,
                st.session_state.active_docs,
                TOP_K_RETRIEVAL
            )

            if retrieved_context:
                st.subheader("Retrieved Context:")
                for i, doc in enumerate(retrieved_context):
                    st.write(f"**{i+1}.** {doc}")
                
                st.subheader("Generated Answer:")
                answer = generate_answer(user_query, retrieved_context, st.session_state.qa_pipeline)
                st.write(answer)
            else:
                st.info("No relevant information found in the current news index.")

st.markdown("---")
st.caption("This demo illustrates 'Index Hotswapping' for dynamic knowledge updates in a QA system.")
