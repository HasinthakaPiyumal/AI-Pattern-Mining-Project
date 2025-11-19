
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# --- Configuration --- #
KB_FILE = "medical_knowledge_base.csv"  # A CSV with 'title' and 'text' columns
LM_MODEL_NAME = "allenai/led-base-16384"  # Example: Longformer Encoder-Decoder for longer context
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 5

# --- Load Models --- #
@st.cache_resource
def load_lm_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(LM_MODEL_NAME)
    model = pipeline("text2text-generation", model=LM_MODEL_NAME, tokenizer=tokenizer)
    return model, tokenizer

@st.cache_resource
def load_embedding_model():
    return AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_embedding_tokenizer():
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

lm_pipeline, lm_tokenizer = load_lm_pipeline()
embedding_model = load_embedding_model()
embedding_tokenizer = load_embedding_tokenizer()

# --- Knowledge Base Operations --- #
@st.cache_data
def load_knowledge_base(file_path):
    try:
        kb_df = pd.read_csv(file_path)
        return kb_df
    except FileNotFoundError:
        st.error(f"Knowledge base file not found at {file_path}. Please create a '{KB_FILE}' with 'title' and 'text' columns.")
        return pd.DataFrame(columns=['title', 'text'])

@st.cache_data
def get_document_embeddings(texts):
    inputs = embedding_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with np.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

kp_df = load_knowledge_base(KB_FILE)
if not kp_df.empty:
    kb_embeddings = get_document_embeddings(kp_df['text'].tolist())
else:
    kb_embeddings = np.array([])

def retrieve_documents(query, kb_df, kb_embeddings, top_k):
    if kb_df.empty or kb_embeddings.size == 0:
        return [], []

    query_embedding = get_document_embeddings([query])
    similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    retrieved_docs = kb_df.iloc[top_indices].to_dict('records')
    retrieved_scores = similarities[top_indices].tolist()
    return retrieved_docs, retrieved_scores

# --- Reranking (Placeholder - can be enhanced) --- #
def rerank_documents(query, documents):
    # For simplicity, this example just passes through, but a more advanced reranker
    # could re-score based on LM or a fine-tuned model.
    return documents

# --- Conditional Retrieval (Simple Heuristic) --- #
def should_retrieve_knowledge(query):
    # A simple heuristic: trigger retrieval if the query contains medical terms
    # or asks for factual information. Can be improved with an actual classifier.
    medical_keywords = ["drug", "disease", "symptom", "treatment", "diagnosis", "condition", "medication", "therapy"]
    if any(keyword in query.lower() for keyword in medical_keywords) or "what is" in query.lower() or "explain" in query.lower():
        return True
    return False

# --- Streamlit UI --- #
st.set_page_config(layout="wide", page_title="Medical Information Assistant")
st.title("🩺 Medical Information Assistant for Healthcare Professionals")
st.markdown("This assistant provides grounded medical information by dynamically retrieving relevant documents.")

user_query = st.text_area("Enter your medical query here:", height=100)

if st.button("Get Information"):
    if user_query:
        with st.spinner("Processing query..."):
            if should_retrieve_knowledge(user_query):
                st.info("Retrieving relevant medical knowledge...")
                retrieved_docs, _ = retrieve_documents(user_query, kp_df, kb_embeddings, TOP_K_RETRIEVAL)
                reranked_docs = rerank_documents(user_query, retrieved_docs)

                if reranked_docs:
                    context = "\n\n".join([f"Document Title: {doc['title']}\nDocument Text: {doc['text']}" for doc in reranked_docs])
                    prompt = f"Based on the following medical documents, answer the query accurately and cite your sources. If the documents do not contain enough information, state that.\n\nDocuments:\n{context}\n\nQuery: {user_query}\n\nAnswer:"
                    st.subheader("Retrieved Documents (Context for LM):")
                    for i, doc in enumerate(reranked_docs):
                        st.expander(f"Document {i+1}: {doc['title']}").write(doc['text'])

                else:
                    prompt = f"Answer the following medical query to the best of your knowledge. If you cannot provide a factual answer, state that.\n\nQuery: {user_query}\n\nAnswer:"
                    st.warning("No relevant documents found. Answering based on LM's general knowledge (may be less accurate or attributed).")

            else:
                prompt = f"Answer the following medical query to the best of your knowledge. If you cannot provide a factual answer, state that.\n\nQuery: {user_query}\n\nAnswer:"
                st.info("Knowledge retrieval skipped (query does not seem to require external grounding).")

            # Generate response from LM
            lm_response = lm_pipeline(prompt, max_length=500, do_sample=False)[0]['generated_text']

            st.subheader("Assistant's Response:")
            st.write(lm_response)

            if should_retrieve_knowledge(user_query) and reranked_docs:
                st.subheader("Sources:")
                for doc in reranked_docs:
                    st.markdown(f"- {doc['title']}")

    else:
        st.warning("Please enter a query.")

# --- How to use --- #
st.sidebar.header("How to Use:")
st.sidebar.markdown("1. Ensure you have a `medical_knowledge_base.csv` file in the same directory.")
st.sidebar.markdown("   It should have two columns: `title` and `text`.")
st.sidebar.markdown("   Example `medical_knowledge_base.csv` content:")
st.sidebar.code("""title,text
'Diabetes Mellitus Type 2 Overview','Diabetes mellitus type 2 (formerly non-insulin-dependent diabetes mellitus) is a chronic metabolic disorder characterized by high blood sugar levels. It is caused by insulin resistance and relative insulin deficiency.'
'Hypertension Guidelines','Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.'
""")
st.sidebar.markdown("2. Run the Streamlit app: `streamlit run medical_assistant.py`")
st.sidebar.markdown("3. Enter your medical queries in the text area and click 'Get Information'.")
