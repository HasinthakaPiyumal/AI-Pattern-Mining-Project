import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
import chromadb

# --- Configuration --- 
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L-2"
LLM_MODEL_NAME = "distilgpt2" # A small model for demonstration. Replace with larger for better quality
CHROMA_COLLECTION_NAME = "medical_documents"

# --- 1. Knowledge Base & Data Ingestion --- 
@st.cache_resource
def initialize_knowledge_base():
    # Example medical documents
    medical_texts = [
        "Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood sugar.",
        "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. A blood pressure reading of 130/80 mmHg or higher is generally considered hypertension.",
        "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out, and shortness of breath. For some people, asthma is a minor nuisance. For others, it can be a major problem that interferes with daily activities and may lead to a life-threatening asthma attack.",
        "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold. Symptoms include runny nose, sore throat, cough, congestion, and sometimes body aches or headache.",
        "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain from conditions such as muscle aches, toothaches, common cold, and headaches. It is also used to reduce pain and swelling in conditions such as arthritis. Aspirin is also known as acetylsalicylic acid.",
        "Paracetamol, also known as acetaminophen, is a medication used to treat pain and fever. It is commonly used for headaches, muscle aches, arthritis, backache, toothaches, colds, and fevers. It works by blocking the production of certain chemicals in the brain called prostaglandins, which are involved in the sensation of pain and fever.",
        "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus. Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Symptoms can include fever, cough, fatigue, and loss of taste or smell.",
        "Vaccines work by training the immune system to recognize and combat pathogens, such as viruses or bacteria. They expose the body to a safe form of a pathogen, or antigens that resemble the pathogen, allowing the immune system to develop antibodies and memory cells without causing actual illness."
    ]

    documents = [Document(text=t) for t in medical_texts]

    # Initialize ChromaDB client and collection
    db = chromadb.Client()
    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    # Create LlamaIndex VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, embed_model=embed_model)
    return index

# --- 2. Language Model (LM) Integration --- 
@st.cache_resource
def initialize_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    llm = HuggingFaceLLM(
        model_name=LLM_MODEL_NAME,
        tokenizer_name=LLM_MODEL_NAME,
        query_wrapper_prompt="""<|startoftext|>{query_str}<|endoftext|>""",
        context_window=1024, # Adjust based on chosen LLM
        max_new_tokens=256,
        generate_kwargs={
            "temperature": 0.1,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id
        },
        device_map="auto"
    )
    Settings.llm = llm
    return llm

# --- 3. Query Engine Setup (Retriever + Reranker + LLM) --- 
@st.cache_resource
def setup_query_engine(index):
    # Initialize reranker
    reranker = SentenceTransformerRerank(
        model=RERANK_MODEL_NAME,
        top_n=3 # Number of documents to keep after reranking
    )
    
    # Configure retriever
    retriever = index.as_retriever(similarity_top_k=5) # Retrieve more initially before reranking
    
    # Create query engine with reranking
    query_engine = index.as_query_engine(
        retriever=retriever,
        node_postprocessors=[reranker],
        llm=Settings.llm
    )
    return query_engine

# --- 4. Streamlit UI --- 
st.title("Medical Information Assistant")
st.write("Ask any medical question and get grounded information!")

# Initialize components
index = initialize_knowledge_base()
llm = initialize_llm()
query_engine = setup_query_engine(index)

query = st.text_input("Your medical question:")

if st.button("Get Answer") and query:
    with st.spinner("Searching for information..."):
        # Conditional Retrieval Logic (for demo, always triggered)
        # In a real app, you might check query complexity or keywords here.
        if True: # Always retrieve for this demo
            response = query_engine.query(query)
            
            st.subheader("Answer:")
            st.write(response.response)
            
            st.subheader("Sources:")
            if response.source_nodes:
                for i, node in enumerate(response.source_nodes):
                    st.write(f"**Source {i+1}:**")
                    st.write(node.text[:200] + "...") # Display snippet
                    st.write(f"*Similarity Score (initial retrieval):* {node.score:.2f}") # Original similarity
            else:
                st.write("No specific sources found.")
        else:
            # Fallback for when retrieval is not triggered (e.g., if query is very simple)
            # This part would typically just pass the query directly to the LLM without context.
            st.write("Retrieval not triggered for this query. Responding without external knowledge.")
            # For this demo, we'll just show a placeholder.
            st.write("Sorry, I cannot provide an answer for this type of query without grounding.")

elif not query and st.button("Get Answer"):
    st.warning("Please enter a medical question.")