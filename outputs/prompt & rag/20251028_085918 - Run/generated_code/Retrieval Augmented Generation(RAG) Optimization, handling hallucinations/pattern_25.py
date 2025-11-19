import streamlit as st
import os
from dotenv import load_dotenv
from loguru import logger

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# --- Configuration and Initialization ---
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-rag-index")

logger.add("file.log", rotation="500 MB")

# Initialize Pinecone
try:
    pinecone = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384, # Corresponds to 'all-MiniLM-L6-v2'
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1') # Or choose your preferred cloud/region
        )
    index = pinecone.Index(PINECONE_INDEX_NAME)
    logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    st.error(f"Failed to initialize Pinecone: {e}")
    st.stop()

# Embedding Model for retrieval
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Initialize PineconeVectorStore
vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
logger.info("PineconeVectorStore initialized.")

# Dummy Data Ingestion (for demonstration)
def ingest_dummy_data(vecstore):
    if vecstore.similarity_search("medical", k=1):
        logger.info("Dummy data already present or similar data found, skipping ingestion.")
        return

    dummy_medical_docs = [
        "Diabetes mellitus is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored for energy. With diabetes, your body either doesn't make enough insulin or can't effectively use the insulin it does make. Common symptoms include increased thirst, frequent urination, and unexplained weight loss.",
        "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Risk factors include obesity, lack of exercise, and high salt intake. It's often called a silent killer because it usually has no symptoms.",
        "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from your colon on the lower right side of your abdomen. Appendicitis causes pain in your lower right abdomen. However, in most people, pain begins around the navel and then moves. As inflammation worsens, appendicitis pain typically increases and eventually becomes severe.",
        "Myocardial infarction, commonly known as a heart attack, occurs when blood flow to a part of your heart is blocked for a long enough time, that part of the heart muscle is damaged or dies. Symptoms can include chest pain, shortness of breath, pain in the left arm, and lightheadedness.",
        "Pneumonia is an infection that inflames air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses, and fungi, can cause pneumonia.",
        "Migraine is a headache of varying intensity, often accompanied by nausea and sensitivity to light and sound. Migraines are thought to be caused by abnormal brain activity, which can be triggered by stress, certain foods, or hormonal changes."
    ]
    logger.info(f"Ingesting {len(dummy_medical_docs)} dummy medical documents.")
    vecstore.add_texts(dummy_medical_docs)
    logger.info("Dummy data ingestion complete.")

ingest_dummy_data(vectorstore)

# LLM Setup
try:
    # Using a smaller model for demonstration on typical hardware
    model_name = "mistralai/Mistral-7B-Instruct-v0.2" # A good general-purpose instruct model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    # Using a pipeline for simpler text generation
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
    )
    logger.info(f"LLM '{model_name}' loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load LLM: {e}. Please ensure you have sufficient GPU memory.")
    st.error(f"Failed to load LLM: {e}. This application requires significant GPU resources. Try a smaller model or run locally with appropriate hardware.")
    st.stop()

class CustomLLM:
    def __call__(self, prompt: str) -> str:
        result = llm_pipeline(prompt)
        return result[0]['generated_text'].strip()

llm = CustomLLM()

# --- Langchain RAG Pipeline ---

# 1. Contextualize question (History-aware Retriever)
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm_pipeline, vectorstore.as_retriever(), contextualize_q_prompt)

# 2. Answer question based on context (Document Combination)
qa_system_prompt = """You are a medical diagnostic assistant. Your goal is to provide accurate and relevant medical information \
based on the patient's symptoms, medical history, and the provided medical context. \
If the information is insufficient to provide a confident diagnosis or treatment suggestion, state that you need more information. \
Do not hallucinate. Be cautious and prioritize patient safety. \

Use the following retrieved context to answer the question:
{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm_pipeline, qa_prompt)

# 3. Create the RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)


# --- Adaptive Retrieval & Self-Reflection Logic (Simplified for a single file) ---
# In a real-world scenario, this would involve more complex agents/tools
# and explicit confidence scoring/re-querying based on LLM output analysis.

def adaptive_rag_process(query: str, chat_history: list) -> str:
    # Initial RAG call
    response = rag_chain.invoke({"input": query, "chat_history": chat_history})
    generated_answer = response["answer"]
    
    # Simple Self-Reflection: Check if the LLM explicitly states it needs more information
    # This is a simplified heuristic. A robust system would use structured output parsing or specific prompts.
    if "insufficient information" in generated_answer.lower() or "need more information" in generated_answer.lower():
        logger.warning("LLM indicated insufficient information. Attempting to refine retrieval (simplified).")
        # In a real adaptive RAG, here you'd analyze the LLM's feedback
        # to formulate a new, more targeted query or change retrieval strategy.
        # For this example, we'll just log and return the current answer, 
        # implying the system *could* loop here with a refined query.
        return generated_answer + "\n\n*Self-reflection: The system noted a lack of sufficient information for a definitive answer. Further data or a refined query may be needed.*"
    
    return generated_answer

# --- Streamlit UI ---
st.set_page_config(page_title="Medical Diagnostic Assistant", layout="wide")
st.title("🩺 Medical Diagnostic Assistant with Adaptive RAG")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input
user_query = st.chat_input("Describe patient symptoms and medical history...")

if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.spinner("Analyzing and retrieving medical insights..."):
        try:
            response = adaptive_rag_process(user_query, st.session_state.chat_history)
            ai_message = AIMessage(content=response)
            st.session_state.chat_history.append(ai_message)
            st.chat_message("assistant").markdown(response)
        except Exception as e:
            logger.error(f"Error during RAG process: {e}")
            st.error(f"An error occurred: {e}. Please try again or check logs.")

st.sidebar.header("Information")
st.sidebar.info("This assistant helps healthcare professionals with diagnostic insights using an Adaptive RAG approach. It retrieves relevant medical information and attempts to self-reflect on its knowledge sufficiency.")

st.sidebar.header("Disclaimer")
st.sidebar.warning("This tool is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")

