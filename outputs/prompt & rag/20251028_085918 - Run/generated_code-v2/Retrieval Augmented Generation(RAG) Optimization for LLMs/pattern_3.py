import os
from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel
import requests

import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Environment Variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Knowledge Base Setup (Dummy) ---
def setup_dummy_knowledge_base():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Create dummy medical documents
    docs = [
        {"filename": "medical_guideline_1.txt", "content": "Medical Guideline 1: For common cold, rest and fluids are recommended. Antivirals are not typically effective. Seek medical attention if symptoms worsen or persist beyond 10 days. Always consult a physician for diagnosis and treatment. This guideline is for informational purposes only.\nSymptoms include: runny nose, sore throat, cough, congestion, slight body aches or a mild headache, sneezing, low-grade fever, general feeling of being unwell (malaise)."},
        {"filename": "drug_info_paracetamol.txt", "content": "Drug Information: Paracetamol (Acetaminophen) is a pain reliever and fever reducer. Dosage for adults is typically 500mg-1000mg every 4-6 hours, not exceeding 4000mg in 24 hours. Overdose can cause severe liver damage. Avoid alcohol while taking paracetamol. Consult a pharmacist or doctor before use, especially if pregnant or breastfeeding or with liver/kidney conditions.\nSide effects: Nausea, stomach pain, loss of appetite, dark urine, clay-colored stools, jaundice (yellowing of the skin or eyes)."},
        {"filename": "research_paper_diabetes.txt", "content": "Recent Research in Diabetes: New insulins and GLP-1 receptor agonists are showing promising results in managing type 2 diabetes. Lifestyle interventions including diet and exercise remain foundational. Early diagnosis and patient education are crucial for preventing complications such as neuropathy, retinopathy, and nephropathy. Continuous glucose monitoring has significantly improved patient outcomes. The importance of personalized treatment plans is increasingly recognized in the field.\nType 1 diabetes, an autoimmune disease, requires lifelong insulin therapy. Type 2 diabetes is characterized by insulin resistance and relative insulin deficiency."}
    ]

    for doc in docs:
        filepath = os.path.join(data_dir, doc["filename"])
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(doc["content"])
    return data_dir

# --- RAG Pipeline Initialization ---
def initialize_rag_pipeline(data_dir):
    # Load documents
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vectorstore
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_api_key)

    # Prompt Template
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    RAG_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )
    return qa_chain

# --- FastAPI Backend ---
app = FastAPI()

# Initialize RAG pipeline once for the FastAPI app
dummy_data_dir = setup_dummy_knowledge_base()
rag_chain = initialize_rag_pipeline(dummy_data_dir)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def process_query(request: QueryRequest):
    response = rag_chain.invoke({"query": request.query})
    return {"answer": response["result"]}


# --- Streamlit Frontend ---
if __name__ == "__main__":
    st.set_page_config(page_title="Medical Information Assistant")
    st.title("🩺 Medical Information Assistant (RAG)")
    st.markdown("Ask questions about medical guidelines, drug information, or research.")

    user_query = st.text_area("Enter your medical query here:", height=100)

    if st.button("Get Answer") and user_query:
        if not openai_api_key:
            st.error("OPENAI_API_KEY not found in environment variables. Please set it.")
        else:
            with st.spinner("Fetching and generating answer..."):
                try:
                    # Make a request to the FastAPI backend
                    fastapi_url = "http://localhost:8000/query"
                    response = requests.post(fastapi_url, json={"query": user_query})
                    response.raise_for_status() # Raise an exception for HTTP errors
                    result = response.json()
                    st.subheader("Answer:")
                    st.write(result["answer"])
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Please ensure the backend is running on http://localhost:8000.")
                    st.info("To run the backend, save this code as `medical_rag_assistant.py` and run in a separate terminal: `uvicorn medical_rag_assistant:app --host 0.0.0.0 --port 8000`")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    st.markdown("--- Your knowledge base consists of dummy medical guidelines, drug info, and research papers.")
    st.markdown("**How to run this application:**")
    st.markdown("1. Save this code as `medical_rag_assistant.py`.")
    st.markdown("2. Create a `.env` file in the same directory with `OPENAI_API_KEY='your_openai_api_key'`.")
    st.markdown("3. Install necessary libraries: `pip install -r requirements.txt` (or install individually: `fastapi uvicorn requests streamlit langchain-community langchain-openai pydantic sentence-transformers chromadb python-dotenv`) ")
    st.markdown("4. **Start the FastAPI backend (Terminal 1):** `uvicorn medical_rag_assistant:app --host 0.0.0.0 --port 8000`")
    st.markdown("5. **Start the Streamlit frontend (Terminal 2):** `streamlit run medical_rag_assistant.py`")
    st.markdown("6. Access the Streamlit app in your browser (usually `http://localhost:8501`).")



# To make this file runnable by `uvicorn`, the FastAPI app object 'app' is at the top level.
# The Streamlit code is guarded by `if __name__ == "__main__":` and also includes instructions
# to help the user understand how to run both components separately. 
# The `if __name__ == "__main__":` block is standard for Streamlit entry points. 
# For uvicorn, it imports 'app' directly.
