import os
import uvicorn
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

import openai

import chromadb
from sentence_transformers import SentenceTransformer

import streamlit as st
import sys

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
CHROMA_DB_PATH = "./chroma_db"

class QueryRequest(BaseModel):
    patient_id: str
    query: str

class QueryResponse(BaseModel):
    response: str
    context_used: List[str]

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    medical_records_collection = client.get_or_create_collection(name="medical_records")
except Exception:
    client.delete_collection(name="medical_records")
    medical_records_collection = client.create_collection(name="medical_records")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()

def ingest_dummy_data():
    if medical_records_collection.count() == 0:
        documents = [
            {"id": "pat1_rec1", "text": "Patient John Doe, 55, presented with chest pain on 2023-10-26. Diagnosed with angina. Prescribed nitroglycerin."},
            {"id": "pat1_rec2", "text": "Follow-up for John Doe on 2023-11-15. Chest pain improved. Discussed diet and exercise."},
            {"id": "pat2_rec1", "text": "Patient Jane Smith, 30, diagnosed with seasonal allergies in April 2023. Prescribed antihistamines."},
            {"id": "pat1_conv1", "text": "Doctor's note from recent conversation with John Doe: Expressed concerns about potential side effects of medication."},
            {"id": "pat1_conv2", "text": "Nurse's log from John Doe's last visit: Discussed blood pressure monitoring and importance of adherence."},
            {"id": "medical_research_heart_disease", "text": "A study on the efficacy of statins in preventing recurrent cardiovascular events in patients with a history of angina."},
            {"id": "drug_info_nitroglycerin", "text": "Nitroglycerin is a vasodilator used to treat and prevent angina. Common side effects include headache and dizziness."},
        ]
        metadatas = [
            {"patient_id": "pat1", "type": "medical_record", "date": "2023-10-26"},
            {"patient_id": "pat1", "type": "medical_record", "date": "2023-11-15"},
            {"patient_id": "pat2", "type": "medical_record", "date": "2023-04-01"},
            {"patient_id": "pat1", "type": "conversation_summary", "date": "2024-01-20"},
            {"patient_id": "pat1", "type": "conversation_log", "date": "2024-01-25"},
            {"patient_id": "global_knowledge", "type": "medical_research"},
            {"patient_id": "global_knowledge", "type": "drug_information"},
        ]
        embeddings = [get_embedding(doc["text"]) for doc in documents]
        ids = [doc["id"] for doc in documents]
        
        medical_records_collection.add(
            embeddings=embeddings,
            documents=[doc["text"] for doc in documents],
            metadatas=metadatas,
            ids=ids
        )
    else:
        pass

ingest_dummy_data()

def get_relevant_context(patient_id: str, query: str, top_k: int = 5) -> List[str]:
    query_embedding = get_embedding(query)

    patient_results = medical_records_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"patient_id": patient_id},
        include=["documents"]
    )
    patient_context = patient_results.get("documents", [[]])[0] if patient_results and patient_results.get("documents") else []

    general_results = medical_records_collection.query(
        query_embeddings=[query_embedding],
        n_results=2,
        where={"patient_id": {"$ne": patient_id}, "type": {"$in": ["medical_research", "drug_information"]}},
        include=["documents"]
    )
    general_context = general_results.get("documents", [[]])[0] if general_results and general_results.get("documents") else []

    combined_context = []
    combined_context.extend(patient_context)
    combined_context.extend(general_context)

    max_context_length_chars = 1000
    final_context_str = ""
    for doc in combined_context:
        if len(final_context_str) + len(doc) + 2 < max_context_length_chars:
            final_context_str += doc + "\n"
        else:
            remaining_length = max_context_length_chars - len(final_context_str) - 2
            if remaining_length > 0:
                final_context_str += doc[:remaining_length] + "...\n"
            break
    
    return [s for s in final_context_str.strip().split("\n") if s]


def call_llm(prompt: str) -> str:
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        return "LLM call skipped: OpenAI API key not configured. Returning mock response. Prompt: " + prompt[:200] + "..."

    try:
        openai.api_key = OPENAI_API_KEY
        client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Provide concise and accurate information based on the given context. If the context does not contain enough information, clearly state that you cannot answer based on the provided information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error communicating with LLM: {e}. Returning mock response. Prompt: {prompt[:200]}..."


app = FastAPI()

@app.post("/query", response_model=QueryResponse)
async def process_query_api(request: QueryRequest):
    relevant_context = get_relevant_context(request.patient_id, request.query)
    
    context_str = "\n".join(relevant_context)
    full_prompt = (
        f"Patient ID: {request.patient_id}\n"
        f"Medical Context:\n{context_str}\n\n"
        f"User Query: {request.query}\n\n"
        f"Based on the medical context, answer the user's query."
    )
    
    llm_response = call_llm(full_prompt)
    
    return QueryResponse(response=llm_response, context_used=relevant_context)


def streamlit_frontend():
    st.set_page_config(page_title="Smart Medical Assistant")
    st.title("👨‍⚕️ Smart Medical Assistant")

    st.markdown(
        """
        This assistant helps doctors and patients by providing information from medical records and general knowledge,
        effectively managing long context using a Retrieval Augmented Generation (RAG) approach.
        """
    )

    st.header("Ask a Question")

    patient_id = st.text_input("Patient ID (e.g., 'pat1', 'pat2')", value="pat1")
    user_query = st.text_area("Your medical question:", "What were John Doe's last reported symptoms and treatment for chest pain?")

    if st.button("Get Answer"):
        if not patient_id or not user_query:
            st.error("Please provide both Patient ID and a query.")
        else:
            with st.spinner("Processing your query..."):
                try:
                    api_url = "http://127.0.0.1:8000/query"
                    payload = {"patient_id": patient_id, "query": user_query}
                    
                    response = httpx.post(api_url, json=payload, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    st.success("Response from Assistant:")
                    st.write(data["response"])

                    with st.expander("Context Used"):
                        if data["context_used"]:
                            for i, context_item in enumerate(data["context_used"]):
                                st.write(f"- {context_item}")
                        else:
                            st.write("No specific context retrieved for this query.")

                except httpx.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Please ensure it is running by executing 'python main.py' in a separate terminal.")
                except httpx.exceptions.RequestError as e:
                    st.error(f"An error occurred during the API request: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if "streamlit" in sys.argv[0]:
        streamlit_frontend()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)