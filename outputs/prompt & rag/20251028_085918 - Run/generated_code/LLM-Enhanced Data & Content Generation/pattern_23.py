import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
from sentence_transformers import SentenceTransformer
from chromadb import Client, Settings
import networkx as nx
import json
import threading
import time

# --- 1. Medical Knowledge Graph (Mock) ---
# In-memory conceptual KG using NetworkX
def create_mock_kg():
    G = nx.Graph()

    # Diseases
    G.add_node("Influenza", type="disease", description="A common viral infection that can be deadly, especially in high-risk groups.")
    G.add_node("Pneumonia", type="disease", description="An infection that inflames air sacs in one or both lungs, which may fill with fluid.")
    G.add_node("Diabetes Type 2", type="disease", description="A chronic condition that affects the way the body processes blood sugar (glucose).")
    G.add_node("Hypertension", type="disease", description="High blood pressure, a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.")

    # Symptoms
    G.add_node("Fever", type="symptom")
    G.add_node("Cough", type="symptom")
    G.add_node("Shortness of Breath", type="symptom")
    G.add_node("Fatigue", type="symptom")
    G.add_node("Increased Thirst", type="symptom")
    G.add_node("Frequent Urination", type="symptom")
    G.add_node("Blurred Vision", type="symptom")
    G.add_add_node("Headache", type="symptom")
    G.add_node("Chest Pain", type="symptom")

    # Treatments
    G.add_node("Antivirals", type="treatment")
    G.add_node("Antibiotics", type="treatment")
    G.add_node("Insulin", type="treatment")
    G.add_node("Lifestyle Changes", type="treatment")
    G.add_node("ACE Inhibitors", type="treatment")
    G.add_node("Diuretics", type="treatment")

    # Relationships
    G.add_edge("Influenza", "Fever", relation="has_symptom")
    G.add_edge("Influenza", "Cough", relation="has_symptom")
    G.add_edge("Influenza", "Fatigue", relation="has_symptom")
    G.add_edge("Influenza", "Antivirals", relation="treatable_by")

    G.add_edge("Pneumonia", "Fever", relation="has_symptom")
    G.add_edge("Pneumonia", "Cough", relation="has_symptom")
    G.add_edge("Pneumonia", "Shortness of Breath", relation="has_symptom")
    G.add_edge("Pneumonia", "Antibiotics", relation="treatable_by")

    G.add_edge("Diabetes Type 2", "Increased Thirst", relation="has_symptom")
    G.add_edge("Diabetes Type 2", "Frequent Urination", relation="has_symptom")
    G.add_edge("Diabetes Type 2", "Fatigue", relation="has_symptom")
    G.add_edge("Diabetes Type 2", "Blurred Vision", relation="has_symptom")
    G.add_edge("Diabetes Type 2", "Insulin", relation="treatable_by")
    G.add_edge("Diabetes Type 2", "Lifestyle Changes", relation="treatable_by")

    G.add_edge("Hypertension", "Headache", relation="has_symptom")
    G.add_edge("Hypertension", "Chest Pain", relation="has_symptom")
    G.add_edge("Hypertension", "Fatigue", relation="has_symptom")
    G.add_edge("Hypertension", "ACE Inhibitors", relation="treatable_by")
    G.add_edge("Hypertension", "Diuretics", relation="treatable_by")
    G.add_edge("Hypertension", "Lifestyle Changes", relation="treatable_by")

    return G

mock_kg = create_mock_kg()

# --- 2. Embedding Model and Vector Store (Chroma) ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = Client(Settings(allow_reset=True))
try:
    chroma_client.delete_collection(name="medical_kg_embeddings")
except:
    pass # Collection might not exist yet
medical_collection = chroma_client.get_or_create_collection(name="medical_kg_embeddings")

def embed_and_add_to_vector_store(kg_graph):
    documents = []
    metadatas = []
    ids = []
    counter = 0

    for node, data in kg_graph.nodes(data=True):
        doc_content = f"Node: {node}. Type: {data.get('type', 'unknown')}. Description: {data.get('description', 'N/A')}"
        documents.append(doc_content)
        metadatas.append({"entity": node, "type": data.get('type')})
        ids.append(f"node_{counter}")
        counter += 1

    for u, v, data in kg_graph.edges(data=True):
        doc_content = f"Relationship: {u} is {data.get('relation', 'related to')} {v}."
        documents.append(doc_content)
        metadatas.append({"source": u, "target": v, "relation": data.get('relation')})
        ids.append(f"edge_{counter}")
        counter += 1

    if documents:
        embeddings = embedding_model.encode(documents).tolist()
        medical_collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)

embed_and_add_to_vector_store(mock_kg)

# --- 3. LLM Simulation ---
def simulate_llm_reasoning(query, context):
    if not context:
        return f"Based on the query '{query}', I couldn't retrieve specific medical context. Please provide more details."
    
    context_str = "\n".join([doc for doc in context])
    
    # Simple keyword-based simulation of LLM reasoning
    diagnosis = "Uncertain Diagnosis"
    recommendation = "Consult a medical professional for a proper diagnosis."

    if "Fever" in query and "Cough" in query:
        if "Shortness of Breath" in query:
            diagnosis = "Potential Pneumonia"
            recommendation = "Consider antibiotics and rest. Monitor oxygen levels."
        else:
            diagnosis = "Potential Influenza"
            recommendation = "Suggest antivirals if caught early, otherwise symptomatic treatment and rest."
    elif "Thirst" in query and "Urination" in query:
        diagnosis = "Possible Diabetes Type 2"
        recommendation = "Recommend blood glucose test and lifestyle modifications."
    elif "Headache" in query and "Chest Pain" in query:
        diagnosis = "Possible Hypertension"
        recommendation = "Advise blood pressure monitoring and lifestyle changes, potentially ACE inhibitors."
    
    return f"\nQuery: {query}\n\nRetrieved Medical Context:\n{context_str}\n\n--- LLM Reasoning ---\n**Diagnosis:** {diagnosis}\n**Recommendations:** {recommendation}\n\n*Note: This is a simulated response and not real medical advice.*"

# --- 4. LangChain-like Orchestration (Simplified) ---
def retrieve_and_reason(query):
    # 1. Embed Query
    query_embedding = embedding_model.encode([query]).tolist()

    # 2. Contextual Retrieval
    results = medical_collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include_documents=True
    )

    retrieved_context = [doc for doc in results['documents'][0]] if results['documents'] else []

    # 3. LLM Reasoning
    reasoning_output = simulate_llm_reasoning(query, retrieved_context)
    return reasoning_output

# --- 5. FastAPI Backend ---
app = FastAPI()

class QueryRequest(BaseModel):
    patient_query: str

@app.post("/diagnose")
async def diagnose_medical_condition(request: QueryRequest):
    recommendation = retrieve_and_reason(request.patient_query)
    return {"recommendation": recommendation}

# --- 6. Streamlit Frontend ---
def run_streamlit_app():
    st.title("Medical Diagnostic and Treatment Recommendation System")
    st.markdown("*(Disclaimer: This system is for informational purposes only and not a substitute for professional medical advice.)*\n")

    patient_symptoms = st.text_area("Enter patient symptoms or medical query:",
                                      "e.g., Patient has fever, cough, and fatigue for 3 days. What could it be?")

    if st.button("Get Diagnosis and Recommendations"):
        if patient_symptoms:
            with st.spinner("Processing your query..."):
                try:
                    response = requests.post("http://localhost:8000/diagnose", 
                                             json={"patient_query": patient_symptoms})
                    response.raise_for_status() # Raise an exception for HTTP errors
                    result = response.json()
                    st.subheader("Diagnosis and Recommendations:")
                    st.write(result.get("recommendation", "No recommendation found."))
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend API. Please ensure the FastAPI server is running.")
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter patient symptoms or a query.")


def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True # Allow the thread to exit when the main program exits
    fastapi_thread.start()

    # Give FastAPI a moment to start up
    time.sleep(2)
    
    run_streamlit_app()
