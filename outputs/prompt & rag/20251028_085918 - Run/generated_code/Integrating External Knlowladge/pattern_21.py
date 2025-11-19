import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import json
import os

# Mock external libraries/services
class MockPubMedAPI:
    def search(self, query):
        if "diabetes" in query.lower():
            return {"results": ["Recent study on Type 2 diabetes management", "Clinical trials for new insulin therapies"]}
        return {"results": ["No specific medical articles found for your query."]}

class MockDrugDatabaseAPI:
    def get_drug_info(self, drug_name):
        if "metformin" in drug_name.lower():
            return {"name": "Metformin", "class": "Biguanide", "uses": "Type 2 Diabetes", "side_effects": "Nausea, diarrhea"}
        return {"name": drug_name, "info": "Information not found."}

class MockEHRSystem:
    def get_patient_history(self, patient_id):
        if patient_id == "P001":
            return {"patient_id": "P001", "conditions": ["Type 2 Diabetes", "Hypertension"], "medications": ["Metformin", "Lisinopril"], "allergies": ["Penicillin"]}
        return {"patient_id": patient_id, "history": "Not found."}

class MockVectorDB:
    def __init__(self):
        self.store = {}
        self.embeddings_map = {}

    def add_document(self, doc_id, text, embedding):
        self.store[doc_id] = text
        self.embeddings_map[doc_id] = embedding

    def search(self, query_embedding, top_k=1):
        # Simplified: just return a dummy document
        if self.store:
            doc_id = list(self.store.keys())[0]
            return [(doc_id, self.store[doc_id], 0.9)]
        return []

class MockKnowledgeGraph:
    def __init__(self):
        self.graph = {
            "Diabetes": [("has_symptom", "Frequent urination"), ("treated_by", "Metformin")],
            "Hypertension": [("has_symptom", "Headaches"), ("treated_by", "Lisinopril")],
            "Metformin": [("treats", "Diabetes"), ("class", "Biguanide")]
        }

    def query_relation(self, entity, relation_type=None):
        results = []
        if entity in self.graph:
            for rel, target in self.graph[entity]:
                if relation_type is None or rel == relation_type:
                    results.append((entity, rel, target))
        return results

# Mock LLM and Embedding models
class MockLLM:
    def generate(self, prompt, tools=None):
        if "pubmed" in prompt.lower() and tools and "pubmed_search" in tools:
            query = prompt.split("pubmed:")[-1].strip()
            pubmed_results = tools["pubmed_search"](query)
            return f"Based on PubMed: {pubmed_results}"
        if "drug info" in prompt.lower() and tools and "drug_info_tool" in tools:
            drug_name = prompt.split("drug info:")[-1].strip()
            drug_details = tools["drug_info_tool"](drug_name)
            return f"Drug details: {drug_details}"
        if "patient history" in prompt.lower() and tools and "ehr_lookup" in tools:
            patient_id = prompt.split("patient history:")[-1].strip()
            history = tools["ehr_lookup"](patient_id)
            return f"Patient {patient_id} history: {history}"
        if "knowledge graph" in prompt.lower() and tools and "kg_query_tool" in tools:
            entity = prompt.split("knowledge graph:")[-1].strip()
            kg_results = tools["kg_query_tool"](entity)
            return f"KG results for {entity}: {kg_results}"
        if "web browse" in prompt.lower() and tools and "browser_tool" in tools:
            url = prompt.split("web browse:")[-1].strip()
            web_content = tools["browser_tool"](url)
            return f"Web content from {url}: {web_content[:100]}..."
        if "diagnosis for" in prompt.lower():
            return "Potential diagnosis: Consult a specialist."
        return f"LLM response to: {prompt}"

class MockEmbeddingModel:
    def embed(self, text):
        return [hash(text) % 1000]

# Initialize Mocks
pubmed_api = MockPubMedAPI()
drug_db_api = MockDrugDatabaseAPI()
ehr_system = MockEHRSystem()
vector_db = MockVectorDB()
kg_system = MockKnowledgeGraph()
llm = MockLLM()
embedding_model = MockEmbeddingModel()

# --- Knowledge Consolidation Pipelines (Simplified as functions) ---
def entity_linker(text):
    entities = []
    if "diabetes" in text.lower(): entities.append("Diabetes")
    if "metformin" in text.lower(): entities.append("Metformin")
    return {"text": text, "entities": entities}

def evidence_synthesizer(retrieved_data):
    synthesis = ""
    for item in retrieved_data:
        if isinstance(item, dict) and "results" in item: synthesis += f"\nResearch: {item['results']}"
        if isinstance(item, dict) and "info" in item: synthesis += f"\nDrug Info: {item['info']}"
        if isinstance(item, dict) and "history" in item: synthesis += f"\nPatient History: {item['history']}"
        if isinstance(item, list) and all(isinstance(t, tuple) for t in item): synthesis += f"\nKG Relations: {item}"
    return synthesis if synthesis else "No synthesis possible."

def info_structurer(consolidated_info):
    structured_output = {"summary": consolidated_info[:200], "details": consolidated_info}
    return structured_output

# --- Langchain-like Tools (Simplified) ---
def pubmed_search_tool(query: str):
    return pubmed_api.search(query)

def drug_info_tool(drug_name: str):
    return drug_db_api.get_drug_info(drug_name)

def ehr_lookup_tool(patient_id: str):
    return ehr_system.get_patient_history(patient_id)

def kg_query_tool(entity: str, relation: str = None):
    return kg_system.query_relation(entity, relation)

def browser_tool(url: str):
    if not url.startswith("https://www.who.int/") and not url.startswith("https://www.cdc.gov/") and not url.startswith("https://www.fda.gov/"):
        return "Access to this URL is restricted for safety."
    return f"Simulated web content from {url} (controlled access)." # Mock web content

agent_tools = {
    "pubmed_search": pubmed_search_tool,
    "drug_info_tool": drug_info_tool,
    "ehr_lookup": ehr_lookup_tool,
    "kg_query_tool": kg_query_tool,
    "browser_tool": browser_tool,
}

# --- FastAPI Backend ---
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    patient_id: str = None

@app.post("/mrda/query")
async def mrda_query(request: QueryRequest):
    try:
        user_query = request.query
        patient_context = None
        if request.patient_id:
            patient_context = ehr_lookup_tool(request.patient_id)

        llm_prompt = f"Given the user query: '{user_query}'"
        if patient_context: llm_prompt += f" and patient context: {patient_context}"
        llm_prompt += "\nUse available tools to answer."

        # Simulate LLM deciding which tool to use and generating a response
        # In a real Langchain agent, this would be handled dynamically.
        response_content = llm.generate(llm_prompt, tools=agent_tools)

        retrieved_data = []
        if "pubmed" in response_content.lower():
            retrieved_data.append(pubmed_search_tool(user_query))
        if "drug details" in response_content.lower():
            drug_name_from_prompt = "metformin" # Simplified extraction
            retrieved_data.append(drug_info_tool(drug_name_from_prompt))
        if "kg results" in response_content.lower():
            entity_from_prompt = "Diabetes" # Simplified extraction
            retrieved_data.append(kg_query_tool(entity_from_prompt))
        if "web content" in response_content.lower():
            url_from_prompt = "https://www.who.int/health-topics/diabetes" # Simplified extraction
            retrieved_data.append(browser_tool(url_from_prompt))

        # Knowledge Consolidation Pipeline
        linked_entities = entity_linker(user_query + str(patient_context) + response_content)
        synthesized_evidence = evidence_synthesizer(retrieved_data)
        structured_output = info_structurer(synthesized_evidence + "\n" + response_content)

        final_response = {
            "query": user_query,
            "llm_raw_response": response_content,
            "retrieved_data": retrieved_data,
            "linked_entities": linked_entities["entities"],
            "synthesized_evidence": synthesized_evidence,
            "structured_response": structured_output
        }

        return final_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Streamlit Frontend ---
st.set_page_config(page_title="MRDA - Medical Assistant")
st.title("Medical Research and Diagnostic Assistant")

if 'fastapi_process' not in st.session_state:
    st.session_state.fastapi_process = None

def start_fastapi():
    if st.session_state.fastapi_process is None:
        # This is a highly simplified way to start FastAPI from Streamlit
        # In a real deployment, FastAPI would be running as a separate service
        # and Streamlit would simply call its deployed URL.
        # For local development, running it in a separate thread/process is common.
        # Here, we are just pretending it's running.
        st.session_state.fastapi_process = True
        st.success("FastAPI backend is conceptually running.")

start_fastapi()

query_input = st.text_area("Enter your medical query:", "What are the latest treatments for Type 2 Diabetes?")
patient_id_input = st.text_input("Optional: Enter Patient ID for context (e.g., P001):")

if st.button("Get Assistance"):
    if not query_input:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing your request..."):
            try:
                api_url = "http://127.0.0.1:8000/mrda/query"
                payload = {"query": query_input}
                if patient_id_input: payload["patient_id"] = patient_id_input

                response = requests.post(api_url, json=payload)
                response.raise_for_status() 
                result = response.json()

                st.subheader("MRDA Response:")
                st.json(result)

                st.markdown("### Key Findings")
                st.write(result["structured_response"]["summary"])
                st.markdown("### Detailed Information")
                st.write(result["structured_response"]["details"])

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Make sure it is running (e.g., in a separate terminal via `uvicorn mrda_app:app --reload`).")
                st.info("To run FastAPI, save this code as `mrda_app.py` and run `uvicorn mrda_app:app --reload` in your terminal. Then run `streamlit run mrda_app.py` in another terminal.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error from backend: {e}")
                st.json(response.json() if 'response' in locals() else "No response object")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

