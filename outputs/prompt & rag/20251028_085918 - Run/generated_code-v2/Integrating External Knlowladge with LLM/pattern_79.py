import streamlit as st
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import networkx as nx
import requests
import json
import threading
import time

# --- 1. Medical Knowledge Graph (KG) Service --- 
class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self._build_mock_kg()

    def _build_mock_kg(self):
        self.graph.add_nodes_from([
            ("Diabetes", {"type": "disease"}),
            ("Insulin", {"type": "drug"}),
            ("Metformin", {"type": "drug"}),
            ("High Blood Sugar", {"type": "symptom"}),
            ("Pancreas", {"type": "organ"}),
            ("Type 2 Diabetes", {"type": "disease"}),
            ("Heart Disease", {"type": "complication"}),
            ("Kidney Failure", {"type": "complication"})
        ])
        self.graph.add_edges_from([
            ("Diabetes", "High Blood Sugar", {"relation": "has_symptom"}),
            ("Diabetes", "Insulin", {"relation": "treated_by"}),
            ("Type 2 Diabetes", "Metformin", {"relation": "treated_by"}),
            ("Insulin", "Pancreas", {"relation": "produced_by"}),
            ("Type 2 Diabetes", "Diabetes", {"relation": "is_a"}),
            ("Diabetes", "Heart Disease", {"relation": "can_lead_to"}),
            ("Diabetes", "Kidney Failure", {"relation": "can_lead_to"}),
            ("Metformin", "High Blood Sugar", {"relation": "reduces"})
        ])

    def lookup_entity(self, entity_name):
        found_entities = [node for node in self.graph.nodes if entity_name.lower() in node.lower()]
        return found_entities

    def get_neighbors(self, node, depth=1):
        if node not in self.graph:
            return []
        
        visited = {node}
        queue = [(node, 0)]
        relevant_facts = []

        while queue:
            current_node, current_depth = queue.pop(0)
            if current_depth > depth:
                continue

            for neighbor in self.graph.neighbors(current_node):
                relation = self.graph.get_edge_data(current_node, neighbor)['relation']
                relevant_facts.append(f"{current_node} {relation} {neighbor}")
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_depth + 1))
        return relevant_facts

# --- 2. LLM-based Topic Entity Extraction Module (Mock) ---
class LLMEntityExtractor:
    def __init__(self, medical_kg: MedicalKnowledgeGraph):
        self.medical_kg = medical_kg
        self.known_entities = set(node.lower() for node in medical_kg.graph.nodes)

    def extract_entities(self, question: str) -> list[str]:
        extracted = []
        words = question.lower().split()
        for entity in self.known_entities:
            if entity in question.lower():
                extracted.append(entity.replace(" ", "_").title()) # Simple capitalization
        return list(set(extracted))

# --- 3. KG Search and Reasoning Module --- 
class KGSearchAndReasoning:
    def __init__(self, medical_kg: MedicalKnowledgeGraph):
        self.medical_kg = medical_kg

    def search(self, entities: list[str], max_depth: int = 2) -> str:
        if not entities:
            return "No relevant medical entities identified for search."

        all_facts = []
        for entity in entities:
            # Try to find the exact entity in KG, handle case variations
            found_nodes = self.medical_kg.lookup_entity(entity)
            for node in found_nodes:
                facts = self.medical_kg.get_neighbors(node, depth=max_depth)
                all_facts.extend(facts)
        
        if not all_facts:
            return f"Could not find specific information related to {', '.join(entities)} in the knowledge graph."
        
        # Simple aggregation for prototyping
        response = f"Based on the entities ({', '.join(entities)}) and the knowledge graph, here's what I found:\n"
        for fact in sorted(list(set(all_facts))):
            response += f"- {fact}\n"
        return response

# --- FastAPI Backend Setup --- 
app = FastAPI()
medical_kg = MedicalKnowledgeGraph()
llm_extractor = LLMEntityExtractor(medical_kg)
kg_searcher = KGSearchAndReasoning(medical_kg)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def process_query(request: QueryRequest):
    extracted_entities = llm_extractor.extract_entities(request.question)
    kg_response = kg_searcher.search(extracted_entities)
    return {"extracted_entities": extracted_entities, "answer": kg_response}

# --- Streamlit Frontend Setup --- 
def streamlit_app():
    st.set_page_config(page_title="Medical KG QA System")
    st.title("🧠 Medical Knowledge Graph QA System")
    st.markdown("Ask a medical question and get answers from a knowledge graph, powered by LLM entity extraction.")

    user_question = st.text_input("Enter your medical question:", "What are the treatments for diabetes and its complications?")

    if st.button("Get Answer"):
        if user_question:
            st.info("Processing your question...")
            try:
                response = requests.post("http://localhost:8000/query", json={"question": user_question})
                response.raise_for_status() # Raise an exception for HTTP errors
                data = response.json()
                
                st.subheader("Extracted Entities:")
                if data["extracted_entities"]:
                    st.write(", ".join(data["extracted_entities"]))
                else:
                    st.write("No specific medical entities extracted.")

                st.subheader("KG Answer:")
                st.write(data["answer"])

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Make sure it's running (run this script and open http://localhost:8501).")
            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with backend: {e}")
        else:
            st.warning("Please enter a question.")

# --- Main entry point to run both FastAPI and Streamlit ---
def run_fastapi():
    # Use a custom Uvicorn configuration to suppress verbose logging
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning", workers=1)
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    # Check if a FastAPI thread is already running (e.g., in Streamlit re-runs)
    if "fastapi_thread" not in st.session_state:
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        st.session_state["fastapi_thread"] = fastapi_thread
        st.write("FastAPI backend starting...")
        time.sleep(2) # Give FastAPI a moment to start

    streamlit_app()
