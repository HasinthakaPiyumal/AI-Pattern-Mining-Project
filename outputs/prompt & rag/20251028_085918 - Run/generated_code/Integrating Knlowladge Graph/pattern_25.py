import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import networkx as nx
import json
import re
import sys
import threading
import time
import requests

medical_kg_data = {
    "nodes": [
        {"id": "Symptom:Fever", "type": "Symptom", "description": "Elevated body temperature."},
        {"id": "Symptom:Cough", "type": "Symptom", "description": "A sudden, forceful expulsion of air from the lungs."},
        {"id": "Symptom:Headache", "type": "Symptom", "description": "Pain in the head."},
        {"id": "Disease:Flu", "type": "Disease", "description": "A common viral infection that can be deadly."},
        {"id": "Disease:Common Cold", "type": "Disease", "description": "A viral infectious disease of the upper respiratory tract."},
        {"id": "Treatment:Rest", "type": "Treatment", "description": "Taking a break from physical activity."},
        {"id": "Treatment:Paracetamol", "type": "Treatment", "description": "A medication used to treat pain and fever."},
        {"id": "Drug:Aspirin", "type": "Drug", "description": "A salicylate drug, often used as an analgesic."},
        {"id": "Symptom:Sore Throat", "type": "Symptom", "description": "Pain or irritation of the throat."},
        {"id": "Disease:Strep Throat", "type": "Disease", "description": "A bacterial infection of the throat and tonsils."},
        {"id": "Treatment:Antibiotics", "type": "Treatment", "description": "Medications that destroy or slow down the growth of bacteria."},
    ],
    "edges": [
        {"source": "Symptom:Fever", "target": "Disease:Flu", "type": "INDICATES"},
        {"source": "Symptom:Cough", "target": "Disease:Flu", "type": "INDICATES"},
        {"source": "Symptom:Headache", "target": "Disease:Flu", "type": "INDICATES"},
        {"source": "Disease:Flu", "target": "Treatment:Rest", "type": "RECOMMENDED_TREATMENT"},
        {"source": "Disease:Flu", "target": "Treatment:Paracetamol", "type": "RECOMMENDED_TREATMENT"},
        {"source": "Symptom:Cough", "target": "Disease:Common Cold", "type": "INDICATES"},
        {"source": "Symptom:Sore Throat", "target": "Disease:Common Cold", "type": "INDICATES"},
        {"source": "Disease:Common Cold", "target": "Treatment:Rest", "type": "RECOMMENDED_TREATMENT"},
        {"source": "Disease:Common Cold", "target": "Drug:Aspirin", "type": "MAY_BE_TREATED_WITH"},
        {"source": "Symptom:Sore Throat", "target": "Disease:Strep Throat", "type": "INDICATES"},
        {"source": "Symptom:Fever", "target": "Disease:Strep Throat", "type": "INDICATES"},
        {"source": "Disease:Strep Throat", "target": "Treatment:Antibiotics", "type": "RECOMMENDED_TREATMENT"},
    ]
}

kg_graph = nx.DiGraph()
for node_data in medical_kg_data["nodes"]:
    kg_graph.add_node(node_data["id"], **node_data)
for edge_data in medical_kg_data["edges"]:
    kg_graph.add_edge(edge_data["source"], edge_data["target"], type=edge_data["type"])

def query_kg(query_type, entity=None, relationship=None, target_type=None):
    results = []
    if query_type == "get_related":
        if entity and relationship:
            for source, target, data in kg_graph.edges(data=True):
                if source == entity and data["type"] == relationship:
                    results.append(target)
        elif entity and target_type:
             for neighbor in kg_graph.neighbors(entity):
                 if kg_graph.nodes[neighbor]["type"] == target_type:
                     results.append(neighbor)
    elif query_type == "get_entities_by_type":
        for node, data in kg_graph.nodes(data=True):
            if data["type"] == target_type:
                results.append(node)
    elif query_type == "get_node_description":
        if entity and entity in kg_graph.nodes:
            return kg_graph.nodes[entity].get("description", "No description available.")
        return None
    return results

def get_llm_response(prompt):
    if "diagnose" in prompt.lower() and "fever" in prompt.lower() and "cough" in prompt.lower():
        return "Based on your symptoms (fever, cough), potential conditions include Flu and Common Cold. Flu typically involves more severe body aches and fatigue. Common Cold is milder."
    elif "treatment for Flu" in prompt:
        return "Recommended treatments for Flu are Rest, Paracetamol, and staying hydrated. Antiviral drugs may be prescribed in some cases."
    elif "explain Flu" in prompt:
        return query_kg("get_node_description", entity="Disease:Flu") or "Flu is a viral infection that attacks your respiratory system. Common symptoms include fever, cough, sore throat, and body aches."
    elif "explain Common Cold" in prompt:
        return query_kg("get_node_description", entity="Disease:Common Cold") or "Common Cold is a viral infectious disease of the upper respiratory tract. Symptoms are usually milder than flu."
    elif "explain Strep Throat" in prompt:
        return query_kg("get_node_description", entity="Disease:Strep Throat") or "Strep throat is a bacterial infection of the throat and tonsils, often causing fever and a very sore throat."
    elif "path from Symptom:Fever" in prompt:
        return "Symptom:Fever -> INDICATES -> Disease:Flu"
    else:
        return "I need more information to provide a precise answer. Please refine your query."

def extract_entities(text):
    entities = []
    keywords = {
        "fever": "Symptom:Fever", "cough": "Symptom:Cough", "headache": "Symptom:Headache",
        "flu": "Disease:Flu", "common cold": "Disease:Common Cold", "rest": "Treatment:Rest",
        "paracetamol": "Treatment:Paracetamol", "aspirin": "Drug:Aspirin", "sore throat": "Symptom:Sore Throat",
        "strep throat": "Disease:Strep Throat", "antibiotics": "Treatment:Antibiotics"
    }
    text_lower = text.lower()
    for keyword, entity_id in keywords.items():
        if keyword in text_lower:
            entities.append(entity_id)
    return list(set(entities))

def semantic_parse_and_query(natural_language_query):
    entities = extract_entities(natural_language_query)
    if "diagnose" in natural_language_query.lower() or any("Symptom" in e for e in entities):
        symptoms = [e for e in entities if "Symptom" in e]
        if symptoms:
            return {"type": "get_diagnosis", "symptoms": symptoms}
        return {"type": "no_match"}
    elif "treatment for" in natural_language_query.lower() or any("Disease" in e for e in entities):
        disease = [e for e in entities if "Disease" in e]
        if disease:
            return {"type": "get_related", "entity": disease[0], "relationship": "RECOMMENDED_TREATMENT", "target_type": "Treatment"}
        return {"type": "no_match"}
    elif "explain" in natural_language_query.lower() or any("Disease" in e for e in entities):
        entity_to_explain = [e for e in entities if "Disease" in e or "Symptom" in e or "Treatment" in e]
        if entity_to_explain:
            return {"type": "get_explanation", "entity": entity_to_explain[0]}
        return {"type": "no_match"}
    return {"type": "no_match"}

def retrieve_and_augment(llm_prompt, kg_query_results):
    kg_facts = "\n".join([f"- {fact}" for fact in kg_query_results])
    if kg_facts:
        return f"{llm_prompt}\n\nRelevant KG Facts:\n{kg_facts}"
    return llm_prompt

def llm_guided_beam_search(start_entity, target_type, beam_width=2, max_depth=3):
    paths = []
    current_beam = [(start_entity, [start_entity])]

    for _ in range(max_depth):
        next_beam = []
        for current_node, current_path in current_beam:
            if not kg_graph.has_node(current_node):
                continue
            neighbors_with_edges = []
            for neighbor in kg_graph.neighbors(current_node):
                if kg_graph.has_edge(current_node, neighbor):
                    edge_type = kg_graph.edges[current_node, neighbor]["type"]
                    neighbors_with_edges.append((neighbor, edge_type))

            scored_neighbors = []
            for neighbor, edge_type in neighbors_with_edges:
                score = 1
                if kg_graph.nodes[neighbor]["type"] == target_type:
                    score = 10
                elif neighbor.startswith("Disease") and target_type == "Disease":
                    score = 5
                elif neighbor.startswith("Treatment") and target_type == "Treatment":
                    score = 5
                scored_neighbors.append((neighbor, score, edge_type))

            scored_neighbors.sort(key=lambda x: x[1], reverse=True)
            for neighbor, _, edge_type in scored_neighbors[:beam_width]:
                current_node_name = current_node.split(':', 1)[1] if ':' in current_node else current_node
                neighbor_name = neighbor.split(':', 1)[1] if ':' in neighbor else neighbor
                path_segment = f"{current_node.split(':')[0]}:{current_node_name} -> {edge_type} -> {neighbor.split(':')[0]}:{neighbor_name}"
                new_path_repr = current_path + [path_segment]
                paths.append(new_path_repr)
                next_beam.append((neighbor, new_path_repr))
        current_beam = next_beam
        if not current_beam:
            break
    return paths

class ClinicalAgent:
    def __init__(self):
        self.history = []

    def run_diagnosis(self, patient_query):
        self.history.append(f"Patient Query: {patient_query}")

        entities = extract_entities(patient_query)
        self.history.append(f"Extracted Entities: {entities}")

        initial_kg_query = semantic_parse_and_query(patient_query)
        self.history.append(f"Initial KG Query: {initial_kg_query}")

        diagnosis_results = []
        reasoning_steps = []

        if initial_kg_query["type"] == "get_diagnosis" and initial_kg_query["symptoms"]:
            llm_thought = get_llm_response(f"Based on symptoms: {', '.join([s.split(':', 1)[1] for s in initial_kg_query['symptoms']])}, what are the most likely diseases?")
            reasoning_steps.append(f"LLM Thought (Initial Hypothesis): {llm_thought}")
            self.history.append(f"LLM Thought: {llm_thought}")

            potential_diseases_from_llm = [ent for ent in extract_entities(llm_thought) if "Disease" in ent]
            potential_diseases_from_kg = []
            for symptom in initial_kg_query['symptoms']:
                related_diseases = query_kg("get_related", entity=symptom, target_type="Disease")
                potential_diseases_from_kg.extend(related_diseases)

            potential_diseases = list(set(potential_diseases_from_llm + potential_diseases_from_kg))

            if potential_diseases:
                reasoning_steps.append(f"KG Exploration for potential diseases based on LLM hypothesis and direct symptom-disease links: {', '.join([d.split(':',1)[1] for d in potential_diseases])}")
                self.history.append(f"Potential Diseases: {potential_diseases}")

                for disease in potential_diseases:
                    disease_facts = query_kg("get_related", entity=disease, relationship="INDICATES")
                    disease_facts.extend(query_kg("get_related", entity=disease, relationship="RECOMMENDED_TREATMENT"))
                    description = query_kg("get_node_description", entity=disease)
                    if description: disease_facts.append(description)
                    
                    disease_facts_str = "\n".join([f for f in disease_facts if f])
                    reasoning_steps.append(f"Retrieved KG facts for {disease.split(':',1)[1]}:\n{disease_facts_str}")
                    self.history.append(f"KG Facts for {disease}: {disease_facts_str}")

                    rag_prompt = retrieve_and_augment(f"Explain {disease.split(':',1)[1]} and its common symptoms and treatments in simple terms.", disease_facts)
                    llm_explanation = get_llm_response(rag_prompt)
                    reasoning_steps.append(f"LLM Explanation for {disease.split(':',1)[1]} (RAG-enhanced): {llm_explanation}")
                    self.history.append(f"LLM Explanation for {disease}: {llm_explanation}")

                    example_path = []
                    if initial_kg_query["symptoms"]:
                        for symptom in initial_kg_query["symptoms"]:
                            beam_paths = llm_guided_beam_search(symptom, "Disease")
                            for path in beam_paths:
                                # Check if the disease is the last entity in the last triple of the path
                                if path and len(path) > 1 and path[-1].endswith(f"> {disease.split(':', 1)[1] if ':' in disease else disease}"):
                                    example_path = path
                                    break
                            if example_path:
                                break
                    if example_path:
                        reasoning_steps.append(f"Example Reasoning Path to {disease.split(':',1)[1]} (Triple-Based): {' -> '.join(example_path)}")

                    diagnosis_results.append({
                        "disease": disease.replace("Disease:", ""),
                        "explanation": llm_explanation,
                        "kg_facts": [f.split(':', 1)[1] if ':' in f else f for f in disease_facts if f],
                        "reasoning_path_example": example_path
                    })

        if not diagnosis_results:
            llm_response = get_llm_response(f"Given the patient query: {patient_query}, what is your initial assessment?")
            diagnosis_results.append({"disease": "Uncertain/Further Consultation Needed", "explanation": llm_response, "kg_facts": [], "reasoning_path_example": []})
            reasoning_steps.append(f"No specific diagnosis from KG, LLM provides general assessment: {llm_response}")

        final_explanation = "Based on the integration of LLM reasoning and medical Knowledge Graph:\n"
        for res in diagnosis_results:
            final_explanation += f"\n**Potential Diagnosis:** {res['disease']}\n"
            final_explanation += f"**Explanation:** {res['explanation']}\n"
            if res['reasoning_path_example']:
                final_explanation += f"**Example Reasoning Path:** {' -> '.join(res['reasoning_path_example'])}\n"
            final_explanation += f"**Relevant KG Facts:** {', '.join(res['kg_facts'])}\n"

        return {
            "diagnosis": diagnosis_results,
            "overall_explanation": final_explanation,
            "reasoning_steps": reasoning_steps
        }

app = FastAPI()

class PatientQuery(BaseModel):
    query: str

@app.post("/diagnose")
async def diagnose_patient_api(patient_query: PatientQuery):
    agent = ClinicalAgent()
    result = agent.run_diagnosis(patient_query.query)
    return result

def run_streamlit_app():
    st.set_page_config(layout="wide")
    st.title("Clinical Decision Support System with Explainable AI")
    st.markdown("---")

    st.sidebar.header("Patient Information")
    patient_input = st.sidebar.text_area("Enter patient symptoms and medical history (e.g., 'Patient has a fever and cough for 3 days. Also experiencing a headache.')", height=200)

    use_api = st.sidebar.checkbox("Use FastAPI Backend (Run backend separately)", False)

    if st.sidebar.button("Get Diagnosis"):
        if patient_input:
            st.subheader("Processing Request...")
            try:
                if use_api:
                    try:
                        response = requests.post("http://localhost:8000/diagnose", json={"query": patient_input})
                        response.raise_for_status()
                        diagnosis_output = response.json()
                        st.info("Diagnosis retrieved from FastAPI backend.")
                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to FastAPI backend. Please ensure it's running (python clinical_decision_support_system.py --backend).")
                        return
                else:
                    agent = ClinicalAgent()
                    diagnosis_output = agent.run_diagnosis(patient_input)
                    st.info("Diagnosis generated directly by the agent in Streamlit.")

                st.success("Diagnosis Complete!")

                st.subheader("Proposed Diagnoses and Explanations")
                st.markdown(diagnosis_output["overall_explanation"])

                st.subheader("Step-by-Step Reasoning (Chain-of-Thought)")
                for step in diagnosis_output["reasoning_steps"]:
                    st.write(f"- {step}")

                if not use_api:
                    st.subheader("Agent's Internal History")
                    with st.expander("Show Internal History"):
                        for item in agent.history:
                            st.write(item)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.sidebar.warning("Please enter patient information to get a diagnosis.")

def run_backend():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    if "--backend" in sys.argv:
        print("Starting FastAPI backend on http://0.0.0.0:8000")
        run_backend()
    else:
        print("To run the Streamlit frontend: streamlit run clinical_decision_support_system.py")
        print("To run the FastAPI backend separately: python clinical_decision_support_system.py --backend")
        run_streamlit_app() # This line will only be reached if run via `python` directly without --backend. It's mainly for local testing before `streamlit run` is used.
