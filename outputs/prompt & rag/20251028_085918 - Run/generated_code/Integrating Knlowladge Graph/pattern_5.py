"""
MedGraphReason: An AI-Powered Diagnostic Assistant leveraging Knowledge Graph Agentic Reasoning in Healthcare.

This single file contains mocked implementations for the Streamlit UI, FastAPI Backend, 
LLM Agent, and Knowledge Graph (KG) interaction modules to demonstrate the architecture.

To run this application:
1.  Ensure you have the necessary libraries installed: `pip install streamlit fastapi uvicorn pydantic`
2.  Save this code as `medgraph_reason.py`.
3.  Open two separate terminal windows.
4.  In the first terminal, run the FastAPI backend:
    `uvicorn medgraph_reason:app --reload --port 8000`
5.  In the second terminal, run the Streamlit frontend:
    `streamlit run medgraph_reason.py`
6.  Access the Streamlit application in your browser, typically at http://localhost:8501
"""

import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import json
import threading
import time

# --- 1. Mock Medical Knowledge Graph (KG) Data Store ---
# In a real application, this would be a Neo4j database.
# We're using a simple list of dictionaries to simulate medical facts (triples).
medical_knowledge_graph = [
    {"subject": "Fever", "predicate": "is_symptom_of", "object": "Influenza"},
    {"subject": "Cough", "predicate": "is_symptom_of", "object": "Influenza"},
    {"subject": "Headache", "predicate": "is_symptom_of", "object": "Influenza"},
    {"subject": "Influenza", "predicate": "has_treatment", "object": "Antivirals"},
    {"subject": "Influenza", "predicate": "has_treatment", "object": "Rest and Fluids"},
    {"subject": "Influenza", "predicate": "has_prevention", "object": "Vaccination"},
    {"subject": "Chest Pain", "predicate": "is_symptom_of", "object": "Pneumonia"},
    {"subject": "Shortness of Breath", "predicate": "is_symptom_of", "object": "Pneumonia"},
    {"subject": "Pneumonia", "predicate": "has_treatment", "object": "Antibiotics"},
    {"subject": "Diabetes", "predicate": "is_condition_of", "object": "Endocrine System"},
    {"subject": "High Blood Sugar", "predicate": "is_symptom_of", "object": "Diabetes"},
    {"subject": "Diabetes", "predicate": "has_treatment", "object": "Insulin Therapy"},
    {"subject": "Hypertension", "predicate": "is_condition_of", "object": "Cardiovascular System"},
    {"subject": "High Blood Pressure", "predicate": "is_symptom_of", "object": "Hypertension"},
    {"subject": "Hypertension", "predicate": "has_treatment", "object": "ACE Inhibitors"},
    {"subject": "Fatigue", "predicate": "is_symptom_of", "object": "Anemia"},
    {"subject": "Pallor", "predicate": "is_symptom_of", "object": "Anemia"},
    {"subject": "Anemia", "predicate": "has_treatment", "object": "Iron Supplements"},
]

# --- 2. KG Interaction & Semantic Pruning Module (Mock) ---
class KGInteractionModule:
    def __init__(self, kg_data: List[Dict]):
        self.kg_data = kg_data

    def query_translator(self, natural_language_query: str) -> List[str]:
        """Translates natural language input into 'keywords' for KG querying."""
        keywords = [word.lower() for word in natural_language_query.split() if len(word) > 2]
        print(f"[KG] Translating '{natural_language_query}' to keywords: {keywords}")
        return keywords

    def kg_executor(self, keywords: List[str]) -> List[Dict]:
        """Executes a 'query' against the mock KG data and returns relevant facts."""
        results = []
        for fact in self.kg_data:
            # Simple keyword matching across subject, predicate, object
            if any(k in fact["subject"].lower() or 
                   k in fact["predicate"].lower() or 
                   k in fact["object"].lower() for k in keywords):
                results.append(fact)
        print(f"[KG] Executed query with keywords {keywords}, found {len(results)} facts.")
        return results

    def semantic_pruner(self, retrieved_facts: List[Dict], patient_context: str) -> List[Dict]:
        """Mocks semantic pruning by filtering facts based on a simple heuristic.
        In a real system, this would use embeddings and similarity measures."""
        pruned_facts = []
        patient_keywords = self.query_translator(patient_context)
        for fact in retrieved_facts:
            # A fact is 'relevant' if its subject or object is in the patient keywords
            # or if it's a common medical relation.
            relevance_score = 0
            if any(k in fact["subject"].lower() for k in patient_keywords):
                relevance_score += 1
            if any(k in fact["object"].lower() for k in patient_keywords):
                relevance_score += 1
            
            # Prioritize facts directly related to symptoms or diseases
            if "symptom_of" in fact["predicate"] or "treatment" in fact["predicate"]:
                relevance_score += 1

            if relevance_score >= 1: # Simple threshold for inclusion
                pruned_facts.append(fact)
        print(f"[KG] Pruned {len(retrieved_facts)} facts to {len(pruned_facts)} based on patient context.")
        return pruned_facts

# --- 3. LLM Agent Module (Mock) ---
class LLMAgent:
    def __init__(self, kg_interaction_module: KGInteractionModule):
        self.kg_interaction_module = kg_interaction_module

    def generate_query(self, patient_data: Dict) -> str:
        """Mocks LLM generating a natural language query for the KG."""
        symptoms = patient_data.get("symptoms", "")
        history = patient_data.get("medical_history", "")
        query = f"Patient presents with symptoms: {symptoms}. Medical history: {history}. What diseases, associated symptoms, and treatments are relevant?"
        print(f"[LLM Agent] Generated KG query: {query}")
        return query

    def explore_kg(self, natural_language_query: str, patient_context: str) -> List[Dict]:
        """Mocks iterative KG exploration and pruning."""
        keywords = self.kg_interaction_module.query_translator(natural_language_query)
        raw_facts = self.kg_interaction_module.kg_executor(keywords)
        pruned_facts = self.kg_interaction_module.semantic_pruner(raw_facts, patient_context)
        print(f"[LLM Agent] Explored KG, retrieved {len(pruned_facts)} pruned facts.")
        return pruned_facts

    def reason_and_explain(self, patient_data: Dict, kg_facts: List[Dict]) -> Dict:
        """Mocks LLM reasoning and explanation generation based on KG facts."""
        symptoms = patient_data.get("symptoms", "")
        diagnosis_candidates = set()
        treatments = set()
        explanations = []

        explanations.append(f"Based on the patient's reported symptoms: '{symptoms}':")

        for fact in kg_facts:
            if fact["predicate"] == "is_symptom_of" and any(s.lower() in fact["subject"].lower() for s in symptoms.split(",")):
                diagnosis_candidates.add(fact["object"])
                explanations.append(f"- The symptom '{fact['subject']}' is associated with '{fact['object']}'.")
            elif fact["predicate"] == "has_treatment" and fact["subject"] in diagnosis_candidates:
                treatments.add(fact["object"])
                explanations.append(f"- '{fact['object']}' is a known treatment for '{fact['subject']}'.")
            elif fact["predicate"] == "is_symptom_of" and any(s.lower() in fact["object"].lower() for s in symptoms.split(",")):
                 # If a symptom itself is the object (less common but possible in broader KGs)
                 pass

        final_diagnosis = "Unknown"
        if diagnosis_candidates:
            # Simple heuristic: pick the first candidate or combine them
            final_diagnosis = ", ".join(list(diagnosis_candidates))
            explanations.insert(0, f"**Potential Diagnosis:** {final_diagnosis}.")
        else:
            explanations.insert(0, f"**Potential Diagnosis:** No clear diagnosis based on provided symptoms and KG data.")

        if treatments:
            explanations.append(f"\n**Recommended Treatments:** {', '.join(list(treatments))}.")
        else:
            explanations.append("\n**Recommended Treatments:** No specific treatments found in KG for identified conditions.")
        
        explanations.append("\n**Traceability (KG Facts Used):**")
        for fact in kg_facts:
            explanations.append(f"- ({fact['subject']}, {fact['predicate']}, {fact['object']})")

        reasoning_output = {
            "diagnosis": final_diagnosis,
            "treatment_recommendations": list(treatments),
            "explanation": "\n".join(explanations)
        }
        print(f"[LLM Agent] Generated reasoning and explanation for diagnosis: {final_diagnosis}")
        return reasoning_output

# --- 4. FastAPI Backend API ---

# Initialize KG Interaction and LLM Agent Modules
kg_interaction_module = KGInteractionModule(medical_knowledge_graph)
llm_agent = LLMAgent(kg_interaction_module)

app = FastAPI()

class PatientInput(BaseModel):
    symptoms: str
    medical_history: str = ""
    lab_results: str = ""

class DiagnosisOutput(BaseModel):
    diagnosis: str
    treatment_recommendations: List[str]
    explanation: str

@app.post("/diagnose", response_model=DiagnosisOutput)
async def diagnose_patient(patient_data: PatientInput):
    print("[FastAPI] Received diagnosis request.")
    
    # 1. LLM Agentic Query Generation
    llm_query = llm_agent.generate_query(patient_data.dict())
    
    # 2. Iterative KG Exploration & Pruning
    # The patient_context here is a simplified combination of symptoms and history
    patient_context_for_pruning = f"{patient_data.symptoms}, {patient_data.medical_history}"
    retrieved_facts = llm_agent.explore_kg(llm_query, patient_context_for_pruning)
    
    # 3. Faithful Reasoning & Explanation Generation
    diagnosis_result = llm_agent.reason_and_explain(patient_data.dict(), retrieved_facts)
    
    print("[FastAPI] Diagnosis complete, returning result.")
    return DiagnosisOutput(**diagnosis_result)

# --- 5. Streamlit User Interface (Frontend) ---

def streamlit_ui():
    st.set_page_config(page_title="MedGraphReason: AI Diagnostic Assistant", layout="wide")
    st.title("🩺 MedGraphReason: AI-Powered Diagnostic Assistant")
    st.markdown("--- Magnifying Medical Reasoning with Knowledge Graphs and LLM Agents ---")

    st.header("Patient Data Input")

    with st.form("patient_form"):
        symptoms = st.text_area(
            "Patient Symptoms (e.g., Fever, Cough, Headache)",
            placeholder="Enter comma-separated symptoms here...",
            height=100
        )
        medical_history = st.text_area(
            "Medical History (e.g., Asthma, Diabetes)",
            placeholder="Enter relevant medical history...",
            height=80
        )
        lab_results = st.text_area(
            "Lab Results (e.g., High WBC count, Low Hemoglobin)",
            placeholder="Enter relevant lab results... (Not fully utilized in mock)",
            height=80
        )

        submitted = st.form_submit_button("Get Diagnosis")

        if submitted:
            if not symptoms:
                st.error("Please enter at least one symptom to get a diagnosis.")
            else:
                with st.spinner("Analyzing patient data and reasoning with medical knowledge graph..."):
                    try:
                        import requests
                        # Assuming FastAPI is running on http://localhost:8000
                        response = requests.post(
                            "http://localhost:8000/diagnose",
                            json={
                                "symptoms": symptoms,
                                "medical_history": medical_history,
                                "lab_results": lab_results
                            },
                            timeout=30 # Add a timeout for the request
                        )
                        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                        diagnosis_data = response.json()

                        st.success("Diagnosis Complete!")
                        st.header("Diagnostic Insights")
                        st.subheader(f"Potential Diagnosis: {diagnosis_data['diagnosis']}")
                        
                        st.subheader("Explanation and Reasoning")
                        st.markdown(diagnosis_data['explanation'])

                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to the FastAPI backend. Make sure it's running (uvicorn medgraph_reason:app --reload --port 8000).")
                    except requests.exceptions.Timeout:
                        st.error("The request timed out. The backend might be slow or unresponsive.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"An error occurred during diagnosis: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")


# This conditional block ensures that Streamlit runs only when executed directly by `streamlit run`
# and FastAPI app is handled by `uvicorn`.
if __name__ == "__main__":
    # The uvicorn server should be run in a separate process/terminal
    # For demonstration, we just define the app here.
    # print("To run the FastAPI backend, execute in a separate terminal: uvicorn medgraph_reason:app --reload --port 8000")
    
    # If this file is run directly by python (e.g., `python medgraph_reason.py`),
    # we assume it's for Streamlit. `uvicorn` will handle running the FastAPI `app` object.
    streamlit_ui()


