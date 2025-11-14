
import streamlit as st
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import networkx as nx
import json
from typing import List, Dict, Any

# --- 1. Medical Knowledge Graph (MKG) Implementation (using NetworkX for prototyping) ---
class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._populate_sample_data()

    def _populate_sample_data(self):
        # Diseases
        self.add_medical_entity("Type 2 Diabetes", "Disease", description="A chronic condition that affects the way the body processes blood sugar (glucose).")
        self.add_medical_entity("Hypertension", "Disease", description="High blood pressure.")
        self.add_medical_entity("Renal Impairment", "Condition", description="Reduced kidney function.")
        self.add_medical_entity("COVID-19", "Disease", description="A respiratory illness caused by the SARS-CoV-2 virus.")

        # Symptoms
        self.add_medical_entity("Frequent Urination", "Symptom")
        self.add_medical_entity("Increased Thirst", "Symptom")
        self.add_medical_entity("Blurred Vision", "Symptom")
        self.add_medical_entity("Headache", "Symptom")
        self.add_medical_entity("Fatigue", "Symptom")
        self.add_medical_entity("Fever", "Symptom")
        self.add_medical_entity("Cough", "Symptom")

        # Medications
        self.add_medical_entity("Metformin", "Drug", description="Used to treat type 2 diabetes.")
        self.add_medical_entity("Lisinopril", "Drug", description="Used to treat high blood pressure.")
        self.add_medical_entity("Insulin", "Drug", description="Hormone that helps control blood sugar levels.")
        self.add_medical_entity("Remdesivir", "Drug", description="Antiviral medication for COVID-19.")

        # Relationships
        self.add_relationship("Frequent Urination", "indicates", "Type 2 Diabetes")
        self.add_relationship("Increased Thirst", "indicates", "Type 2 Diabetes")
        self.add_relationship("Blurred Vision", "indicates", "Type 2 Diabetes")
        self.add_relationship("Headache", "symptom_of", "Hypertension")
        self.add_relationship("Fatigue", "symptom_of", "Hypertension")
        self.add_relationship("Fever", "symptom_of", "COVID-19")
        self.add_relationship("Cough", "symptom_of", "COVID-19")

        self.add_relationship("Metformin", "treats", "Type 2 Diabetes")
        self.add_relationship("Lisinopril", "treats", "Hypertension")
        self.add_relationship("Insulin", "treats", "Type 2 Diabetes")
        self.add_relationship("Remdesivir", "treats", "COVID-19")

        self.add_relationship("Metformin", "contraindicated_in", "Renal Impairment")
        self.add_relationship("Lisinopril", "contraindicated_in", "Pregnancy") # Example contraindication

        self.add_relationship("Type 2 Diabetes", "complication_of", "Obesity") # Example

    def add_medical_entity(self, name: str, entity_type: str, description: str = ""):
        if not self.graph.has_node(name):
            self.graph.add_node(name, type=entity_type, description=description)

    def add_relationship(self, source: str, relation: str, target: str, properties: dict = None):
        if not self.graph.has_node(source):
            self.add_medical_entity(source, "Unknown") # Add as unknown if not exists
        if not self.graph.has_node(target):
            self.add_medical_entity(target, "Unknown") # Add as unknown if not exists

        if properties is None:
            properties = {}
        self.graph.add_edge(source, target, relation=relation, **properties)

    def get_related_entities(self, entity_name: str, relation: str = None) -> List[Dict[str, str]]:
        results = []
        if entity_name in self.graph:
            for neighbor in self.graph.neighbors(entity_name):
                edge_data = self.graph.get_edge_data(entity_name, neighbor)
                if relation is None or edge_data.get("relation") == relation:
                    results.append({"entity": neighbor, "type": self.graph.nodes[neighbor].get("type", "Unknown"), "relation": edge_data.get("relation")})
            for pre_neighbor in self.graph.predecessors(entity_name):
                edge_data = self.graph.get_edge_data(pre_neighbor, entity_name)
                if relation is None or edge_data.get("relation") == relation:
                    results.append({"entity": pre_neighbor, "type": self.graph.nodes[pre_neighbor].get("type", "Unknown"), "relation": edge_data.get("relation")})
        return results

    def find_paths(self, start_entity: str, end_entity: str, max_length: int = 3) -> List[List[Dict[str, str]]]:
        paths_found = []
        if start_entity not in self.graph or end_entity not in self.graph:
            return []

        for path in nx.all_simple_paths(self.graph, source=start_entity, target=end_entity, cutoff=max_length):
            formatted_path = []
            for i in range(len(path) - 1):
                source_node = path[i]
                target_node = path[i+1]
                edge_data = self.graph.get_edge_data(source_node, target_node)
                relation = edge_data.get("relation", "unknown_relation") if edge_data else "unknown_relation"
                formatted_path.append({"source": source_node, "relation": relation, "target": target_node})
            paths_found.append(formatted_path)
        return paths_found

    def get_entity_description(self, entity_name: str) -> str:
        if entity_name in self.graph:
            return self.graph.nodes[entity_name].get("description", "No description available.")
        return "Entity not found."

# Initialize the MKG globally for the FastAPI app
mkg = MedicalKnowledgeGraph()

# --- 2. Mock Large Language Model (LLM) Implementation ---
class MockLLM:
    def __init__(self, mkg_instance: MedicalKnowledgeGraph):
        self.mkg = mkg_instance

    def _semantic_parse_query(self, natural_language_query: str) -> Dict[str, str]:
        # A very basic mock semantic parser. In a real system, this would use an actual NLP model.
        query_lower = natural_language_query.lower()
        if "contraindications for metformin" in query_lower and "renal impairment" in query_lower:
            return {"action": "find_contraindications", "drug": "Metformin", "condition": "Renal Impairment"}
        elif "treats hypertension" in query_lower:
            return {"action": "find_treatments", "disease": "Hypertension"}
        elif "symptoms of" in query_lower:
            entity = natural_language_query.split("symptoms of")[-1].strip().replace("?", "").title()
            return {"action": "find_symptoms", "disease": entity}
        elif "related to" in query_lower:
            entity = natural_language_query.split("related to")[-1].strip().replace("?", "").title()
            return {"action": "find_related", "entity": entity}
        return {"action": "unknown"}

    def _extract_topic_entities(self, text: str) -> List[str]:
        # A very basic mock entity extractor
        entities = []
        medical_terms = ["Type 2 Diabetes", "Hypertension", "Metformin", "Renal Impairment", "COVID-19",
                         "Frequent Urination", "Headache", "Fever", "Cough", "Lisinopril"]
        for term in medical_terms:
            if term.lower() in text.lower():
                entities.append(term)
        return list(set(entities)) # Return unique entities

    def reason_on_graph(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Simulate LLM as an agent doing multi-step reasoning on the KG
        st.write("--- LLM Agent Reasoning --- (Mock)")
        symptoms = patient_data.get("symptoms", [])
        history = patient_data.get("medical_history", [])
        lab_results = patient_data.get("lab_results", {})

        potential_diagnoses = {}
        for symptom in symptoms:
            related_to_symptom = self.mkg.get_related_entities(symptom, relation="indicates")
            for item in related_to_symptom:
                if item["type"] == "Disease":
                    potential_diagnoses[item["entity"]] = potential_diagnoses.get(item["entity"], 0) + 1

        ranked_diagnoses = sorted(potential_diagnoses.items(), key=lambda item: item[1], reverse=True)
        st.write(f"Initial potential diagnoses based on symptoms: {ranked_diagnoses}")

        # Further reasoning (mock: check for contraindications for common treatments)
        recommendations = []
        for diag, _ in ranked_diagnoses:
            treatments = self.mkg.get_related_entities(diag, relation="treated_by") # Placeholder relation
            for t in treatments:
                drug = t["entity"]
                contraindications = self.mkg.get_related_entities(drug, relation="contraindicated_in")
                is_contraindicated = False
                for contra in contraindications:
                    if contra["entity"] in history or (contra["entity"] == "Renal Impairment" and lab_results.get("creatinine", 0) > 1.2): # Mock lab result check
                        is_contraindicated = True
                        recommendations.append({"diagnosis": diag, "treatment": drug, "status": "Contraindicated", "reason": f"Patient has {contra['entity']}"})
                        break
                if not is_contraindicated:
                    recommendations.append({"diagnosis": diag, "treatment": drug, "status": "Recommended"})
            if not treatments:
                 recommendations.append({"diagnosis": diag, "treatment": "No specific treatment found in KG (mock)", "status": "Investigation Needed"})

        return recommendations

    def generate_augmented_response(self, query: str, kg_facts: List[Dict[str, Any]]) -> str:
        # Simulate RAG and KDCoT
        st.write("--- LLM Response Generation (RAG/KDCoT) --- (Mock)")
        response_parts = []
        response_parts.append(f"Based on your query: '{query}', and leveraging our Medical Knowledge Graph:")

        if not kg_facts:
            response_parts.append("No direct relevant facts found in the knowledge graph.")
        else:
            response_parts.append("Relevant facts from the KG:")
            for fact in kg_facts:
                if "source" in fact and "relation" in fact and "target" in fact:
                    response_parts.append(f"- {fact['source']} {fact['relation'].replace('_', ' ')} {fact['target']}.")
                elif "entity" in fact and "relation" in fact and "type" in fact:
                    response_parts.append(f"- {query.split('of')[-1].strip().replace('?', '')} {fact['relation'].replace('_', ' ')} {fact['entity']} (Type: {fact['type']}).")
                else:
                    response_parts.append(f"- {fact}") # Fallback for unexpected format

            # Simulate KDCoT by adding an explanation
            response_parts.append("\nTherefore, based on this knowledge, here is a reasoned insight:")
            if "find_contraindications" in query and any(f.get("relation") == "contraindicated_in" for f in kg_facts):
                response_parts.append("It appears that there are specific contraindications to consider based on the patient's conditions and the medication in question.")
            elif any(f.get("relation") in ["indicates", "symptom_of"] for f in kg_facts):
                response_parts.append("The retrieved symptoms strongly suggest certain medical conditions.")
            elif any(f.get("relation") == "treats" for f in kg_facts):
                response_parts.append("The knowledge graph indicates potential treatment options for the identified condition.")
            else:
                response_parts.append("Further analysis might be required to provide a definitive answer given the facts.")

        return "\n".join(response_parts)

mock_llm = MockLLM(mkg)

# --- 3. FastAPI Backend --- (Runs as a separate process)
app = FastAPI(
    title="MedAdvisor AI API",
    description="API for MedAdvisor AI, integrating LLMs with a Medical Knowledge Graph.",
    version="1.0.0",
)

class PatientData(BaseModel):
    symptoms: List[str]
    medical_history: List[str]
    lab_results: Dict[str, float] = {}
    natural_language_query: str = ""

class KGQuery(BaseModel):
    query: str

@app.post("/diagnose")
async def diagnose_patient(patient_data: PatientData):
    st.write("Received diagnosis request at FastAPI") # This won't show in FastAPI console, but for debugging if run together
    # LLM as Agent & Structured Reasoning
    recommendations = mock_llm.reason_on_graph(patient_data.dict())

    # Further augmentation if a query is provided
    final_response = {"diagnoses_and_treatments": recommendations}
    if patient_data.natural_language_query:
        parsed_query = mock_llm._semantic_parse_query(patient_data.natural_language_query)
        kg_facts = []
        if parsed_query.get("action") == "find_contraindications":
            facts = mkg.get_related_entities(parsed_query["drug"], "contraindicated_in")
            for fact in facts:
                kg_facts.append({"source": parsed_query["drug"], "relation": "contraindicated_in", "target": fact["entity"]})
        elif parsed_query.get("action") == "find_treatments":
            facts = mkg.get_related_entities(parsed_query["disease"], "treats")
            for fact in facts:
                kg_facts.append({"source": fact["entity"], "relation": "treats", "target": parsed_query["disease"]})
        elif parsed_query.get("action") == "find_symptoms":
            facts = mkg.get_related_entities(parsed_query["disease"], "indicates") # or symptom_of
            for fact in facts:
                kg_facts.append({"source": fact["entity"], "relation": "indicates", "target": parsed_query["disease"]})
        elif parsed_query.get("action") == "find_related":
             facts = mkg.get_related_entities(parsed_query["entity"])
             for fact in facts:
                kg_facts.append({"source": fact["entity"], "relation": fact["relation"], "target": patient_data.natural_language_query.split('related to')[-1].strip().replace('?', '').title()})


        augmented_explanation = mock_llm.generate_augmented_response(patient_data.natural_language_query, kg_facts)
        final_response["augmented_explanation"] = augmented_explanation

    return final_response

@app.post("/query_kg_direct")
async def query_kg_direct(kg_query: KGQuery):
    st.write(f"Received direct KG query: {kg_query.query}")
    # Mock entity extraction and semantic parsing for direct KG query
    entities = mock_llm._extract_topic_entities(kg_query.query)
    results = []
    if entities:
        for entity in entities:
            related = mkg.get_related_entities(entity)
            description = mkg.get_entity_description(entity)
            results.append({"entity": entity, "description": description, "related_info": related})
    else:
        results.append({"message": "No specific medical entities recognized in query for direct KG lookup."})
    return {"kg_results": results}

# --- 4. Streamlit User Interface (Client) ---
def run_streamlit_ui():
    st.set_page_config(page_title="MedAdvisor AI", layout="wide")
    st.title("🧠 MedAdvisor AI: Intelligent Healthcare Assistant")
    st.markdown("---\n_Leveraging LLMs and Knowledge Graphs for enhanced medical reasoning._")

    st.sidebar.header("Patient Information")
    symptoms_input = st.sidebar.text_area("Enter Patient Symptoms (comma-separated):", "Frequent Urination, Increased Thirst")
    history_input = st.sidebar.text_area("Enter Medical History/Conditions (comma-separated):", "Renal Impairment")
    lab_results_input = st.sidebar.text_area("Enter Lab Results (JSON format, e.g., {\"creatinine\": 1.5}):", "{\"creatinine\": 1.5}")

    try:
        symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
        history = [h.strip() for h in history_input.split(",") if h.strip()]
        lab_results = json.loads(lab_results_input) if lab_results_input else {}
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON for Lab Results. Please use valid JSON format.")
        return

    st.sidebar.header("Natural Language Query")
    nl_query = st.sidebar.text_area("Ask MedAdvisor AI a question (e.g., 'What are the contraindications for Metformin with renal impairment?'):", "What are the contraindications for Metformin with renal impairment?")

    st.header("AI Diagnosis & Recommendations")
    if st.sidebar.button("Get Diagnosis & Insights"):
        patient_data_payload = {
            "symptoms": symptoms,
            "medical_history": history,
            "lab_results": lab_results,
            "natural_language_query": nl_query
        }

        st.info("Sending request to MedAdvisor AI backend...")
        try:
            import requests
            response = requests.post("http://localhost:8000/diagnose", json=patient_data_payload)
            if response.status_code == 200:
                result = response.json()
                st.subheader("Potential Diagnoses & Treatment Suggestions:")
                for item in result.get("diagnoses_and_treatments", []):
                    st.markdown(f"- **Diagnosis:** {item.get('diagnosis')}, **Treatment:** {item.get('treatment')}, **Status:** {item.get('status')}")
                    if item.get('reason'):
                        st.caption(f"  _Reason:_ {item.get('reason')}")
                
                if result.get("augmented_explanation"):
                    st.subheader("LLM Augmented Explanation (RAG & KDCoT):")
                    st.write(result["augmented_explanation"])

            else:
                st.error(f"Error from backend: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the MedAdvisor AI backend. Please ensure the FastAPI server is running at http://localhost:8000.")

    st.header("Direct Knowledge Graph Query")
    direct_kg_query_input = st.text_input("Enter a direct query for the Knowledge Graph (e.g., 'symptoms of Type 2 Diabetes'):", "symptoms of Type 2 Diabetes")

    if st.button("Query KG Directly"):
        kg_query_payload = {"query": direct_kg_query_input}
        st.info("Sending direct KG query to MedAdvisor AI backend...")
        try:
            import requests
            response = requests.post("http://localhost:8000/query_kg_direct", json=kg_query_payload)
            if response.status_code == 200:
                result = response.json()
                st.subheader("Direct KG Query Results:")
                for item in result.get("kg_results", [])  :
                    st.markdown(f"**Entity:** {item.get('entity', 'N/A')}")
                    st.write(f"**Description:** {item.get('description', 'N/A')}")
                    st.write("**Related Information:**")
                    if item.get('related_info'):
                        for rel in item['related_info']:
                            st.markdown(f"  - {rel.get('entity')} ({rel.get('type')}) - {rel.get('relation')}")
                    else:
                        st.write("  No direct relations found.")

            else:
                st.error(f"Error from backend: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the MedAdvisor AI backend. Please ensure the FastAPI server is running at http://localhost:8000.")

# --- Instructions to Run ---
if __name__ == "__main__":
    st.markdown("""
    ## How to Run This Application:

    1.  **Save this code** as `medadvisor_ai.py`.
    2.  **Install dependencies** (if you haven't already):
        `pip install streamlit uvicorn fastapi "python-multipart" networkx pydantic requests`
    3.  **Run the FastAPI backend** in your terminal:
        `uvicorn medadvisor_ai:app --reload`
    4.  **Open a new terminal** and **run the Streamlit frontend**:
        `streamlit run medadvisor_ai.py`

    _Ensure both processes are running simultaneously for the application to function._
    """)
    run_streamlit_ui()


