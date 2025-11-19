from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import networkx as nx

# --- 1. Data Ingestion and KG Construction (Simulated) ---
class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        # Simulate some medical knowledge
        self.add_triple("PatientX", "has_symptom", "fever")
        self.add_triple("PatientX", "has_symptom", "cough")
        self.add_triple("fever", "is_symptom_of", "influenza")
        self.add_triple("cough", "is_symptom_of", "influenza")
        self.add_triple("influenza", "treated_by", "oseltamivir")
        self.add_triple("oseltamivir", "is_drug_class", "antiviral")
        self.add_triple("fever", "is_symptom_of", "common_cold")
        self.add_triple("common_cold", "treated_by", "rest_fluids")
        self.add_triple("PatientX", "has_symptom", "headache")
        self.add_triple("headache", "is_symptom_of", "migraine")
        self.add_triple("migraine", "treated_by", "sumatriptan")
        self.add_triple("sumatriptan", "is_drug_class", "triptan")

    def add_triple(self, subject, predicate, obj):
        self.graph.add_edge(subject, obj, relation=predicate)

    def get_neighbors(self, entity, relation_type=None, depth=1):
        results = []
        if entity not in self.graph:
            return results

        for neighbor in nx.neighbors(self.graph, entity):
            for u, v, data in self.graph.edges(entity, neighbor, data=True):
                if relation_type is None or data['relation'] == relation_type:
                    results.append((u, data['relation'], v))
        return results

    def find_paths(self, start_entity, end_entity, max_length=3):
        paths = []
        if start_entity not in self.graph or end_entity not in self.graph:
            return []
        for path in nx.all_simple_paths(self.graph, source=start_entity, target=end_entity, cutoff=max_length):
            triples = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                for edge_data in self.graph.get_edge_data(u, v).values():
                    triples.append((u, edge_data['relation'], v))
            paths.append(triples)
        return paths

# --- 2. LLM Integration and Fine-tuning (Simulated) ---
class LLMInterface:
    def __init__(self):
        pass # In a real system, initialize an LLM model here (e.g., from transformers)

    def extract_entities(self, text: str) -> List[str]:
        # Simulate entity extraction using simple keyword matching for demonstration
        known_entities = ["fever", "cough", "headache", "influenza", "common_cold", "migraine", "oseltamivir", "sumatriptan", "rest_fluids"]
        extracted = [entity for entity in known_entities if entity in text.lower()]
        return extracted

    def semantic_parse(self, natural_language_query: str, entities: List[str]) -> Dict[str, Any]:
        # Simulate semantic parsing to generate a KG query structure
        # In a real system, this would involve complex LLM prompting to generate Cypher/SPARQL
        query_type = "diagnosis_recommendation"
        if "symptoms" in natural_language_query.lower() or "diagnose" in natural_language_query.lower():
            query_type = "diagnosis"
        elif "treatment" in natural_language_query.lower() or "recommend" in natural_language_query.lower():
            query_type = "treatment"

        return {"query_type": query_type, "entities": entities}

    def reason(self, patient_data: str, kg_facts: List[str]) -> Dict[str, str]:
        # Simulate LLM reasoning based on patient data and KG facts
        # This would be the core LLM call, incorporating KDCoT and RoG patterns
        reasoning_steps = []
        diagnosis = "Uncertain"
        treatment = "Consult a physician"

        patient_lower = patient_data.lower()

        # Simple rule-based reasoning for demonstration purposes
        if "fever" in patient_lower and "cough" in patient_lower:
            if any("influenza" in fact.lower() for fact in kg_facts):
                diagnosis = "Influenza"
                reasoning_steps.append("Patient exhibits fever and cough, symptoms associated with influenza according to KG.")
                if any("oseltamivir" in fact.lower() for fact in kg_facts):
                    treatment = "Oseltamivir and supportive care"
                    reasoning_steps.append("Oseltamivir is a known treatment for influenza from KG.")
                else:
                    treatment = "Supportive care (rest, fluids)"
            elif any("common_cold" in fact.lower() for fact in kg_facts):
                diagnosis = "Common Cold"
                reasoning_steps.append("Patient exhibits fever and cough, symptoms associated with common cold according to KG.")
                treatment = "Supportive care (rest, fluids)"
                reasoning_steps.append("Rest and fluids are common treatments for common cold from KG.")

        if "headache" in patient_lower and "migraine" in patient_lower:
            diagnosis = "Migraine"
            reasoning_steps.append("Patient reports headache, diagnosed as migraine.")
            if any("sumatriptan" in fact.lower() for fact in kg_facts):
                treatment = "Sumatriptan"
                reasoning_steps.append("Sumatriptan is a known treatment for migraine from KG.")

        if not reasoning_steps:
            reasoning_steps.append("Unable to find specific reasoning path with provided facts. More information or broader KG exploration might be needed.")

        return {
            "diagnosis": diagnosis,
            "treatment_recommendation": treatment,
            "explanation": "\n".join(reasoning_steps)
        }

# --- 3. Reasoning and Recommendation Engine ---
class ReasoningEngine:
    def __init__(self, kg: MedicalKnowledgeGraph, llm: LLMInterface):
        self.kg = kg
        self.llm = llm

    def _triple_to_string(self, triple):
        return f"({triple[0]} -[{triple[1]}]-> {triple[2]})"

    def diagnose_and_recommend(self, patient_symptoms: str, patient_history: str = "") -> Dict[str, Any]:
        full_query = f"Symptoms: {patient_symptoms}. History: {patient_history}"

        # LLM-based Topic Entity Extraction
        extracted_entities = self.llm.extract_entities(full_query)

        # Semantic Parsing for KGQA (simulated for initial query construction)
        parsed_query = self.llm.semantic_parse(full_query, extracted_entities)

        kg_retrieved_facts = []
        kg_exploration_paths = []

        # RAG for KGs & LLM-Guided Beam Search (Simulated)
        # Iterative Prompting for Guided KG Exploration
        for entity in extracted_entities:
            # Retrieve direct neighbors
            neighbors = self.kg.get_neighbors(entity, depth=1)
            for triple in neighbors:
                kg_retrieved_facts.append(self._triple_to_string(triple))

            # Find paths to potential diseases/treatments (simulated beam search)
            # In a real system, LLM would guide target selection for beam search
            potential_targets = ["influenza", "common_cold", "migraine", "oseltamivir", "sumatriptan", "rest_fluids"]
            for target in potential_targets:
                paths = self.kg.find_paths(entity, target, max_length=2)
                for path_triples in paths:
                    kg_exploration_paths.append(", ".join([self._triple_to_string(t) for t in path_triples]))

        # Hybrid Pruning Strategy (simulated: only keep unique facts/paths)
        unique_kg_facts = list(set(kg_retrieved_facts + kg_exploration_paths))

        # Reasoning on Graphs (RoG) and Knowledge-Driven Chain-of-Thought (KDCoT)
        # LLM reason function will take all relevant facts.
        reasoning_output = self.llm.reason(full_query, unique_kg_facts)

        return {
            "patient_query": full_query,
            "extracted_entities": extracted_entities,
            "kg_facts_used": unique_kg_facts,
            "diagnosis": reasoning_output["diagnosis"],
            "treatment_recommendation": reasoning_output["treatment_recommendation"],
            "explanation": reasoning_output["explanation"]
        }

# --- 4. User Interface and API Layer (FastAPI) ---
app = FastAPI(
    title="Medical Diagnostic & Treatment Recommendation System",
    description="Leveraging LLMs and Knowledge Graphs for accurate clinical insights."
)

# Initialize components
medical_kg = MedicalKnowledgeGraph()
llm_interface = LLMInterface()
reasoning_engine = ReasoningEngine(kg=medical_kg, llm=llm_interface)

class PatientInput(BaseModel):
    symptoms: str
    history: str = ""

class MedicalRecommendation(BaseModel):
    patient_query: str
    extracted_entities: List[str]
    kg_facts_used: List[str]
    diagnosis: str
    treatment_recommendation: str
    explanation: str

@app.post("/diagnose", response_model=MedicalRecommendation, summary="Get medical diagnosis and treatment recommendation")
async def diagnose(patient_input: PatientInput):
    try:
        recommendation = reasoning_engine.diagnose_and_recommend(
            patient_symptoms=patient_input.symptoms,
            patient_history=patient_input.history
        )
        return MedicalRecommendation(**recommendation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. Deployment and Monitoring (Notes) ---
# For deployment, this FastAPI app would be containerized with Docker.
# Kubernetes would manage orchestration.
# Monitoring tools like Prometheus/Grafana would track API performance.
# This code focuses on the core logic; deployment infrastructure is external.

# To run this application:
# 1. Save the code as main.py
# 2. Install dependencies: pip install fastapi uvicorn pydantic networkx
# 3. Run: uvicorn main:app --reload
# 4. Access the API at http://127.0.0.1:8000/docs for Swagger UI.