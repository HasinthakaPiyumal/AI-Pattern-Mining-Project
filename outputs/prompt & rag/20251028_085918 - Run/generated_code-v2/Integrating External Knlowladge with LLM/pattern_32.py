import networkx as nx
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._populate_graph()

    def _populate_graph(self):
        # Diseases
        self.graph.add_node("Influenza", type="disease")
        self.graph.add_node("Pneumonia", type="disease")
        self.graph.add_node("Diabetes Type 2", type="disease")
        self.graph.add_node("Hypertension", type="disease")
        self.graph.add_node("Migraine", type="disease")
        self.graph.add_node("COVID-19", type="disease")

        # Symptoms
        self.graph.add_node("Fever", type="symptom")
        self.graph.add_node("Cough", type="symptom")
        self.graph.add_node("Sore Throat", type="symptom")
        self.graph.add_node("Fatigue", type="symptom")
        self.graph.add_node("Headache", type="symptom")
        self.graph.add_node("Shortness of Breath", type="symptom")
        self.graph.add_node("High Blood Sugar", type="symptom")
        self.graph.add_node("High Blood Pressure", type="symptom")
        self.graph.add_node("Muscle Aches", type="symptom")
        self.graph.add_node("Loss of Taste/Smell", type="symptom")

        # Treatments
        self.graph.add_node("Antivirals", type="treatment")
        self.graph.add_node("Antibiotics", type="treatment")
        self.graph.add_node("Insulin", type="treatment")
        self.graph.add_node("ACE Inhibitors", type="treatment")
        self.graph.add_node("Pain Relievers", type="treatment")
        self.graph.add_node("Ventilator Support", type="treatment")

        # Relations (Disease -> Symptom)
        self.graph.add_edge("Influenza", "Fever", relation="causes_symptom")
        self.graph.add_edge("Influenza", "Cough", relation="causes_symptom")
        self.graph.add_edge("Influenza", "Sore Throat", relation="causes_symptom")
        self.graph.add_edge("Influenza", "Fatigue", relation="causes_symptom")
        self.graph.add_edge("Influenza", "Muscle Aches", relation="causes_symptom")

        self.graph.add_edge("Pneumonia", "Fever", relation="causes_symptom")
        self.graph.add_edge("Pneumonia", "Cough", relation="causes_symptom")
        self.graph.add_edge("Pneumonia", "Shortness of Breath", relation="causes_symptom")
        self.graph.add_edge("Pneumonia", "Fatigue", relation="causes_symptom")

        self.graph.add_edge("Diabetes Type 2", "Fatigue", relation="causes_symptom")
        self.graph.add_edge("Diabetes Type 2", "High Blood Sugar", relation="causes_symptom")

        self.graph.add_edge("Hypertension", "High Blood Pressure", relation="causes_symptom")
        self.graph.add_edge("Hypertension", "Headache", relation="causes_symptom") # Sometimes

        self.graph.add_edge("Migraine", "Headache", relation="causes_symptom")
        self.graph.add_edge("Migraine", "Fatigue", relation="causes_symptom")

        self.graph.add_edge("COVID-19", "Fever", relation="causes_symptom")
        self.graph.add_edge("COVID-19", "Cough", relation="causes_symptom")
        self.graph.add_edge("COVID-19", "Shortness of Breath", relation="causes_symptom")
        self.graph.add_edge("COVID-19", "Fatigue", relation="causes_symptom")
        self.graph.add_edge("COVID-19", "Loss of Taste/Smell", relation="causes_symptom")
        self.graph.add_edge("COVID-19", "Muscle Aches", relation="causes_symptom")

        # Relations (Symptom -> Disease) for reverse lookup
        for u, v, data in list(self.graph.edges(data=True)):
            if data["relation"] == "causes_symptom":
                self.graph.add_edge(v, u, relation="is_symptom_of")

        # Relations (Disease -> Treatment)
        self.graph.add_edge("Influenza", "Antivirals", relation="treated_by")
        self.graph.add_edge("Pneumonia", "Antibiotics", relation="treated_by")
        self.graph.add_edge("Diabetes Type 2", "Insulin", relation="treated_by")
        self.graph.add_edge("Hypertension", "ACE Inhibitors", relation="treated_by")
        self.graph.add_edge("Migraine", "Pain Relievers", relation="treated_by")
        self.graph.add_edge("COVID-19", "Antivirals", relation="treated_by")
        self.graph.add_edge("COVID-19", "Ventilator Support", relation="treated_by")

    def get_neighbors(self, entity: str, relation_type: str = None) -> List[Dict[str, str]]:
        neighbors = []
        if entity not in self.graph:
            return neighbors

        for neighbor in self.graph.neighbors(entity):
            for _, _, data in self.graph.get_edge_data(entity, neighbor).items(): # Handle multiple edges if needed
                if relation_type is None or data.get("relation") == relation_type:
                    neighbors.append({"entity": neighbor, "relation": data.get("relation")})
        return neighbors

    def get_entity_type(self, entity: str) -> str:
        return self.graph.nodes.get(entity, {}).get("type")


class LLMAgent:
    def extract_entities(self, text: str) -> List[str]:
        # Simulate LLM extracting medical entities from patient text
        # In a real scenario, this would use an actual LLM with NER capabilities
        medical_terms = [
            "fever", "cough", "sore throat", "fatigue", "headache", 
            "shortness of breath", "high blood sugar", "high blood pressure",
            "muscle aches", "loss of taste/smell",
            "influenza", "pneumonia", "diabetes type 2", "hypertension", "migraine", "covid-19"
        ]
        extracted = [term for term in medical_terms if term in text.lower()]
        return [term.title() for term in extracted] # Capitalize for KG matching

    def evaluate_paths(self, paths: List[List[str]], patient_symptoms: List[str]) -> List[Dict[str, Any]]:
        # Simulate LLM evaluating and scoring reasoning paths
        # A more sophisticated LLM would assess medical relevance and consistency
        evaluated_paths = []
        for path in paths:
            score = 0
            explanation = []
            last_entity = path[-1]
            if last_entity in patient_symptoms: # Direct match for a symptom
                score += 10
                explanation.append(f"Path ends with a patient symptom: {last_entity}")

            # Check if path leads to a potential disease and symptoms align
            disease_candidate = None
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if kg.get_entity_type(v) == "disease" and kg.get_entity_type(u) == "symptom":
                     disease_candidate = v
                     break
                elif kg.get_entity_type(u) == "disease" and kg.get_entity_type(v) == "symptom" and v in patient_symptoms:
                    disease_candidate = u
                    score += 5 # Reward for linking disease to patient symptom
                    explanation.append(f"Path links disease {u} to patient symptom {v}")

            if kg.get_entity_type(last_entity) == "disease":
                disease_candidate = last_entity
                score += 15 # Reward for reaching a disease node
                explanation.append(f"Path reaches a disease candidate: {last_entity}")

            # Simple symptom coverage for scoring
            path_symptoms = [node for node in path if kg.get_entity_type(node) == "symptom"]
            matched_symptoms = set(path_symptoms).intersection(set(patient_symptoms))
            score += len(matched_symptoms) * 5
            if matched_symptoms:
                explanation.append(f"Path covers patient symptoms: {', '.join(matched_symptoms)}")

            evaluated_paths.append({"path": path, "score": score, "explanation": "; ".join(explanation)})
        
        # Sort by score for pruning later
        evaluated_paths.sort(key=lambda x: x["score"], reverse=True)
        return evaluated_paths

    def prune_paths(self, evaluated_paths: List[Dict[str, Any]], beam_width: int) -> List[List[str]]:
        # Simulate LLM pruning paths based on score
        return [p["path"] for p in evaluated_paths[:beam_width]]

    def reason_and_diagnose(self, final_paths: List[Dict[str, Any]], patient_data: "PatientData") -> Dict[str, Any]:
        # Simulate LLM generating a diagnosis and explanation
        if not final_paths:
            return {
                "diagnosis": "Undetermined",
                "explanation": "Could not find sufficient reasoning paths in the knowledge graph.",
                "reasoning_path": []
            }
        
        best_path_info = final_paths[0] # Assuming sorted by score
        best_path = best_path_info["path"]
        score = best_path_info["score"]

        potential_disease = "Undetermined"
        for node in reversed(best_path): # Look for the last disease in the path
            if kg.get_entity_type(node) == "disease":
                potential_disease = node
                break
        
        if potential_disease != "Undetermined":
            explanation = f"Based on the patient's symptoms (\'{', '.join(patient_data.symptoms)}\') and medical history (\'{patient_data.medical_history}\'), the most probable diagnosis is {potential_disease}. " \
                          f"This conclusion is supported by the following reasoning path from the knowledge graph: {' -> '.join(best_path)}. " \
                          f"LLM evaluation indicates a high relevance score of {score}."
        else:
             explanation = f"Based on the patient's symptoms (\'{', '.join(patient_data.symptoms)}\') and medical history (\'{patient_data.medical_history}\'), potential insights were found, but a definitive disease diagnosis could not be established directly. The best path found was: {' -> '.join(best_path)}. Score: {score}. Further medical consultation is recommended."

        return {
            "diagnosis": potential_disease,
            "explanation": explanation,
            "reasoning_path": best_path
        }


class ThinkOnGraphFramework:
    def __init__(self, llm_agent: LLMAgent, kg: MedicalKnowledgeGraph):
        self.llm_agent = llm_agent
        self.kg = kg

    def initialize(self, patient_data: "PatientData") -> List[str]:
        # Phase 1: Initialization - Extract initial topic entities
        input_text = f"Symptoms: {', '.join(patient_data.symptoms)}. Medical History: {patient_data.medical_history}. Lab Results: {patient_data.lab_results}"
        initial_entities = self.llm_agent.extract_entities(input_text)
        print(f"[ToG-Init] Initial entities extracted: {initial_entities}")
        return initial_entities

    def explore(self, initial_entities: List[str], patient_data: "PatientData", beam_width: int, max_exploration_depth: int) -> List[List[str]]:
        # Phase 2: Exploration - Iterative Beam Search
        current_top_paths = [[entity] for entity in initial_entities if self.kg.graph.has_node(entity)]
        
        if not current_top_paths: # Handle case where initial entities are not in KG
            return []

        for depth in range(max_exploration_depth):
            print(f"[ToG-Expl] --- Depth {depth + 1} --- Current paths: {current_top_paths}")
            new_candidate_paths = []

            # Step 1: Search - Extend paths
            for path in current_top_paths:
                last_node = path[-1]
                neighbors = self.kg.get_neighbors(last_node)
                for neighbor_info in neighbors:
                    neighbor_entity = neighbor_info["entity"]
                    # Avoid cycles for simplicity, especially for short paths
                    if neighbor_entity not in path or len(path) < 2:
                        new_candidate_paths.append(path + [neighbor_entity])
            
            if not new_candidate_paths:
                break # No new paths to explore

            # Step 2: Prune - Evaluate and select top-N paths
            evaluated_candidates = self.llm_agent.evaluate_paths(new_candidate_paths, patient_data.symptoms)
            current_top_paths = self.llm_agent.prune_paths(evaluated_candidates, beam_width)
            
            if not current_top_paths:
                break

            # Check for sufficiency (simplified: if a disease is found in top paths)
            sufficient = False
            for path in current_top_paths:
                if any(self.kg.get_entity_type(node) == "disease" for node in path):
                    sufficient = True
                    break
            if sufficient and depth >= 1: # Require at least 2 hops to consider sufficient for reasoning
                print(f"[ToG-Expl] Sufficient paths found at depth {depth + 1}. Proceeding to Reasoning.")
                break

        print(f"[ToG-Expl] Final paths after exploration: {current_top_paths}")
        return current_top_paths

    def reason(self, final_paths: List[List[str]], patient_data: "PatientData") -> Dict[str, Any]:
        # Phase 3: Reasoning - Evaluate and generate answer
        # Re-evaluate final paths to get scores for the best reasoning
        if not final_paths:
             return {
                "diagnosis": "Undetermined",
                "explanation": "No reasoning paths were found during exploration.",
                "reasoning_path": []
            }

        evaluated_final_paths = self.llm_agent.evaluate_paths(final_paths, patient_data.symptoms)
        diagnosis_result = self.llm_agent.reason_and_diagnose(evaluated_final_paths, patient_data)
        print(f"[ToG-Reason] Diagnosis Result: {diagnosis_result}")
        return diagnosis_result

    def diagnose(self, patient_data: "PatientData", beam_width: int = 3, max_depth: int = 3) -> Dict[str, Any]:
        print("\n--- Starting ToG Diagnosis Process ---")
        initial_entities = self.initialize(patient_data)
        
        if not initial_entities:
            return {
                "diagnosis": "Undetermined",
                "explanation": "No relevant initial entities extracted from patient data.",
                "reasoning_path": []
            }

        final_paths = self.explore(initial_entities, patient_data, beam_width, max_depth)
        
        if not final_paths:
            return {
                "diagnosis": "Undetermined",
                "explanation": "Exploration did not yield any meaningful reasoning paths.",
                "reasoning_path": []
            }

        result = self.reason(final_paths, patient_data)
        print("--- ToG Diagnosis Process Completed ---\n")
        return result


# FastAPI Models
class PatientData(BaseModel):
    symptoms: List[str]
    medical_history: str = ""
    lab_results: str = ""


# Initialize KG and LLM Agent
kg = MedicalKnowledgeGraph()
llm_agent = LLMAgent()
tog_framework = ThinkOnGraphFramework(llm_agent, kg)


@app.post("/diagnose", response_model=Dict[str, Any])
def get_diagnosis(patient_data: PatientData):
    """Diagnose a patient based on symptoms, medical history, and lab results using the ToG framework."""
    diagnosis_result = tog_framework.diagnose(patient_data)
    return diagnosis_result

