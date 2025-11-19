import networkx as nx
from transformers import pipeline
import random

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_medical_entity(self, entity_id, entity_type, attributes=None):
        if attributes is None:
            attributes = {}
        self.graph.add_node(entity_id, type=entity_type, **attributes)

    def add_medical_relationship(self, source_id, target_id, relation_type, attributes=None):
        if attributes is None:
            attributes = {}
        self.graph.add_edge(source_id, target_id, type=relation_type, **attributes)

    def get_related_information(self, entity_id, relation_type=None):
        related_info = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor)
            if relation_type is None or edge_data.get("type") == relation_type:
                related_info.append((neighbor, edge_data, self.graph.nodes[neighbor]))
        return related_info

class ClinicalDecisionSupportSystem:
    def __init__(self, kg, llm_model_name="distilbert-base-uncased-distilled-squad"):
        self.kg = kg
        self.qa_pipeline = pipeline("question-answering", model=llm_model_name)

    def query_system(self, patient_symptoms, patient_history):
        # Step 1: Retrieve relevant medical knowledge from KG
        relevant_kg_info = []
        for symptom in patient_symptoms:
            # Simulate retrieval based on symptoms (simplified)
            nodes = [n for n, data in self.kg.graph.nodes(data=True) if symptom.lower() in str(n).lower() or symptom.lower() in str(data.get("description", "")).lower()]
            for node in nodes:
                relevant_kg_info.extend(self.kg.get_related_information(node))
                relevant_kg_info.append((node, self.kg.graph.nodes[node]))

        # Step 2: Combine retrieved info with patient history
        context = f"Patient History: {patient_history}. "
        context += "Relevant Medical Knowledge: "
        for item in relevant_kg_info:
            if isinstance(item[1], dict) and "type" in item[1]: # This is a node
                context += f"Medical Condition: {item[0]}, Description: {item[1].get("description", "N/A")}. "
            elif len(item) == 3: # This is a relationship
                context += f"Relationship: {item[0]} {item[1].get("type", "related to")} {item[2].get("id", "N/A")}. "

        # Step 3: Reason using LLM
        question = f"Given the patient\\'s symptoms \\'{'\\, \\'.join(patient_symptoms)}\\' and their history, what are potential diagnoses and treatment plans? Explain your reasoning."
        llm_response = self.qa_pipeline(question=question, context=context)

        diagnosis = llm_response["answer"] if llm_response["score"] > 0.5 else "Unable to confidently determine a diagnosis based on available information."
        explanation = f"Confidence Score: {llm_response["score"]:.2f}. " + (llm_response["answer"] if llm_response["score"] > 0.5 else "")
        
        # Simulate a more elaborate treatment plan based on diagnosis
        if "Fever" in patient_symptoms and "Headache" in patient_symptoms and "Flu" in diagnosis:
            treatment_plan = "Suggest rest, hydration, and over-the-counter fever reducers. Consider antiviral medication if caught early."
        else:
            treatment_plan = "Further investigation and specialist consultation recommended."

        return {"diagnosis": diagnosis, "treatment_plan": treatment_plan, "reasoning": explanation, "llm_raw_output": llm_response}

# Example Usage
if __name__ == "__main__":
    # Initialize Knowledge Graph
    medical_kg = KnowledgeGraph()
    medical_kg.add_medical_entity("Flu", "condition", {"description": "Influenza is a viral infection that attacks your respiratory system.", "symptoms": ["fever", "cough", "sore throat", "muscle aches", "fatigue"]})
    medical_kg.add_medical_entity("Common Cold", "condition", {"description": "A viral infection of your nose and throat.", "symptoms": ["runny nose", "sore throat", "cough", "sneezing"]})
    medical_kg.add_medical_entity("Paracetamol", "medication", {"dosage": "500mg", "purpose": "fever reducer, pain relief"})
    medical_kg.add_medical_entity("Rest", "treatment", {"description": "Important for recovery from viral infections."})
    medical_kg.add_medical_relationship("Flu", "Paracetamol", "treated_by")
    medical_kg.add_medical_relationship("Flu", "Rest", "treated_by")
    medical_kg.add_medical_relationship("Common Cold", "Rest", "treated_by")

    # Initialize Clinical Decision Support System
    cds_system = ClinicalDecisionSupportSystem(medical_kg)

    # Patient Query 1
    patient_symptoms_1 = ["fever", "cough", "fatigue"]
    patient_history_1 = "Patient has a history of seasonal allergies."
    recommendations_1 = cds_system.query_system(patient_symptoms_1, patient_history_1)
    print("\n--- Patient Query 1 ---")
    print(f"Symptoms: {patient_symptoms_1}")
    print(f"History: {patient_history_1}")
    print(f"Diagnosis: {recommendations_1["diagnosis"]}")
    print(f"Treatment Plan: {recommendations_1["treatment_plan"]}")
    print(f"Reasoning: {recommendations_1["reasoning"]}")

    # Patient Query 2
    patient_symptoms_2 = ["runny nose", "sneezing"]
    patient_history_2 = "No significant medical history."
    recommendations_2 = cds_system.query_system(patient_symptoms_2, patient_history_2)
    print("\n--- Patient Query 2 ---")
    print(f"Symptoms: {patient_symptoms_2}")
    print(f"History: {patient_history_2}")
    print(f"Diagnosis: {recommendations_2["diagnosis"]}")
    print(f"Treatment Plan: {recommendations_2["treatment_plan"]}")
    print(f"Reasoning: {recommendations_2["reasoning"]}")
