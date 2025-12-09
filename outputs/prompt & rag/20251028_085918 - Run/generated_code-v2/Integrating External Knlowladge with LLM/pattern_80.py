import networkx as nx
import random

def simulate_llm_reasoning(query, context):
    if "initial symptoms" in query:
        return f"LLM Initial Hypothesis for '{context}': Consider conditions like 'Common Cold', 'Flu', 'Allergies'. Focus exploration on related symptoms and recent patient history."
    elif "complex evaluation" in query:
        return f"LLM Complex Evaluation for '{context}': Based on additional data, confirming presence of 'Fever' and 'Body Aches' points strongly to 'Flu'."
    elif "final diagnosis" in query:
        return f"LLM Final Diagnosis for '{context}': Patient likely has 'Flu'. Recommended treatment: rest, fluids, antiviral medication (if within 48 hours of symptom onset)."
    return f"LLM Reasoning for '{query}' with context '{context}'."

def simulate_lightweight_pruning(candidates, query_context):
    relevant_candidates = []
    for candidate in candidates:
        if "symptom" in query_context and "treatment" in candidate.lower():
            continue
        if "flu" in query_context.lower() and "allergies" in candidate.lower() and random.random() < 0.7:
            continue
        relevant_candidates.append(candidate)
    return relevant_candidates[:min(len(relevant_candidates), 3)] # Limit to top 3 for simulation

class MedicalDiagnosticAssistant:
    def __init__(self):
        self.kg = nx.DiGraph()
        self._build_knowledge_graph()

    def _build_knowledge_graph(self):
        symptoms = ["Cough", "Fever", "Sore Throat", "Headache", "Body Aches", "Runny Nose", "Fatigue", "Rash", "Nausea"]
        conditions = ["Common Cold", "Flu", "Allergies", "Strep Throat", "Measles", "Gastroenteritis"]
        treatments = ["Rest", "Fluids", "Pain Relievers", "Antihistamines", "Antibiotics", "Antiviral Medication", "Vaccination"]

        self.kg.add_nodes_from(symptoms, type="symptom")
        self.kg.add_nodes_from(conditions, type="condition")
        self.kg.add_nodes_from(treatments, type="treatment")

        # Symptom-Condition relations
        self.kg.add_edge("Cough", "Common Cold", relation="has_symptom")
        self.kg.add_edge("Fever", "Common Cold", relation="has_symptom")
        self.kg.add_edge("Runny Nose", "Common Cold", relation="has_symptom")

        self.kg.add_edge("Cough", "Flu", relation="has_symptom")
        self.kg.add_edge("Fever", "Flu", relation="has_symptom")
        self.kg.add_edge("Body Aches", "Flu", relation="has_symptom")
        self.kg.add_edge("Fatigue", "Flu", relation="has_symptom")

        self.kg.add_edge("Runny Nose", "Allergies", relation="has_symptom")
        self.kg.add_edge("Sore Throat", "Strep Throat", relation="has_symptom")
        self.kg.add_edge("Rash", "Measles", relation="has_symptom")
        self.kg.add_edge("Nausea", "Gastroenteritis", relation="has_symptom")

        # Condition-Treatment relations
        self.kg.add_edge("Common Cold", "Rest", relation="treated_by")
        self.kg.add_edge("Common Cold", "Fluids", relation="treated_by")
        self.kg.add_edge("Common Cold", "Pain Relievers", relation="treated_by")

        self.kg.add_edge("Flu", "Rest", relation="treated_by")
        self.kg.add_edge("Flu", "Fluids", relation="treated_by")
        self.kg.add_edge("Flu", "Antiviral Medication", relation="treated_by")

        self.kg.add_edge("Allergies", "Antihistamines", relation="treated_by")
        self.kg.add_edge("Strep Throat", "Antibiotics", relation="treated_by")

    def diagnose(self, symptoms):
        patient_context = f"Initial symptoms: {', '.join(symptoms)}"
        print(f"Patient: {patient_context}")

        # LLM Initial Hypothesis
        initial_hypothesis = simulate_llm_reasoning("initial symptoms", patient_context)
        print(f"\nLLM Initial Hypothesis: {initial_hypothesis}")
        
        current_nodes = set(symptoms)
        explored_nodes = set()
        diagnosis_candidates = set()
        
        # Simulate iterative graph traversal
        for depth in range(3): # Max depth for exploration
            new_nodes_to_explore = set()
            candidates_for_pruning = []
            
            for node in current_nodes:
                if node in explored_nodes: # Avoid re-exploring current batch to generate pruning candidates
                    continue
                
                neighbors = list(self.kg.neighbors(node))
                for neighbor in neighbors:
                    candidates_for_pruning.append(neighbor)
                
                explored_nodes.add(node)

            if not candidates_for_pruning and not new_nodes_to_explore: # Stop if no new paths
                break

            # Lightweight Pruning
            pruned_candidates = simulate_lightweight_pruning(candidates_for_pruning, patient_context)
            print(f"  Depth {depth+1} - Lightweight Pruning filtered {len(candidates_for_pruning)} to {len(pruned_candidates)} candidates for '{patient_context}'.")
            
            for candidate in pruned_candidates:
                new_nodes_to_explore.add(candidate)
                if self.kg.nodes[candidate].get("type") == "condition":
                    diagnosis_candidates.add(candidate)
            
            # Selective LLM Engagement for complex evaluations
            if depth == 1 and diagnosis_candidates: # Simulate complex evaluation at a certain depth
                llm_complex_query = f"complex evaluation based on current symptoms {', '.join(symptoms)} and potential conditions {', '.join(diagnosis_candidates)}"
                llm_complex_result = simulate_llm_reasoning(llm_complex_query, patient_context)
                print(f"  Depth {depth+1} - Selective LLM Engagement: {llm_complex_result}")

            current_nodes = new_nodes_to_explore
            if not current_nodes:
                break

        # Final LLM Diagnosis
        final_diagnosis_context = f"After exploring graph with initial symptoms {', '.join(symptoms)} and finding potential conditions: {', '.join(diagnosis_candidates)}"
        final_diagnosis = simulate_llm_reasoning("final diagnosis", final_diagnosis_context)
        print(f"\nFinal Diagnosis: {final_diagnosis}")
        return final_diagnosis

if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()
    
    print("--- Scenario 1: Flu Symptoms ---")
    assistant.diagnose(symptoms=["Cough", "Fever", "Body Aches", "Fatigue"])

    print("\n--- Scenario 2: Allergy Symptoms ---")
    assistant.diagnose(symptoms=["Runny Nose", "Sore Throat", "Headache"])

    print("\n--- Scenario 3: Strep Throat Symptoms ---")
    assistant.diagnose(symptoms=["Sore Throat", "Fever"])
