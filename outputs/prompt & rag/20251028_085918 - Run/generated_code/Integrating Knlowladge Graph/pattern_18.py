import networkx as nx
import random

# 1. Medical Knowledge Graph (KG) Module
def load_medical_kg():
    """Loads a sample medical Knowledge Graph using networkx."""
    kg = nx.DiGraph()

    # Add nodes (entities)
    diseases = ["Common Cold", "Influenza", "Pneumonia", "Bronchitis", "Migraine", "Tension Headache", "Appendicitis", "Gallstones"]
    symptoms = ["fever", "cough", "sore throat", "runny nose", "headache", "fatigue", "muscle aches", "chills", "shortness of breath", "abdominal pain", "nausea", "vomiting", "jaundice"]
    drugs = ["Paracetamol", "Ibuprofen", "Antibiotics", "Antivirals", "Migraine Medication"]
    treatments = ["rest", "fluids", "surgery", "pain relief", "antibiotic therapy"]
    labs = ["CBC", "Chest X-ray", "CT Scan", "Ultrasound"]

    kg.add_nodes_from(diseases, type="disease")
    kg.add_nodes_from(symptoms, type="symptom")
    kg.add_nodes_from(drugs, type="drug")
    kg.add_nodes_from(treatments, type="treatment")
    kg.add_nodes_from(labs, type="lab")

    # Add edges (relationships)
    # Disease -> Symptom (presents_with)
    kg.add_edge("Common Cold", "fever", relation="presents_with")
    kg.add_edge("Common Cold", "cough", relation="presents_with")
    kg.add_edge("Common Cold", "sore throat", relation="presents_with")
    kg.add_edge("Common Cold", "runny nose", relation="presents_with")

    kg.add_edge("Influenza", "fever", relation="presents_with")
    kg.add_edge("Influenza", "cough", relation="presents_with")
    kg.add_edge("Influenza", "fatigue", relation="presents_with")
    kg.add_edge("Influenza", "muscle aches", relation="presents_with")
    kg.add_edge("Influenza", "chills", relation="presents_with")

    kg.add_edge("Pneumonia", "fever", relation="presents_with")
    kg.add_edge("Pneumonia", "cough", relation="presents_with")
    kg.add_edge("Pneumonia", "shortness of breath", relation="presents_with")
    kg.add_edge("Pneumonia", "chills", relation="presents_with")

    kg.add_edge("Bronchitis", "cough", relation="presents_with")
    kg.add_edge("Bronchitis", "sore throat", relation="presents_with")
    kg.add_edge("Bronchitis", "fatigue", relation="presents_with")

    kg.add_edge("Migraine", "headache", relation="presents_with")
    kg.add_edge("Migraine", "nausea", relation="presents_with")

    kg.add_edge("Tension Headache", "headache", relation="presents_with")

    kg.add_edge("Appendicitis", "abdominal pain", relation="presents_with")
    kg.add_edge("Appendicitis", "nausea", relation="presents_with")
    kg.add_edge("Appendicitis", "vomiting", relation="presents_with")

    kg.add_edge("Gallstones", "abdominal pain", relation="presents_with")
    kg.add_edge("Gallstones", "nausea", relation="presents_with")
    kg.add_edge("Gallstones", "jaundice", relation="presents_with")

    # Symptom -> Disease (causes, inverse of presents_with)
    # (Implicitly handled by searching graph for presents_with)

    # Disease -> Treatment (treated_by)
    kg.add_edge("Common Cold", "rest", relation="treated_by")
    kg.add_edge("Common Cold", "fluids", relation="treated_by")
    kg.add_edge("Influenza", "Antivirals", relation="treated_by")
    kg.add_edge("Influenza", "rest", relation="treated_by")
    kg.add_edge("Pneumonia", "Antibiotics", relation="treated_by")
    kg.add_edge("Pneumonia", "rest", relation="treated_by")
    kg.add_edge("Migraine", "Migraine Medication", relation="treated_by")
    kg.add_edge("Appendicitis", "surgery", relation="treated_by")
    kg.add_edge("Gallstones", "surgery", relation="treated_by")

    # Treatment -> Drug (uses)
    kg.add_edge("pain relief", "Paracetamol", relation="uses")
    kg.add_edge("pain relief", "Ibuprofen", relation="uses")
    kg.add_edge("antibiotic therapy", "Antibiotics", relation="uses")

    # Symptom -> Lab (suggests_lab)
    kg.add_edge("shortness of breath", "Chest X-ray", relation="suggests_lab")
    kg.add_edge("abdominal pain", "CT Scan", relation="suggests_lab")
    kg.add_edge("abdominal pain", "Ultrasound", relation="suggests_lab")
    kg.add_edge("fever", "CBC", relation="suggests_lab")

    # Drug -> Contraindication (contraindicates) - Simplified
    # kg.add_edge("Ibuprofen", "Kidney Disease", relation="contraindicates") # Example

    return kg

# 2. LLM Interaction & Agent Module (Simulated)
class SimulatedLLM:
    """A class that mimics an LLM's generate method for demonstration."""
    def generate(self, prompt: str) -> str:
        # Simple heuristic-based responses for demonstration
        if "diagnostic hypotheses" in prompt.lower() or "diagnose" in prompt.lower():
            if "fever" in prompt.lower() and "cough" in prompt.lower():
                if "shortness of breath" in prompt.lower():
                    return "Based on fever, cough, and shortness of breath, consider Pneumonia or severe Influenza. Further investigation needed for Bronchitis."
                return "Based on fever and cough, consider Common Cold, Influenza, or Bronchitis."
            elif "headache" in prompt.lower():
                return "For headache, consider Migraine or Tension Headache."
            elif "abdominal pain" in prompt.lower():
                if "nausea" in prompt.lower() and "vomiting" in prompt.lower():
                    return "Abdominal pain with nausea and vomiting suggests Appendicitis or Gallstones. Differentiate with location of pain."
                return "Abdominal pain might indicate various conditions, including Appendicitis or Gallstones."
            else:
                return "Based on the input, various conditions are possible. More specific symptoms or lab results would help narrow down."
        elif "refine" in prompt.lower():
            return "Refining diagnosis... based on new information, focusing on the most probable cause."
        elif "evidence for" in prompt.lower():
            return "Providing evidence from medical literature (simulated)."
        return 