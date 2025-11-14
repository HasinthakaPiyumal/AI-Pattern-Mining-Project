from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MedicalKnowledgeBase:
    def __init__(self):
        # Simulate patient Electronic Health Records
        self.patient_ehr = {
            "P001": {
                "name": "Alice Smith",
                "age": 55,
                "conditions": ["Type 2 Diabetes", "Hypertension"],
                "medications": ["Metformin", "Lisinopril"],
                "allergies": ["Penicillin"],
                "recent_visits": ["2023-10-20: Blood pressure check, stable", "2023-08-15: Diabetes management review"]
            },
            "P002": {
                "name": "Bob Johnson",
                "age": 30,
                "conditions": ["Asthma"],
                "medications": ["Albuterol inhaler"],
                "allergies": [],
                "recent_visits": ["2023-11-01: Asthma flare-up, managed with inhaler"]
            }
        }

        # Simulate drug information database
        self.drug_info = {
            "Metformin": {
                "class": "Biguanide",
                "uses": "Type 2 diabetes",
                "side_effects": ["Nausea", "Diarrhea", "Lactic acidosis (rare)"],
                "interactions": ["Cimetidine", "Alcohol"]
            },
            "Lisinopril": {
                "class": "ACE Inhibitor",
                "uses": "Hypertension, Heart Failure",
                "side_effects": ["Cough", "Dizziness", "Fatigue"],
                "interactions": ["Potassium-sparing diuretics", "NSAIDs"]
            },
            "Albuterol": {
                "class": "Beta-2 agonist",
                "uses": "Asthma, COPD",
                "side_effects": ["Tremors", "Headache", "Nervousness"],
                "interactions": []
            }
        }

        # Simulate medical literature for RAG
        self.medical_literature_docs = [
            "A study on the efficacy of SGLT2 inhibitors in reducing cardiovascular events in patients with type 2 diabetes.",
            "Recent guidelines for the management of essential hypertension in elderly patients.",
            "Understanding the molecular mechanisms of action for common bronchodilators in asthma.",
            "The role of ACE inhibitors in improving outcomes for heart failure patients.",
            "New insights into the genetic predispositions of severe penicillin allergies."
        ]
        # Using a tiny pre-trained model for demonstration. In a real scenario, use a larger model.
        # This will download the model the first time it's run.
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_embeddings = self.model.encode(self.medical_literature_docs)

    def get_patient_ehr(self, patient_id: str) -> dict:
        """Retrieves Electronic Health Record for a given patient ID."""
        return self.patient_ehr.get(patient_id, {"error": "Patient not found."})

    def get_drug_info(self, drug_name: str) -> dict:
        """Retrieves detailed information about a specific drug."""
        return self.drug_info.get(drug_name, {"error": "Drug information not found."})

    def search_medical_literature(self, query: str, top_k: int = 2) -> list[str]:
        """Searches simulated medical literature for relevant articles."""
        if not query:
            return []
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        
        # Get top_k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            results.append(f"Score: {similarities[i]:.4f} - Doc: {self.medical_literature_docs[i]}")
        return results
