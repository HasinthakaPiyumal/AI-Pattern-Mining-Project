"""
FactualExpert: Simulates fetching facts from a medical database based on symptoms.
Uses a placeholder for Retrieval Augmented Generation (RAG).
"""

class FactualExpert:
    def __init__(self):
        self.medical_database = {
            "fever": ["Influenza", "Common Cold", "Bacterial Infection"],
            "cough": ["Bronchitis", "Pneumonia", "Asthma"],
            "headache": ["Migraine", "Tension Headache", "Sinusitis"],
            "fatigue": ["Anemia", "Hypothyroidism", "Chronic Fatigue Syndrome"],
            "sore throat": ["Streptococcal Pharyngitis", "Viral Pharyngitis"],
            "chest pain": ["Heart Attack", "Angina", "Pleurisy"],
            "shortness of breath": ["Asthma", "COPD", "Heart Failure"]
        }

    def _retrieve_facts(self, symptom):
        """
        Simulates retrieval from a medical database.
        In a real scenario, this would involve a vector database query or similar RAG.
        """
        return self.medical_database.get(symptom.lower(), [])

    def diagnose(self, symptoms):
        """
        Provides factual diagnoses based on symptoms by simulating database retrieval.
        """
        potential_diagnoses = {}
        supporting_facts = []

        for symptom in symptoms:
            facts = self._retrieve_facts(symptom)
            if facts:
                supporting_facts.append(f"Facts for '{symptom}': {', '.join(facts)}")
                for diagnosis in facts:
                    potential_diagnoses[diagnosis] = potential_diagnoses.get(diagnosis, 0) + 1

        # Rank diagnoses by how many symptoms they are linked to
        sorted_diagnoses = sorted(potential_diagnoses.items(), key=lambda item: item[1], reverse=True)
        
        result = {
            "expert_name": "Factual Expert",
            "diagnoses": [{ "name": diag[0], "score": diag[1] } for diag in sorted_diagnoses],
            "reasoning": supporting_facts if supporting_facts else ["No direct factual matches found for symptoms."]
        }
        return result

# Example Usage (for testing the expert independently)
if __name__ == "__main__":
    expert = FactualExpert()
    case1 = ["fever", "cough"]
    print(f"Case 1: {case1}")
    print(expert.diagnose(case1))
    
    case2 = ["headache", "fatigue", "sore throat"]
    print(f"\nCase 2: {case2}")
    print(expert.diagnose(case2))
    
    case3 = ["unknown symptom"]
    print(f"\nCase 3: {case3}")
    print(expert.diagnose(case3))