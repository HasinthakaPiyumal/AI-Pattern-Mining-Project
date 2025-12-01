class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = {
            "fever": {"diseases": ["Flu", "Common Cold", "Malaria"], "treatments": ["Rest", "Fluids", "Paracetamol"]},
            "cough": {"diseases": ["Common Cold", "Bronchitis", "Pneumonia"], "treatments": ["Cough syrup", "Rest", "Fluids"]},
            "sore throat": {"diseases": ["Common Cold", "Streptococcal Pharyngitis"], "treatments": ["Gargle with salt water", "Lozenges"]},
            "fatigue": {"diseases": ["Flu", "Common Cold", "Anemia"], "treatments": ["Rest", "Nutritious diet"]},
            "headache": {"diseases": ["Flu", "Common Cold", "Migraine"], "treatments": ["Painkillers", "Rest"]},
            "muscle aches": {"diseases": ["Flu"], "treatments": ["Painkillers", "Rest"]},
            "runny nose": {"diseases": ["Common Cold", "Allergies"], "treatments": ["Antihistamines", "Decongestants"]},
            "shortness of breath": {"diseases": ["Pneumonia", "Asthma"], "treatments": ["Inhaler", "Oxygen therapy (severe)"]},
            "chills": {"diseases": ["Flu", "Malaria"], "treatments": ["Warm blankets", "Fluids"]}
        }

    def query(self, symptom_keywords):
        found_diseases = []
        found_treatments = []
        for keyword in symptom_keywords:
            if keyword in self.graph:
                found_diseases.extend(self.graph[keyword]["diseases"])
                found_treatments.extend(self.graph[keyword]["treatments"])
        return list(set(found_diseases)), list(set(found_treatments))

class LLMAgent:
    def __init__(self, kg):
        self.kg = kg

    def _generate_kg_query(self, symptoms):
        # Simulate LLM extracting keywords from symptoms
        symptom_keywords = [s.strip().lower() for s in symptoms.split(',')]
        return symptom_keywords

    def _kg_interaction(self, query_keywords):
        return self.kg.query(query_keywords)

    def _reasoning_engine(self, matched_diseases, matched_treatments, original_symptoms):
        if not matched_diseases:
            return "I couldn't find a clear diagnosis based on your symptoms in my knowledge graph. Please consult a medical professional.", []

        # Simple reasoning: prioritize the most frequently appearing disease
        disease_counts = {disease: matched_diseases.count(disease) for disease in matched_diseases}
        most_likely_disease = max(disease_counts, key=disease_counts.get)

        return f"Based on your symptoms, you might have {most_likely_disease}.", matched_treatments

    def _response_generation(self, diagnosis, treatments):
        response = f"Diagnosis: {diagnosis}"
        if treatments:
            response += "\nRecommended Treatments: " + ", ".join(treatments)
        else:
            response += "\nNo specific treatments recommended based on current knowledge."
        return response

    def process_symptoms(self, symptoms_input):
        # 1. Prompting Module (implicit: direct input)

        # 2. KG Query Generation
        query_keywords = self._generate_kg_query(symptoms_input)

        # 3. KG Interaction Module
        matched_diseases, matched_treatments = self._kg_interaction(query_keywords)

        # 4. Reasoning Engine
        diagnosis_message, final_treatments = self._reasoning_engine(matched_diseases, matched_treatments, symptoms_input)

        # 5. Response Generation
        final_response = self._response_generation(diagnosis_message, final_treatments)

        return final_response

def main():
    print("Welcome to the Medical Diagnosis and Treatment Recommendation System (KG-Agent Prototype)")
    print("Enter your symptoms separated by commas (e.g., fever, cough, headache). Type 'exit' to quit.")

    medical_kg = MedicalKnowledgeGraph()
    llm_agent = LLMAgent(medical_kg)

    while True:
        user_symptoms = input("\nEnter your symptoms: ")
        if user_symptoms.lower() == 'exit':
            print("Thank you for using the system. Goodbye!")
            break

        response = llm_agent.process_symptoms(user_symptoms)
        print("\n" + response)

if __name__ == "__main__":
    main()