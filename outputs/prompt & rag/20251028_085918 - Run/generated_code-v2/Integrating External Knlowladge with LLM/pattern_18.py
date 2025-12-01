class MedicalKnowledgeGraph:
    def __init__(self):
        self.triples = []

    def add_triple(self, entity1, relation, entity2):
        self.triples.append((entity1, relation, entity2))

    def query(self, keyword):
        results = []
        for triple in self.triples:
            if keyword.lower() in str(triple[0]).lower() or \
               keyword.lower() in str(triple[1]).lower() or \
               keyword.lower() in str(triple[2]).lower():
                results.append(triple)
        return results

    def find_paths(self, start_entity):
        relevant_triples = []
        for triple in self.triples:
            if start_entity.lower() in str(triple[0]).lower() or \
               start_entity.lower() in str(triple[2]).lower():
                relevant_triples.append(triple)
        return relevant_triples

class PromptGenerator:
    def format_triples(self, triples_list):
        if not triples_list:
            return "No specific medical knowledge found."
        formatted_strings = []
        for e1, r, e2 in triples_list:
            formatted_strings.append(f"{e1} {r} {e2}")
        return "; ".join(formatted_strings) + "."

    def construct_llm_prompt(self, patient_data, formatted_kg_info):
        symptoms = ", ".join(patient_data.get("symptoms", []))
        risk_factors = ", ".join(patient_data.get("risk_factors", []))
        travel_history = patient_data.get("travel_history", "N/A")

        prompt = f"""Medical Knowledge:\n{formatted_kg_info}\n\nPatient Information:\nSymptoms: {symptoms}\nRisk Factors: {risk_factors}\nTravel History: {travel_history}\n\nBased on the medical knowledge and patient information, what are the most likely diagnoses and recommended further tests? Please provide a concise diagnostic summary."""
        return prompt

class LLMIntegration:
    def send_prompt_to_llm(self, prompt_string):
        print("\n--- Sending to LLM ---")
        print(prompt_string)
        print("\n--- LLM Response (Simulated) ---")
        # Simulate an LLM response based on the example scenario
        if "Tuberculosis" in prompt_string and "cough" in prompt_string and "fever" in prompt_string:
            return "Likely Diagnosis: Tuberculosis. Recommended Tests: Chest X-ray, Sputum culture, Tuberculin skin test."
        else:
            return "Unable to provide a specific diagnosis based on the provided information. Consider broader differential diagnoses and additional tests."

def main():
    kg = MedicalKnowledgeGraph()

    # Populate KG with example medical facts
    kg.add_triple("persistent cough", "associated_with", "Tuberculosis")
    kg.add_triple("fever", "associated_with", "Tuberculosis")
    kg.add_triple("fatigue", "associated_with", "Tuberculosis")
    kg.add_triple("chest pain", "associated_with", "Tuberculosis")
    kg.add_triple("smoking", "increases_risk_of", "Tuberculosis")
    kg.add_triple("Southeast Asia", "endemic_for", "Tuberculosis")
    kg.add_triple("shortness of breath", "symptom_of", "Asthma")
    kg.add_triple("wheezing", "symptom_of", "Asthma")
    kg.add_triple("pollen", "trigger_for", "Asthma")
    kg.add_triple("headache", "symptom_of", "Migraine")
    kg.add_triple("nausea", "symptom_of", "Migraine")

    prompt_gen = PromptGenerator()
    llm_integrator = LLMIntegration()

    print("Welcome to the Medical Diagnostic Assistant!")
    print("Please enter patient information.")

    patient_symptoms_input = input("Enter symptoms (comma-separated): ")
    patient_risk_factors_input = input("Enter risk factors (comma-separated): ")
    patient_travel_history_input = input("Enter recent travel history: ")

    patient_data = {
        "symptoms": [s.strip() for s in patient_symptoms_input.split(",") if s.strip()],
        "risk_factors": [r.strip() for r in patient_risk_factors_input.split(",") if r.strip()],
        "travel_history": patient_travel_history_input.strip()
    }

    all_patient_keywords = []
    all_patient_keywords.extend(patient_data["symptoms"])
    all_patient_keywords.extend(patient_data["risk_factors"])
    if patient_data["travel_history"] and patient_data["travel_history"] != "N/A":
        all_patient_keywords.append(patient_data["travel_history"])

    retrieved_kg_triples = []
    for keyword in all_patient_keywords:
        retrieved_kg_triples.extend(kg.find_paths(keyword))
    
    # Remove duplicate triples
    retrieved_kg_triples = list(set(retrieved_kg_triples))

    formatted_kg_info = prompt_gen.format_triples(retrieved_kg_triples)
    llm_prompt = prompt_gen.construct_llm_prompt(patient_data, formatted_kg_info)
    diagnostic_output = llm_integrator.send_prompt_to_llm(llm_prompt)

    print("\n--- Diagnostic Result ---")
    print(diagnostic_output)

if __name__ == "__main__":
    main()