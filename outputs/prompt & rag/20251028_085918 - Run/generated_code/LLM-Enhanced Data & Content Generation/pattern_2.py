
class MedicalDiagnosticAssistant:
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Simulates loading a comprehensive medical knowledge base."""
        return {
            "Influenza": {
                "symptoms": ["fever", "cough", "sore throat", "muscle aches", "fatigue", "headache"],
                "causes": "Influenza virus infection",
                "treatments": "Antiviral medications (oseltamivir, zanamivir), rest, fluids, pain relievers.",
                "description": "A common viral infection that can be deadly, especially in high-risk groups."
            },
            "Common Cold": {
                "symptoms": ["runny nose", "sneezing", "sore throat", "mild cough", "congestion"],
                "causes": "Rhinoviruses, coronaviruses",
                "treatments": "Rest, fluids, over-the-counter cold medications (decongestants, pain relievers).",
                "description": "A mild viral infection of the nose and throat."
            },
            "Streptococcal Pharyngitis": {
                "symptoms": ["sore throat", "difficulty swallowing", "fever", "red spots on roof of mouth", "swollen tonsils"],
                "causes": "Streptococcus pyogenes bacteria",
                "treatments": "Antibiotics (penicillin, amoxicillin), pain relievers.",
                "description": "A bacterial infection of the throat and tonsils."
            },
            "Allergies": {
                "symptoms": ["sneezing", "itchy eyes", "runny nose", "skin rash", "hives"],
                "causes": "Exposure to allergens (pollen, dust mites, pet dander, certain foods)",
                "treatments": "Antihistamines, corticosteroids, allergen avoidance.",
                "description": "An immune system reaction to a substance that is normally harmless."
            },
            "Pneumonia": {
                "symptoms": ["cough with phlegm", "fever", "chills", "shortness of breath", "chest pain"],
                "causes": "Bacteria, viruses, fungi",
                "treatments": "Antibiotics (for bacterial), antiviral (for viral), antifungal (for fungal), oxygen therapy, fluids.",
                "description": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid."
            }
        }

    def retrieve_medical_context(self, symptoms_query):
        """Retrieves relevant medical information from the knowledge base based on symptoms."""
        symptoms_query = [s.strip().lower() for s in symptoms_query.split(',')]
        relevant_diseases = {}

        for disease, data in self.knowledge_base.items():
            disease_symptoms = [s.lower() for s in data["symptoms"]]
            # Check for overlap in symptoms
            matching_symptoms = [s for s in symptoms_query if s in disease_symptoms]

            if matching_symptoms:
                # Score based on number of matching symptoms
                score = len(matching_symptoms)
                if score > 0:
                    relevant_diseases[disease] = {
                        "score": score,
                        "data": data
                    }
        
        # Sort by score to get most relevant first
        sorted_diseases = sorted(relevant_diseases.items(), key=lambda item: item[1]["score"], reverse=True)
        
        # Return top N relevant contexts (e.g., top 3 for this simulation)
        retrieved_context = []
        for disease, info in sorted_diseases[:3]:
            retrieved_context.append(f"Disease: {disease}\nSymptoms: {', '.join(info['data']['symptoms'])}. \nCauses: {info['data']['causes']}. \nTreatments: {info['data']['treatments']}.\nDescription: {info['data']['description']}\n---")
        return "\n".join(retrieved_context)

    def llm_reasoning_module(self, symptoms_query, retrieved_context):
        """Simulates an LLM reasoning process to provide diagnostic suggestions and treatment plans."""
        if not retrieved_context:
            return "Based on the symptoms provided, I couldn't find a direct match in my knowledge base. Please provide more details or consult a medical professional."

        prompt = (
            f"Given the following patient symptoms: '{symptoms_query}'.\n\n"
            f"And the following retrieved medical information:\n{retrieved_context}\n\n"
            f"As a medical diagnostic assistant, provide potential diagnoses, reasons for the diagnoses based on symptoms, and suggested treatment plans. "
            f"Focus on the most likely conditions and explain your reasoning clearly. If multiple conditions are possible, list them with their differentiating factors."
        )

        # --- Simulate LLM's response generation ---
        # In a real scenario, this would involve calling a true LLM API or model inference.
        # For this demonstration, we'll parse the retrieved context to construct a coherent response.

        response_parts = [
            "### Medical Diagnostic Suggestion\n"
            "Based on the provided symptoms and retrieved medical knowledge, here are potential diagnoses and treatment suggestions:"
        ]

        context_lines = retrieved_context.split('---')
        for context_block in context_lines:
            if context_block.strip():
                lines = context_block.strip().split('\n')
                disease = lines[0].replace('Disease: ', '').strip()
                symptoms = lines[1].replace('Symptoms: ', '').strip()
                causes = lines[2].replace('Causes: ', '').strip()
                treatments = lines[3].replace('Treatments: ', '').strip()
                description = lines[4].replace('Description: ', '').strip()
                
                # Simple check for symptom overlap to simulate reasoning
                query_symptoms_list = [s.strip().lower() for s in symptoms_query.split(',')]
                disease_symptoms_list = [s.strip().lower() for s in symptoms.replace('.', '').split(', ')]
                common_symptoms = set(query_symptoms_list).intersection(set(disease_symptoms_list))

                if common_symptoms:
                    response_parts.append(
                        f"\n**Potential Diagnosis: {disease}**\n"
                        f"*   **Reasoning:** This diagnosis is suggested due to the presence of symptoms like {', '.join(common_symptoms)} which align with {disease}.\n"
                        f"*   **Description:** {description}\n"
                        f"*   **Recommended Treatment:** {treatments}\n"
                    )
        
        if len(response_parts) == 1: # Only header, no diagnoses found
            return "Based on the symptoms provided, I couldn't find a direct match in my knowledge base. Please provide more details or consult a medical professional."

        return "\n".join(response_parts)

    def unified_interaction_layer(self, symptoms_query):
        """Orchestrates retrieval and reasoning for a unified response."""
        print(f"\n[Assistant]: Retrieving relevant medical information for: '{symptoms_query}'...")
        retrieved_context = self.retrieve_medical_context(symptoms_query)
        
        print("[Assistant]: Performing LLM-powered reasoning...")
        diagnostic_suggestions = self.llm_reasoning_module(symptoms_query, retrieved_context)
        
        return diagnostic_suggestions

    def run_cli(self):
        """Runs the command-line interface for the assistant."""
        print("\n--- Medical Diagnostic Assistant (Unified Retrieval & Reasoning) ---")
        print("Enter patient symptoms, separated by commas (e.g., 'fever, cough, sore throat').")
        print("Type 'exit' to quit.")

        while True:
            user_input = input("\nEnter symptoms: ").strip()
            if user_input.lower() == 'exit':
                print("Exiting Medical Diagnostic Assistant. Goodbye!")
                break
            elif not user_input:
                print("[Assistant]: Please enter some symptoms to get a diagnosis.")
                continue
            
            response = self.unified_interaction_layer(user_input)
            print(response)

if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()
    assistant.run_cli()
