class MedicalRAG:
    def __init__(self):
        # Simulate a medical knowledge base (in a real app, this would be a vector DB or structured DB)
        self.medical_knowledge = {
            "fever": "Elevated body temperature, often indicative of infection or inflammation.",
            "cough": "A sudden, forceful exhalation of air, a common symptom of respiratory issues.",
            "headache": "Pain in the head, can be a symptom of various conditions, from mild to severe.",
            "fatigue": "Extreme tiredness, often a symptom of illness, stress, or lack of sleep.",
            "sore throat": "Pain or irritation of the throat, often due to infection.",
            "shortness of breath": "Difficulty breathing, can indicate respiratory or cardiac problems.",
            "chest pain": "Discomfort or pain in the chest, requires immediate medical evaluation.",
            "nausea": "Feeling of sickness with an urge to vomit.",
            "vomiting": "Expelling stomach contents through the mouth.",
            "diarrhea": "Frequent, loose, watery stools.",
            "rash": "An area of irritated or swollen skin, often a symptom of allergies or infections.",
            "influenza": "Viral infection affecting the respiratory system, common symptoms include fever, cough, body aches.",
            "common cold": "Viral infection of the nose and throat, milder than flu, with symptoms like runny nose, sneezing, sore throat.",
            "strep throat": "Bacterial infection causing sore throat, often with fever and difficulty swallowing.",
            "pneumonia": "Lung infection, symptoms include cough with phlegm, fever, chills, shortness of breath.",
            "gastroenteritis": "Stomach flu, symptoms include nausea, vomiting, diarrhea, stomach cramps.",
            "allergic_reaction": "Immune system's response to a substance, symptoms can include rash, itching, swelling, shortness of breath.",
            "migraine": "Severe headache, often accompanied by pulsing pain, sensitivity to light and sound, nausea.",
            "hypertension": "High blood pressure, often asymptomatic but can lead to serious health issues.",
            "diabetes": "Chronic condition affecting how the body turns food into energy, characterized by high blood sugar levels."
        }
        # Simulate an LLM for reasoning (simplified for this example)
        self.llm = self._simulate_llm_reasoning

    def _retrieve_info(self, query):
        """Simulates retrieval from a medical knowledge base."""
        retrieved_facts = []
        for keyword, fact in self.medical_knowledge.items():
            if keyword in query.lower():
                retrieved_facts.append(f"Fact: {fact}")
        return "\n".join(retrieved_facts) if retrieved_facts else "No specific medical facts retrieved for this symptom."

    def _simulate_llm_reasoning(self, prompt):
        """
        Simulates an LLM generating reasoning and a diagnosis.
        This is a highly simplified representation of an LLM.
        """
        response_parts = []
        patient_data = self._extract_patient_data_from_prompt(prompt)

        symptoms = patient_data.get("symptoms", [])
        history = patient_data.get("history", "")
        lab_results = patient_data.get("lab_results", "")

        response_parts.append("--- LLM Reasoning Process (Chain-of-Thought) ---")
        response_parts.append(f"Analyzing patient symptoms: {', '.join(symptoms)}")

        # Step 1: Initial symptom analysis and fact retrieval
        for symptom in symptoms:
            retrieved = self._retrieve_info(symptom)
            response_parts.append(f"  - Symptom '{symptom}': {retrieved}")

        # Step 2: Consider common conditions based on symptoms
        possible_conditions = []
        if "fever" in symptoms and "cough" in symptoms:
            possible_conditions.append("Influenza")
            possible_conditions.append("Common Cold")
            possible_conditions.append("Pneumonia")
        if "sore throat" in symptoms and "fever" in symptoms:
            possible_conditions.append("Strep Throat")
            possible_conditions.append("Influenza")
        if "nausea" in symptoms or "vomiting" in symptoms or "diarrhea" in symptoms:
            possible_conditions.append("Gastroenteritis")
        if "rash" in symptoms:
            possible_conditions.append("Allergic_Reaction")
        if "headache" in symptoms and not any(s in symptoms for s in ["fever", "cough"]):
            possible_conditions.append("Migraine")

        if possible_conditions:
            response_parts.append(f"\nStep 3: Considering common conditions based on symptom clusters:")
            for cond in set(possible_conditions): # Use set to avoid duplicates
                response_parts.append(f"  - {cond}: Often presents with similar symptoms.")
        else:
            response_parts.append("\nStep 3: No clear common conditions immediately apparent from symptom clusters. Further investigation needed.")

        # Step 4: Synthesize a preliminary diagnosis (very basic)
        diagnosis = "Undetermined"
        if "Influenza" in possible_conditions:
            diagnosis = "Possible Influenza"
        elif "Strep Throat" in possible_conditions:
            diagnosis = "Possible Strep Throat"
        elif "Gastroenteritis" in possible_conditions:
            diagnosis = "Possible Gastroenteritis"
        elif "Migraine" in possible_conditions:
            diagnosis = "Possible Migraine"
        elif "Allergic_Reaction" in possible_conditions: # Add allergic reaction
            diagnosis = "Possible Allergic_Reaction"
        elif len(symptoms) > 0 and not possible_conditions:
            diagnosis = "Further investigation required due to non-specific symptoms."
        elif not symptoms:
            diagnosis = "No symptoms provided, cannot make a diagnosis."

        response_parts.append("\nStep 5: Synthesizing preliminary diagnosis.")
        response_parts.append(f"Preliminary Differential Diagnosis: {diagnosis}")
        response_parts.append("--- End LLM Reasoning Process ---")

        return {
            "reasoning_steps": "\n".join(response_parts),
            "preliminary_diagnosis": diagnosis
        }

    def _extract_patient_data_from_prompt(self, prompt):
        """Helper to extract structured data from a conversational prompt."""
        data = {"symptoms": [], "history": "", "lab_results": ""}
        
        symptoms_line = ""
        for line in prompt.split('\n'):
            if "symptoms:" in line.lower():
                symptoms_line = line.lower().split("symptoms:")[1].strip()
                break

        if symptoms_line:
            data["symptoms"] = [s.strip() for s in symptoms_line.split(',') if s.strip()]

        if "history:" in prompt.lower():
            data["history"] = prompt.lower().split("history:")[1].split("lab results:")[0].strip()
        if "lab results:" in prompt.lower():
            data["lab_results"] = prompt.lower().split("lab results:")[1].strip()
        return data

    def get_diagnosis(self, patient_data_prompt):
        """Main method to get a diagnosis with reasoning."""
        return self.llm(patient_data_prompt)