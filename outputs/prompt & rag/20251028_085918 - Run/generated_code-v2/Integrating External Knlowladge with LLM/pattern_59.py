class MedicalKnowledgeGraph:
    def __init__(self):
        self.nodes = {
            "Fever": {"type": "symptom", "description": "Elevated body temperature.", "associated_with": ["Influenza", "Common Cold", "Pneumonia", "Meningitis"]},
            "Cough": {"type": "symptom", "description": "Sudden expulsion of air from the lungs.", "associated_with": ["Influenza", "Common Cold", "Pneumonia", "Bronchitis"]},
            "Headache": {"type": "symptom", "description": "Pain in the head.", "associated_with": ["Influenza", "Common Cold", "Migraine", "Meningitis"]},
            "Sore Throat": {"type": "symptom", "description": "Pain or irritation of the throat.", "associated_with": ["Common Cold", "Strep Throat"]},
            "Fatigue": {"type": "symptom", "description": "Extreme tiredness.", "associated_with": ["Influenza", "Common Cold", "Anemia"]},
            "Stiffness in Neck": {"type": "symptom", "description": "Difficulty moving neck.", "associated_with": ["Meningitis"]},
            "Rash": {"type": "symptom", "description": "Eruption on the skin.", "associated_with": ["Measles", "Meningitis"]},
            "Influenza": {"type": "disease", "description": "Viral infection of the respiratory system.", "symptoms": ["Fever", "Cough", "Headache", "Fatigue"]},
            "Common Cold": {"type": "disease", "description": "Viral infection of the nose and throat.", "symptoms": ["Fever", "Cough", "Headache", "Sore Throat", "Fatigue"]},
            "Pneumonia": {"type": "disease", "description": "Inflammation of the lung.", "symptoms": ["Fever", "Cough", "Fatigue"]},
            "Meningitis": {"type": "disease", "description": "Inflammation of the membranes surrounding the brain and spinal cord.", "symptoms": ["Fever", "Headache", "Stiffness in Neck", "Rash"]},
            "Migraine": {"type": "disease", "description": "Severe headache.", "symptoms": ["Headache"]},
            "Strep Throat": {"type": "disease", "description": "Bacterial infection of the throat.", "symptoms": ["Sore Throat", "Fever"]},
            "Anemia": {"type": "disease", "description": "Lack of healthy red blood cells.", "symptoms": ["Fatigue"]},
            "Measles": {"type": "disease", "description": "Viral infection with a rash.", "symptoms": ["Fever", "Rash"]}
        }

        self.relationships = {
            "Fever": {"associated_with": ["Influenza", "Common Cold", "Pneumonia", "Meningitis"]},
            "Cough": {"associated_with": ["Influenza", "Common Cold", "Pneumonia", "Bronchitis"]},
            "Headache": {"associated_with": ["Influenza", "Common Cold", "Migraine", "Meningitis"]},
            "Sore Throat": {"associated_with": ["Common Cold", "Strep Throat"]},
            "Fatigue": {"associated_with": ["Influenza", "Common Cold", "Anemia"]},
            "Stiffness in Neck": {"associated_with": ["Meningitis"]},
            "Rash": {"associated_with": ["Measles", "Meningitis"]}
        }

class InputPreprocessor:
    def extract_entities(self, patient_input):
        normalized_input = patient_input.lower()
        potential_entities = [
            "fever", "cough", "headache", "sore throat", "fatigue", "stiffness in neck", "rash",
            "influenza", "common cold", "pneumonia", "meningitis", "migraine", "strep throat", "anemia", "measles"
        ]
        extracted = [entity for entity in potential_entities if entity in normalized_input]
        return extracted

class HybridPruningModule:
    def __init__(self, kg):
        self.kg = kg

    def lightweight_pruning(self, extracted_entities):
        relevant_nodes = set()
        for entity in extracted_entities:
            # Directly check if entity is a known symptom or disease in KG
            if entity.capitalize() in self.kg.nodes:
                relevant_nodes.add(entity.capitalize())

            # Find diseases associated with symptoms
            if entity.capitalize() in self.kg.relationships:
                for disease in self.kg.relationships[entity.capitalize()].get("associated_with", []):
                    relevant_nodes.add(disease)

            # Find symptoms for diseases if an entity is a disease
            if entity.capitalize() in self.kg.nodes and self.kg.nodes[entity.capitalize()].get("type") == "disease":
                for symptom in self.kg.nodes[entity.capitalize()].get("symptoms", []):
                    relevant_nodes.add(symptom)
        
        pruned_subgraph = {}
        for node_name in relevant_nodes:
            if node_name in self.kg.nodes:
                pruned_subgraph[node_name] = self.kg.nodes[node_name]
        return pruned_subgraph

    def llm_deeper_reasoning(self, pruned_subgraph, patient_context):
        if not pruned_subgraph:
            return "No specific diagnoses could be inferred from the provided symptoms and limited medical knowledge. Please consult a medical professional."

        possible_diseases = set()
        present_symptoms = set()

        for node_name, node_data in pruned_subgraph.items():
            if node_data["type"] == "disease":
                possible_diseases.add(node_name)
            elif node_data["type"] == "symptom":
                present_symptoms.add(node_name)
        
        diagnosis_likelihood = {}
        for disease in possible_diseases:
            disease_symptoms = set(self.kg.nodes[disease].get("symptoms", []))
            common_symptoms = present_symptoms.intersection(disease_symptoms)
            
            if len(disease_symptoms) > 0:
                match_ratio = len(common_symptoms) / len(disease_symptoms)
            else:
                match_ratio = 0
            
            diagnosis_likelihood[disease] = match_ratio

        # Simulate LLM's nuanced reasoning and differential diagnosis
        # This part would typically involve complex prompt engineering and LLM calls
        # For this simulation, we'll pick the top diseases based on match_ratio
        
        sorted_diagnoses = sorted(diagnosis_likelihood.items(), key=lambda item: item[1], reverse=True)

        if not sorted_diagnoses or sorted_diagnoses[0][1] == 0:
            return "Based on the symptoms, no clear diagnosis can be made from the available knowledge. Further investigation is recommended."

        top_diagnoses = []
        for disease, score in sorted_diagnoses:
            if score > 0:
                top_diagnoses.append(f"{disease} (Symptom Match: {score:.2f})")
        
        # Add a more complex reasoning for top diagnoses, simulating LLM
        if len(top_diagnoses) > 1 and sorted_diagnoses[0][1] == sorted_diagnoses[1][1]:
            reasoning = f"Multiple conditions share similar symptoms. Considering patient context: {patient_context}, and the following matched symptoms: {', '.join(present_symptoms)}, the top possibilities include: {', '.join(top_diagnoses)}."
        elif top_diagnoses:
            reasoning = f"Given patient context: {patient_context}, and the matched symptoms: {', '.join(present_symptoms)}, the most likely diagnosis is {top_diagnoses[0]}."
        else:
            reasoning = "No clear diagnostic path found based on available data."
        
        return {"diagnoses": top_diagnoses, "reasoning": reasoning}

class OutputGenerator:
    def format_output(self, llm_output):
        if isinstance(llm_output, str):
            return {"diagnostic_suggestions": [llm_output], "explanation": llm_output}

        diagnoses = llm_output.get("diagnoses", [])
        reasoning = llm_output.get("reasoning", "No detailed reasoning available.")

        formatted_suggestions = "Diagnostic Suggestions:\n"
        if diagnoses:
            for diag in diagnoses:
                formatted_suggestions += f"- {diag}\n"
        else:
            formatted_suggestions += "- No specific diagnoses suggested.\n"
        
        explanation = f"Explanation: {reasoning}\n\n"
        return {"diagnostic_suggestions": formatted_suggestions, "explanation": explanation}

class MedicalDiagnosisAssistant:
    def __init__(self):
        self.kg = MedicalKnowledgeGraph()
        self.preprocessor = InputPreprocessor()
        self.pruning_module = HybridPruningModule(self.kg)
        self.output_generator = OutputGenerator()

    def diagnose_patient(self, patient_symptoms, patient_history=None, lab_results=None):
        patient_context = f"Patient symptoms: {patient_symptoms}. "
        if patient_history: patient_context += f"History: {patient_history}. "
        if lab_results: patient_context += f"Lab results: {lab_results}. "

        extracted_entities = self.preprocessor.extract_entities(patient_symptoms)
        
        pruned_subgraph = self.pruning_module.lightweight_pruning(extracted_entities)
        
        llm_output = self.pruning_module.llm_deeper_reasoning(pruned_subgraph, patient_context)
        
        final_output = self.output_generator.format_output(llm_output)
        
        return final_output

if __name__ == "__main__":
    assistant = MedicalDiagnosisAssistant()

    # Example 1: Clear case
    print("\n--- Patient 1: Clear Case ---")
    patient_input_1 = "I have a fever, cough, and a bad headache. Feeling very tired."
    diagnosis_1 = assistant.diagnose_patient(patient_input_1)
    print(diagnosis_1["diagnostic_suggestions"])
    print(diagnosis_1["explanation"])

    # Example 2: More ambiguous case
    print("\n--- Patient 2: Ambiguous Case ---")
    patient_input_2 = "My throat is really sore and I have a low-grade fever."
    diagnosis_2 = assistant.diagnose_patient(patient_input_2, patient_history="No significant medical history.")
    print(diagnosis_2["diagnostic_suggestions"])
    print(diagnosis_2["explanation"])

    # Example 3: Specific but less common
    print("\n--- Patient 3: Specific Case ---")
    patient_input_3 = "I have a high fever, severe headache, and my neck feels stiff."
    diagnosis_3 = assistant.diagnose_patient(patient_input_3)
    print(diagnosis_3["diagnostic_suggestions"])
    print(diagnosis_3["explanation"])

    # Example 4: No matching symptoms
    print("\n--- Patient 4: No Match Case ---")
    patient_input_4 = "My elbow hurts."
    diagnosis_4 = assistant.diagnose_patient(patient_input_4)
    print(diagnosis_4["diagnostic_suggestions"])
    print(diagnosis_4["explanation"])
