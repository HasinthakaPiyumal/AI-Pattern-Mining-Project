class DrugInteractionModule:
    def __init__(self):
        self.knowledge_base = {
            ("Warfarin", "Aspirin"): {"type": "Major Interaction", "severity": "High", "recommendation": "Avoid concurrent use if possible; if unavoidable, monitor INR closely."},
            ("Metformin", "Alcohol"): {"type": "Moderate Interaction", "severity": "Medium", "recommendation": "Limit alcohol intake due to increased risk of lactic acidosis."},
            ("Simvastatin", "Grapefruit Juice"): {"type": "Major Interaction", "severity": "High", "recommendation": "Avoid grapefruit juice with Simvastatin due to increased risk of muscle damage."},
            ("Lithium", "Ibuprofen"): {"type": "Moderate Interaction", "severity": "Medium", "recommendation": "Monitor lithium levels closely; Ibuprofen can increase lithium concentrations."}
        }

    def get_drug_interactions(self, medication_list):
        interactions = []
        medication_set = set(medication_list)

        for i, drug1 in enumerate(medication_list):
            for drug2 in medication_list[i+1:]:
                interaction_key1 = (drug1, drug2)
                interaction_key2 = (drug2, drug1)
                if interaction_key1 in self.knowledge_base:
                    interaction_info = self.knowledge_base[interaction_key1].copy()
                    interaction_info["drug_1"] = drug1
                    interaction_info["drug_2"] = drug2
                    interactions.append(interaction_info)
                elif interaction_key2 in self.knowledge_base:
                    interaction_info = self.knowledge_base[interaction_key2].copy()
                    interaction_info["drug_1"] = drug2
                    interaction_info["drug_2"] = drug1
                    interactions.append(interaction_info)
            
            # Check for drug-food interactions (simplified, just alcohol and grapefruit juice for demonstration)
            if drug1 == "Metformin" and "Alcohol" in medication_set:
                if ("Metformin", "Alcohol") in self.knowledge_base:
                    interaction_info = self.knowledge_base[("Metformin", "Alcohol")].copy()
                    interaction_info["drug_1"] = "Metformin"
                    interaction_info["food"] = "Alcohol"
                    interactions.append(interaction_info)
            if drug1 == "Simvastatin" and "Grapefruit Juice" in medication_set:
                 if ("Simvastatin", "Grapefruit Juice") in self.knowledge_base:
                    interaction_info = self.knowledge_base[("Simvastatin", "Grapefruit Juice")].copy()
                    interaction_info["drug_1"] = "Simvastatin"
                    interaction_info["food"] = "Grapefruit Juice"
                    interactions.append(interaction_info)

        return interactions

class MedicalLLMSystem:
    def __init__(self):
        self.drug_interaction_module = DrugInteractionModule()

    def _simulate_llm_response(self, prompt):
        # A very simple placeholder for an LLM. In a real application, this would
        # involve calling an actual LLM API or an embedded LLM.
        if "Major Interaction" in prompt:
            return f"WARNING: Critical drug interactions detected. Please review the following before proceeding:\n\n{prompt}\n\nConsult with a pharmacist or senior clinician immediately." 
        elif "Moderate Interaction" in prompt:
            return f"Important drug interactions identified. Consider monitoring or alternative options:\n\n{prompt}\n\nFurther evaluation is recommended." 
        else:
            return f"No significant drug interactions detected based on current knowledge base. Here is the medical context provided:\n\n{prompt}\n\nProceed with standard medical guidelines."

    def process_patient_query(self, patient_medications, patient_query):
        drug_interaction_info = self.drug_interaction_module.get_drug_interactions(patient_medications)

        interaction_context = ""
        if drug_interaction_info:
            interaction_context = "\n\nDrug Interaction Analysis from Specialized Module:\n"
            for interaction in drug_interaction_info:
                interaction_context += f"- Drug 1: {interaction.get('drug_1', 'N/A')}, Drug 2/Food: {interaction.get('drug_2', interaction.get('food', 'N/A'))}, Type: {interaction.get('type', 'N/A')}, Severity: {interaction.get('severity', 'N/A')}, Recommendation: {interaction.get('recommendation', 'N/A')}\n"
        else:
            interaction_context = "\n\nNo significant drug interactions found for the provided medications based on our knowledge base."

        llm_prompt = (
            f"Patient's medications: {', '.join(patient_medications)}\n"
            f"Patient's query: {patient_query}\n"
            f"{interaction_context}\n"
            "Considering the patient's medications and the potential interactions listed above, please provide a comprehensive medical recommendation/summary."
        )

        augmented_response = self._simulate_llm_response(llm_prompt)
        return augmented_response

if __name__ == "__main__":
    medical_system = MedicalLLMSystem()

    # Test Case 1: Major Interaction
    meds1 = ["Warfarin", "Aspirin", "Metformin"]
    query1 = "What are the potential risks with this medication plan?"
    response1 = medical_system.process_patient_query(meds1, query1)
    print("\n--- Test Case 1: Major Interaction ---")
    print(f"Medications: {meds1}")
    print(f"Query: {query1}")
    print("Response:")
    print(response1)

    # Test Case 2: Moderate Interaction (Drug-Food)
    meds2 = ["Metformin", "Lithium", "Ibuprofen", "Alcohol"]
    query2 = "Is this medication combination safe? Are there any dietary restrictions?"
    response2 = medical_system.process_patient_query(meds2, query2)
    print("\n--- Test Case 2: Moderate Interaction (Drug-Food) ---")
    print(f"Medications: {meds2}")
    print(f"Query: {query2}")
    print("Response:")
    print(response2)

    # Test Case 3: No known interactions
    meds3 = ["Paracetamol", "Amoxicillin"]
    query3 = "Are there any interactions between these two drugs?"
    response3 = medical_system.process_patient_query(meds3, query3)
    print("\n--- Test Case 3: No known interactions ---")
    print(f"Medications: {meds3}")
    print(f"Query: {query3}")
    print("Response:")
    print(response3)

    # Test Case 4: Another major interaction
    meds4 = ["Simvastatin", "Grapefruit Juice"]
    query4 = "Can the patient have grapefruit with their medication?"
    response4 = medical_system.process_patient_query(meds4, query4)
    print("\n--- Test Case 4: Another major interaction ---")
    print(f"Medications: {meds4}")
    print(f"Query: {query4}")
    print("Response:")
    print(response4)