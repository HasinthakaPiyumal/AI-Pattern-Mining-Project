class ClinicalKnowledgeRetrievalModule:
    def __init__(self):
        self.medical_guidelines_db = {
            "hypertension_treatment": "Adults with hypertension should aim for a blood pressure below 130/80 mmHg. Lifestyle modifications include diet, exercise, and reduced sodium intake. Medications may include ACE inhibitors, ARBs, diuretics, or calcium channel blockers.",
            "diabetes_management_type2": "Type 2 diabetes management involves blood glucose monitoring, healthy eating, regular physical activity, and potentially medication (e.g., metformin) or insulin therapy. Regular check-ups for complications are crucial.",
            "asthma_exacerbation": "Acute asthma exacerbations require bronchodilators (e.g., albuterol) and systemic corticosteroids. Oxygen therapy may be needed. Monitor peak flow and consider hospitalization for severe cases."
        }
        self.drug_interactions_db = [
            {"drug1": "Warfarin", "drug2": "Aspirin", "interaction": "Increased risk of bleeding."},
            {"drug1": "Metformin", "drug2": "Iodinated contrast media", "interaction": "Risk of lactic acidosis."},
            {"drug1": "Simvastatin", "drug2": "Grapefruit Juice", "interaction": "Increased risk of muscle problems (myopathy/rhabdomyolysis)."},
            {"drug1": "Warfarin", "drug2": "Antibiotics (e.g., Ciprofloxacin)", "interaction": "Increased anticoagulant effect of Warfarin."}
        ]
        self.patient_history_db = {
            "P1001": "Patient P1001 has a history of Type 2 Diabetes diagnosed 5 years ago, currently on Metformin. Also has mild hypertension controlled with Lisinopril. No known drug allergies.",
            "P1002": "Patient P1002 presented with recurrent asthma exacerbations. Diagnosed with moderate persistent asthma. Uses albuterol for rescue and inhaled corticosteroids daily. No other significant medical history."
        }

    def _retrieve_medical_guidelines(self, query: str) -> str:
        found_guidelines = []
        query_lower = query.lower()
        for topic, guideline_text in self.medical_guidelines_db.items():
            if query_lower in topic.lower() or query_lower in guideline_text.lower():
                found_guidelines.append(f"- {topic.replace('_', ' ').title()}: {guideline_text}")
        return "\n".join(found_guidelines) if found_guidelines else "No specific guidelines found."

    def _retrieve_drug_interactions(self, drugs: list[str]) -> str:
        found_interactions = []
        drugs_lower = [d.lower() for d in drugs]

        for interaction_entry in self.drug_interactions_db:
            drug1_lower = interaction_entry["drug1"].lower()
            drug2_lower = interaction_entry["drug2"].lower()
            interaction_text = interaction_entry["interaction"]

            if len(drugs_lower) == 1:
                if drugs_lower[0] in [drug1_lower, drug2_lower]:
                    found_interactions.append(f"- {interaction_entry['drug1']} and {interaction_entry['drug2']}: {interaction_text}")
            elif len(drugs_lower) > 1:
                if (drug1_lower in drugs_lower and drug2_lower in drugs_lower):
                    found_interactions.append(f"- {interaction_entry['drug1']} and {interaction_entry['drug2']}: {interaction_text}")

        return "\n".join(found_interactions) if found_interactions else "No specific drug interactions found."

    def _retrieve_patient_history(self, patient_id: str) -> str:
        history = self.patient_history_db.get(patient_id, "Patient ID not found.")
        return f"Patient History ({patient_id}): {history}"

    def get_clinical_context(self, query: str, patient_id: str = None, drugs: list[str] = None) -> str:
        context_parts = []
        context_parts.append(f"--- Clinical Context for Query: \"{query}\" ---")

        guidelines = self._retrieve_medical_guidelines(query)
        if guidelines != "No specific guidelines found.":
            context_parts.append("\nMedical Guidelines:\n" + guidelines)

        if patient_id:
            history = self._retrieve_patient_history(patient_id)
            context_parts.append("\n" + history)

        if drugs:
            interactions = self._retrieve_drug_interactions(drugs)
            if interactions != "No specific drug interactions found.":
                context_parts.append("\nDrug Interactions:\n" + interactions)

        context_parts.append("--- End of Clinical Context ---")

        return "\n".join(context_parts)

if __name__ == "__main__":
    retrieval_module = ClinicalKnowledgeRetrievalModule()

    print("\n--- Scenario 1: General Query for Hypertension Treatment ---")
    context1 = retrieval_module.get_clinical_context(query="hypertension treatment")
    print(context1)
    # LLM would receive context1 and generate a response based on hypertension treatment guidelines.

    print("\n--- Scenario 2: Patient-Specific Query with Drug Information ---")
    context2 = retrieval_module.get_clinical_context(query="diabetes management", patient_id="P1001", drugs=["Metformin", "Lisinopril"])
    print(context2)
    # LLM would receive context2 and provide a personalized diabetes management plan, considering patient history and potential drug interactions.

    print("\n--- Scenario 3: Drug Interaction Check ---")
    context3 = retrieval_module.get_clinical_context(query="Warfarin interactions", drugs=["Warfarin", "Aspirin"])
    print(context3)
    # LLM would use context3 to alert about potential bleeding risks and suggest alternatives or monitoring.

    print("\n--- Scenario 4: Single Drug Interaction Check ---")
    context4 = retrieval_module.get_clinical_context(query="Simvastatin concerns", drugs=["Simvastatin"])
    print(context4)
    # LLM would use context4 to provide information about potential side effects or interactions of Simvastatin.

    print("\n--- Scenario 5: Query for unknown topic ---")
    context5 = retrieval_module.get_clinical_context(query="gastric bypass aftercare")
    print(context5)
    # LLM would note the lack of specific guidelines from the module and rely on its general knowledge or ask for more specific input.

    print("\n--- Conceptual LLM Integration Example ---")
    # Imagine an LLM function that takes a raw user query and the retrieved context
    def llm_reasoning_function(user_query: str, clinical_context: str) -> str:
        # In a real application, this would involve sending both the user_query and clinical_context
        # to an LLM API (e.g., OpenAI, Gemini, Hugging Face model).
        # The LLM would then use the clinical_context to ground its response.
        # For this example, we'll just simulate the LLM's understanding.
        print(f"LLM received user query: '{user_query}'")
        print("LLM processing with provided clinical context...")
        if "No specific guidelines found." in clinical_context and "Patient ID not found." in clinical_context and "No specific drug interactions found." in clinical_context:
            return f"Based on available general medical knowledge and without specific retrieved context, for '{user_query}', I can provide general information."
        else:
            return f"Based on the provided clinical context for '{user_query}', the LLM can generate an informed response. For instance, it could synthesize guidelines, patient history, and drug interaction warnings.\n\n[LLM Generated Response incorporating context]"

    user_query_llm = "What is the best approach for a diabetic patient on Metformin experiencing asthma exacerbations?"
    context_for_llm = retrieval_module.get_clinical_context(
        query="diabetes asthma", 
        patient_id="P1001", 
        drugs=["Metformin", "albuterol", "Lisinopril"]
    )
    print(f"\n--- Context provided to LLM ---\n{context_for_llm}")
    llm_response = llm_reasoning_function(user_query_llm, context_for_llm)
    print(llm_response)

    user_query_no_context = "Tell me about quantum physics."
    context_no_relevance = retrieval_module.get_clinical_context(query="quantum physics")
    print(f"\n--- Context provided to LLM ---\n{context_no_relevance}")
    llm_response_no_relevance = llm_reasoning_function(user_query_no_context, context_no_relevance)
    print(llm_response_no_relevance)
