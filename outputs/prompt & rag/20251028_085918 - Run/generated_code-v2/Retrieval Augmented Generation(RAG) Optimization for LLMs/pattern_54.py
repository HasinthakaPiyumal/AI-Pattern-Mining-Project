MEDICAL_DATA = {
    "hypertension": "Hypertension (high blood pressure) is a common condition. Treatment often involves lifestyle changes like diet and exercise, and medications such as ACE inhibitors, ARBs, diuretics, or beta-blockers. Regular monitoring is crucial.",
    "diabetes_type2": "Type 2 diabetes is characterized by insulin resistance. Management includes diet control, exercise, and medications like metformin, sulfonylureas, or insulin therapy. HbA1c monitoring is essential.",
    "metformin_interactions": "Metformin can interact with contrast dyes used in imaging. It should be temporarily discontinued before and after such procedures. Lactic acidosis is a rare but serious side effect.",
    "aspirin_guidelines": "Low-dose aspirin is recommended for secondary prevention of cardiovascular events in patients with established heart disease. It may also be considered for primary prevention in select high-risk individuals, but risks of bleeding should be weighed.",
    "patient_x_history_cardiac": "Patient X has a history of myocardial infarction (MI) 3 years ago. Current medications include atorvastatin and metoprolol. No known drug allergies.",
    "patient_y_history_diabetic": "Patient Y was diagnosed with Type 2 diabetes 5 years ago. Currently on metformin. Recent HbA1c is 7.5%. No significant complications reported yet.",
    "dietary_recommendations_hypertension": "For hypertension, a DASH diet (Dietary Approaches to Stop Hypertension) is highly effective. This emphasizes fruits, vegetables, whole grains, and low-fat dairy products, while reducing saturated and total fats, cholesterol, and sodium.",
    "exercise_guidelines_diabetes": "Regular physical activity, including both aerobic and strength training, is vital for Type 2 diabetes management. Aim for at least 150 minutes of moderate-intensity aerobic exercise per week."
}

def retrieve_evidence(query, patient_history=None):
    """
    Simulates retrieving relevant medical evidence based on a query and patient history.
    This is a basic keyword-based retrieval and consolidation.
    """
    retrieved_info = []
    query_keywords = query.lower().split()

    # Search in general medical data
    for key, value in MEDICAL_DATA.items():
        if any(keyword in key.lower() or keyword in value.lower() for keyword in query_keywords):
            retrieved_info.append(value)
            # Basic consolidation logic for linking related entities/contexts
            if "metformin" in key.lower() and ("diabetes" in query.lower() or (patient_history and "diabetes" in patient_history.lower())):
                if MEDICAL_DATA.get("metformin_interactions") not in retrieved_info:
                     retrieved_info.append(MEDICAL_DATA.get("metformin_interactions", ""))
            if "hypertension" in key.lower() and ("diet" in query.lower() or (patient_history and "hypertension" in patient_history.lower() and "diet" in query.lower())):
                 if MEDICAL_DATA.get("dietary_recommendations_hypertension") not in retrieved_info:
                     retrieved_info.append(MEDICAL_DATA.get("dietary_recommendations_hypertension", ""))
            if "diabetes" in key.lower() and ("exercise" in query.lower() or (patient_history and "diabetes" in patient_history.lower() and "exercise" in query.lower())):
                if MEDICAL_DATA.get("exercise_guidelines_diabetes") not in retrieved_info:
                    retrieved_info.append(MEDICAL_DATA.get("exercise_guidelines_diabetes", ""))

    # Augment with patient-specific history if provided and relevant to the query
    if patient_history:
        patient_history_lower = patient_history.lower()
        if "cardiac" in patient_history_lower or "myocardial infarction" in patient_history_lower:
            if MEDICAL_DATA.get("patient_x_history_cardiac") not in retrieved_info:
                retrieved_info.append(MEDICAL_DATA.get("patient_x_history_cardiac", ""))
        if "diabetes" in patient_history_lower or "type 2 diabetes" in patient_history_lower:
            if MEDICAL_DATA.get("patient_y_history_diabetic") not in retrieved_info:
                retrieved_info.append(MEDICAL_DATA.get("patient_y_history_diabetic", ""))

    # Simple pruning/deduplication
    unique_info = list(dict.fromkeys(retrieved_info)) # Maintain order while deduplicating

    # Form a consolidated evidence chain (simple concatenation)
    consolidated_evidence = "\n".join(unique_info)
    return consolidated_evidence if consolidated_evidence.strip() else "No specific relevant medical evidence found."

def emulate_llm_response(prompt):
    """
    Emulates an LLM's response based on the provided prompt.
    This function will try to incorporate elements from the context if present,
    demonstrating how RAG helps ground the response.
    """
    response_template = "Based on the provided medical context and patient information, the recommended course of action is:"

    # Very basic pattern matching to show grounding
    if "hypertension" in prompt.lower() and "dash diet" in prompt.lower():
        return f"{response_template} Implement lifestyle changes, specifically recommending a DASH diet (emphasizing fruits, vegetables, whole grains, low-fat dairy, and reduced sodium) along with regular exercise. Consider medications such as ACE inhibitors or ARBs, with ongoing monitoring of blood pressure. Regular follow-ups are crucial for managing hypertension effectively."
    elif "type 2 diabetes" in prompt.lower() and "metformin" in prompt.lower():
         if "contrast dyes" in prompt.lower() or "metformin_interactions" in prompt.lower():
              return f"{response_template} For Type 2 diabetes managed with metformin, continue lifestyle modifications (diet, exercise). Crucially, if imaging with contrast dyes is planned, metformin should be temporarily discontinued to prevent lactic acidosis. Regular HbA1c monitoring is advised. Adjust metformin dosage or add alternative agents if glycemic targets are not met."
         else:
              return f"{response_template} For Type 2 diabetes managed with metformin, continue lifestyle modifications (diet, exercise) and monitor HbA1c regularly. If HbA1c remains elevated, consider increasing metformin dosage or adding another oral antidiabetic agent or insulin therapy, following clinical guidelines."
    elif "myocardial infarction" in prompt.lower() and "aspirin" in prompt.lower() and "patient x" in prompt.lower():
        return f"{response_template} For Patient X, with a history of myocardial infarction, low-dose aspirin is strongly recommended for secondary prevention of cardiovascular events. Continue current medications like atorvastatin and metoprolol. Ensure regular cardiovascular risk assessment and management."
    elif "no specific relevant medical evidence found" in prompt.lower():
        return "Given the limited specific medical evidence found for the query and patient history, a comprehensive consultation with a medical professional is advised for a definitive diagnosis and treatment plan."
    else:
        return f"{response_template} Further detailed patient assessment and consultation with specialists are recommended to formulate a precise treatment plan, as the current information requires broader medical context for specific recommendations."

def run_clinical_decision_support_system():
    """
    Main function to simulate the Clinical Decision Support System.
    """
    print("--- Clinical Decision Support System (RAG Enhanced) ---")
    print("This system provides personalized treatment recommendations using Retrieval-Augmented Generation.")

    patient_condition = input("\nEnter patient's primary condition/query (e.g., 'treatment for hypertension', 'diabetes management'): ")
    patient_history = input("Enter relevant patient medical history (e.g., 'Patient X with cardiac history', 'Patient Y with diabetes and recent HbA1c'): ")

    # 1. Retrieval (Knowledge Consolidator Module)
    print("\n--- Step 1: Retrieving and Consolidating Medical Evidence ---")
    consolidated_evidence = retrieve_evidence(patient_condition, patient_history)
    print(f"Consolidated Evidence:\n{consolidated_evidence}")

    # 2. LLM Augmentation
    print("\n--- Step 2: Augmenting LLM with Consolidated Evidence ---")
    prompt_for_llm = f"""
    You are an AI assistant providing medical insights. Generate a personalized treatment recommendation
    based on the following patient details and retrieved medical evidence. Be factual, avoid hallucination,
    and ground your response ONLY in the provided evidence.

    Patient Condition/Query: {patient_condition}
    Patient Medical History: {patient_history}

    Retrieved Medical Evidence:
    {consolidated_evidence}

    Personalized Treatment Recommendation:
    """

    # 3. LLM Generation
    print("\n--- Step 3: Generating Recommendation via LLM ---")
    llm_recommendation = emulate_llm_response(prompt_for_llm)

    # 4. Output
    print("\n--- Final Personalized Treatment Recommendation ---")
    print(llm_recommendation)

if __name__ == "__main__":
    run_clinical_decision_support_system()