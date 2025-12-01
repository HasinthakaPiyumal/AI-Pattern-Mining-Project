import random

def medical_ai_assistant_second_opinion(symptoms: str, medical_history: str, test_results: str) -> dict:
    # Simulate prompt construction
    prompt = (
        f"Patient symptoms: {symptoms}\n"
        f"Patient medical history: {medical_history}\n"
        f"Test results: {test_results}\n\n"
        "Please provide a preliminary diagnostic suggestion, treatment recommendation, and your confidence score from 1 to 10."
    )

    # Simulate LLM Response Generation
    diagnosis_suggestion = ""
    treatment_recommendation = ""
    confidence_score = 0

    # Simple heuristic to influence simulated confidence based on input completeness
    input_completeness = 0
    if symptoms.strip():
        input_completeness += 1
    if medical_history.strip():
        input_completeness += 1
    if test_results.strip():
        input_completeness += 1

    if input_completeness == 3:
        diagnosis_suggestion = "Possible viral infection (e.g., influenza)."
        treatment_recommendation = "Recommend rest, hydration, and symptomatic relief. Consider antiviral if within onset window."
        confidence_score = random.randint(7, 10) # Higher confidence for complete data
    elif input_completeness >= 1:
        diagnosis_suggestion = "Non-specific febrile illness."
        treatment_recommendation = "Monitor symptoms closely. Suggest general supportive care."
        confidence_score = random.randint(4, 6) # Medium confidence for some data
    else:
        diagnosis_suggestion = "Insufficient information for a definitive diagnosis."
        treatment_recommendation = "Advise further diagnostic tests and detailed history taking."
        confidence_score = random.randint(1, 3) # Lower confidence for minimal data
    
    # Simulate LLM response string format
    simulated_llm_response = (
        f"Diagnosis: {diagnosis_suggestion}\n"
        f"Treatment: {treatment_recommendation}\n"
        f"Confidence Score: {confidence_score}/10"
    )

    # Simulate Output Parsing
    parsed_diagnosis = ""
    parsed_treatment = ""
    parsed_confidence = 0

    lines = simulated_llm_response.split('\n')
    for line in lines:
        if line.startswith("Diagnosis:"):
            parsed_diagnosis = line.replace("Diagnosis:", "").strip()
        elif line.startswith("Treatment:"):
            parsed_treatment = line.replace("Treatment:", "").strip()
        elif line.startswith("Confidence Score:"):
            try:
                score_str = line.replace("Confidence Score:", "").replace("/10", "").strip()
                parsed_confidence = int(score_str)
            except ValueError:
                parsed_confidence = 0 # Default if parsing fails

    # Return structured output
    return {
        "diagnosis": parsed_diagnosis,
        "treatment": parsed_treatment,
        "confidence_score": parsed_confidence
    }

