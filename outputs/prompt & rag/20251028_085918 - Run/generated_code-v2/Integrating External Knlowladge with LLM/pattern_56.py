MEDICAL_KG = {
    "fever": {"is_symptom_of": ["flu", "common_cold", "malaria"]},
    "cough": {"is_symptom_of": ["flu", "common_cold", "bronchitis"]},
    "headache": {"is_symptom_of": ["flu", "migraine"]},
    "fatigue": {"is_symptom_of": ["flu", "anemia"]},
    "flu": {"has_symptom": ["fever", "cough", "headache", "fatigue"], "has_treatment": ["rest", "antivirals"]},
    "common_cold": {"has_symptom": ["fever", "cough"], "has_treatment": ["rest", "fluids"]},
    "malaria": {"has_symptom": ["fever"], "has_treatment": ["antimalarials"]},
    "migraine": {"has_symptom": ["headache"], "has_treatment": ["painkillers"]},
    "bronchitis": {"has_symptom": ["cough"], "has_treatment": ["antibiotics"]},
    "rest": {"treats": ["flu", "common_cold"]},
    "antivirals": {"treats": ["flu"]}
}

RELATION_PRUNE_PROMPT = "Given the patient symptoms and current reasoning path, identify the most relevant medical relations (e.g., 'is_symptom_of', 'has_symptom', 'has_treatment', 'treats') from the Knowledge Graph. Score them by relevance."
ENTITY_PRUNE_PROMPT = "Based on the selected relations and current patient context, identify and score the most relevant medical entities (diseases, symptoms, treatments) from the Knowledge Graph that contribute to understanding the patient's condition."
REASONING_PROMPT = "Evaluate the current reasoning path. Is there sufficient information to provide a confident medical diagnosis for the patient's condition, or do we need further exploration? Respond with 'SUFFICIENT' or 'INSUFFICIENT'."
GENERATE_PROMPT = "Synthesize a final medical diagnosis and potential treatment recommendations based on the accumulated reasoning path and identified relevant information."

def simulate_llm_response(prompt, context):
    if RELATION_PRUNE_PROMPT in prompt:
        patient_symptoms = context.get("patient_symptoms", [])
        relations = set()
        for symptom in patient_symptoms:
            if symptom in MEDICAL_KG:
                for rel_type in MEDICAL_KG[symptom].keys():
                    relations.add(rel_type)
        return list(relations) if relations else ["is_symptom_of"]
    elif ENTITY_PRUNE_PROMPT in prompt:
        relevant_relations = context.get("relevant_relations", [])
        reasoning_path = context.get("reasoning_path", [])
        entities = set()
        for item in reasoning_path:
            if item in MEDICAL_KG:
                for rel, targets in MEDICAL_KG[item].items():
                    if rel in relevant_relations:
                        entities.update(targets)
        return list(entities) if entities else ["flu", "common_cold"]
    elif REASONING_PROMPT in prompt:
        reasoning_path = context.get("reasoning_path", [])
        if len(reasoning_path) > 3 and any(item in ["flu", "malaria", "bronchitis"] for item in reasoning_path):
            return "SUFFICIENT"
        return "INSUFFICIENT"
    elif GENERATE_PROMPT in prompt:
        reasoning_path = context.get("reasoning_path", [])
        diagnosis = "Uncertain diagnosis. More information needed."
        treatments = []
        if "flu" in reasoning_path:
            diagnosis = "Probable Flu."
            treatments = ["rest", "antivirals"]
        elif "malaria" in reasoning_path:
            diagnosis = "Possible Malaria."
            treatments = ["antimalarials"]
        elif "bronchitis" in reasoning_path:
            diagnosis = "Likely Bronchitis."
            treatments = ["antibiotics"]
        
        if treatments:
            return f"{diagnosis} Recommended treatments: {', '.join(treatments)}."
        return diagnosis
    return ""

def medical_diagnosis_agent(patient_symptoms, medical_question, max_iterations=5):
    reasoning_path = patient_symptoms[:]
    print(f"Initial Symptoms: {patient_symptoms}")

    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1} ---")
        
        # Step 1: Relation Prune
        context_rel_prune = {"patient_symptoms": patient_symptoms, "reasoning_path": reasoning_path}
        relevant_relations = simulate_llm_response(RELATION_PRUNE_PROMPT, context_rel_prune)
        print(f"LLM selected relations: {relevant_relations}")
        
        # Step 2: Entity Prune
        context_entity_prune = {"relevant_relations": relevant_relations, "reasoning_path": reasoning_path}
        new_entities = simulate_llm_response(ENTITY_PRUNE_PROMPT, context_entity_prune)
        print(f"LLM identified new entities: {new_entities}")
        reasoning_path.extend([entity for entity in new_entities if entity not in reasoning_path])
        print(f"Current Reasoning Path: {reasoning_path}")

        # Step 3: Reasoning Evaluation
        context_reasoning = {"reasoning_path": reasoning_path}
        sufficiency = simulate_llm_response(REASONING_PROMPT, context_reasoning)
        print(f"Reasoning sufficiency: {sufficiency}")
        if sufficiency == "SUFFICIENT":
            print("Sufficient information for diagnosis.")
            break
    else:
        print("Maximum iterations reached without sufficient reasoning.")

    # Step 4: Generate Final Answer
    context_generate = {"reasoning_path": reasoning_path}
    final_diagnosis = simulate_llm_response(GENERATE_PROMPT, context_generate)
    return final_diagnosis

if __name__ == "__main__":
    patient1_symptoms = ["fever", "cough", "headache"]
    medical_question1 = "What is the most likely diagnosis for a patient experiencing fever, cough, and headache?"
    diagnosis1 = medical_diagnosis_agent(patient1_symptoms, medical_question1)
    print(f"\nFinal Diagnosis for Patient 1: {diagnosis1}")

    print("\n" + "="*50 + "\n")

    patient2_symptoms = ["headache"]
    medical_question2 = "What could be the cause of persistent headache?"
    diagnosis2 = medical_diagnosis_agent(patient2_symptoms, medical_question2)
    print(f"\nFinal Diagnosis for Patient 2: {diagnosis2}")

    print("\n" + "="*50 + "\n")

    patient3_symptoms = ["cough"]
    medical_question3 = "What is the diagnosis for a patient with a persistent cough?"
    diagnosis3 = medical_diagnosis_agent(patient3_symptoms, medical_question3)
    print(f"\nFinal Diagnosis for Patient 3: {diagnosis3}")
