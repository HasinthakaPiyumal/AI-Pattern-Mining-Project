MEDICAL_KG = {
    "symptom:fever": {
        "indicates": "disease:flu"
    },
    "symptom:cough": {
        "indicates": "disease:flu",
        "alleviated_by": "treatment:cough_syrup"
    },
    "symptom:sore_throat": {
        "indicates": "disease:cold"
    },
    "disease:flu": {
        "treatable_by": "treatment:rest",
        "caused_by": "virus:influenza"
    },
    "disease:cold": {
        "treatable_by": "treatment:fluids",
        "caused_by": "virus:rhinovirus"
    },
    "treatment:rest": {
        "part_of": "recovery_plan"
    },
    "treatment:fluids": {
        "part_of": "recovery_plan"
    },
    "medication:paracetamol": {
        "treats_symptom": "symptom:fever"
    }
}

def get_diagnosis_and_reasoning(symptoms):
    """
    Simulates a diagnosis based on symptoms and generates an explicit reasoning path.
    """
    reasoning_path = []
    possible_diseases = set()
    recommended_treatments = set()

    for symptom in symptoms:
        symptom_key = f"symptom:{symptom.lower()}"
        if symptom_key in MEDICAL_KG:
            knowledge = MEDICAL_KG[symptom_key]
            for relation, target in knowledge.items():
                reasoning_path.append((symptom_key, relation, target))
                
                if relation == "indicates":
                    possible_diseases.add(target)
                elif relation == "alleviated_by":
                    recommended_treatments.add(target)

    final_diagnosis = "Unknown Condition"
    if possible_diseases:
        # Simple logic: if multiple diseases, just pick one for this simulation
        final_diagnosis = list(possible_diseases)[0]
        reasoning_path.append((final_diagnosis, "is_a", "diagnosis"))

        # Add treatments for the diagnosed disease if available
        if final_diagnosis in MEDICAL_KG:
            for relation, target in MEDICAL_KG[final_diagnosis].items():
                if relation == "treatable_by":
                    recommended_treatments.add(target)
                    reasoning_path.append((final_diagnosis, relation, target))
    
    # Add reasoning for symptom relief if medications are known
    for med, details in MEDICAL_KG.items():
        if med.startswith("medication:"):
            for rel, target_symptom in details.items():
                if rel == "treats_symptom" and target_symptom in [f"symptom:{s.lower()}" for s in symptoms]:
                    recommended_treatments.add(med)
                    reasoning_path.append((med, rel, target_symptom))


    return final_diagnosis, list(recommended_treatments), reasoning_path

def correct_knowledge_graph(entity1, relation, entity2, new_entity2=None, remove=False):
    """
    Allows a user or expert to correct the knowledge graph.
    If remove is True, the triple is removed. If new_entity2 is provided, the target of the triple is updated.
    """
    if entity1 not in MEDICAL_KG:
        print(f"Error: Entity '{entity1}' not found in KG for correction.")
        return

    if remove:
        if relation in MEDICAL_KG[entity1] and MEDICAL_KG[entity1][relation] == entity2:
            del MEDICAL_KG[entity1][relation]
            print(f"Removed knowledge: ({entity1}, {relation}, {entity2})")
        else:
            print(f"Knowledge ({entity1}, {relation}, {entity2}) not found for removal.")
    elif new_entity2:
        if relation in MEDICAL_KG[entity1] and MEDICAL_KG[entity1][relation] == entity2:
            MEDICAL_KG[entity1][relation] = new_entity2
            print(f"Updated knowledge: ({entity1}, {relation}, {entity2}) -> ({entity1}, {relation}, {new_entity2})")
        else:
            print(f"Knowledge ({entity1}, {relation}, {entity2}) not found for update. Adding new: ({entity1}, {relation}, {new_entity2})")
            MEDICAL_KG[entity1][relation] = new_entity2
    else:
        print("No action specified for correction (neither remove nor new_entity2 provided).")


# --- Demonstration --- 
if __name__ == "__main__":
    print("--- Initial Diagnosis ---")
    patient_symptoms_1 = ["fever", "cough"]
    diagnosis_1, treatments_1, reasoning_path_1 = get_diagnosis_and_reasoning(patient_symptoms_1)

    print(f"Patient Symptoms: {', '.join(patient_symptoms_1)}")
    print(f"Diagnosed Condition: {diagnosis_1}")
    print(f"Recommended Treatments: {', '.join(treatments_1)}")
    print("Explicit Reasoning Path:")
    for step in reasoning_path_1:
        print(f"  - {step[0]} --({step[1]})--> {step[2]}")

    print("\n--- Expert Review and Correction ---")
    print("An expert identifies that 'symptom:cough' also indicates 'disease:bronchitis' in some cases,")
    print("and the current KG is missing a specific treatment for 'disease:flu' beyond 'rest' such as 'antivirals'.")

    # Simulate correction 1: Add a new indication for cough
    correct_knowledge_graph("symptom:cough", "indicates", "disease:bronchitis", new_entity2="disease:bronchitis")
    # The previous call would effectively just add/update. A better way to add a *new* relation from scratch if it doesn't exist
    if "symptom:cough" not in MEDICAL_KG:
        MEDICAL_KG["symptom:cough"] = {}
    MEDICAL_KG["symptom:cough"]["can_also_indicate"] = "disease:bronchitis"
    print("Added: (symptom:cough, can_also_indicate, disease:bronchitis)")

    # Simulate correction 2: Update an existing treatment for flu
    # Let's say we want to add a specific antiviral instead of just 'rest'
    # For this demo, let's assume 'treatable_by' can have multiple values or be updated.
    # In a real KG, this might be more complex (e.g., list of treatments).
    correct_knowledge_graph("disease:flu", "treatable_by", "treatment:rest", new_entity2="treatment:antivirals")

    print("\n--- Re-Diagnosis After Correction ---")
    diagnosis_2, treatments_2, reasoning_path_2 = get_diagnosis_and_reasoning(patient_symptoms_1)

    print(f"Patient Symptoms: {', '.join(patient_symptoms_1)}")
    print(f"Diagnosed Condition: {diagnosis_2}")
    print(f"Recommended Treatments: {', '.join(treatments_2)}")
    print("Explicit Reasoning Path:")
    for step in reasoning_path_2:
        print(f"  - {step[0]} --({step[1]})--> {step[2]}")

    print("\n--- Another Correction: Removing outdated knowledge ---")
    print("An expert decides 'symptom:sore_throat' no longer strongly indicates 'disease:cold' in the current context.")
    correct_knowledge_graph("symptom:sore_throat", "indicates", "disease:cold", remove=True)

    print("\n--- Final Diagnosis After Removal ---")
    patient_symptoms_3 = ["sore_throat"]
    diagnosis_3, treatments_3, reasoning_path_3 = get_diagnosis_and_reasoning(patient_symptoms_3)

    print(f"Patient Symptoms: {', '.join(patient_symptoms_3)}")
    print(f"Diagnosed Condition: {diagnosis_3}")
    print(f"Recommended Treatments: {', '.join(treatments_3)}")
    print("Explicit Reasoning Path:")
    for step in reasoning_path_3:
        print(f"  - {step[0]} --({step[1]})--> {step[2]}")
