def load_mock_image_data(image_id):
    if image_id == "img_001":
        return "X-ray_image_data_showing_lung_opacity"
    return "generic_image_data"

def load_mock_patient_history(patient_id):
    if patient_id == "pat_001":
        return "Patient presents with persistent cough, shortness of breath, and fatigue for 2 weeks. No fever. Smoker."
    return "generic_patient_history"

def extract_conceptual_visual_features(image_data):
    if "lung_opacity" in image_data:
        return [0.8, 0.2, 0.1]  # Placeholder for features like opacity, inflammation
    return [0.1, 0.1, 0.1]

def extract_conceptual_textual_features(text_data):
    if "cough" in text_data and "shortness of breath" in text_data:
        return [0.7, 0.3, 0.1]  # Placeholder for features like respiratory distress, inflammation markers
    return [0.2, 0.2, 0.2]

def decompose_diagnostic_problem(symptoms_query):
    sub_questions = []
    if "cough" in symptoms_query and "shortness of breath" in symptoms_query:
        sub_questions.append("Is there evidence of lung infection or inflammation?")
        sub_questions.append("Are there any signs of chronic lung disease?")
        sub_questions.append("What is the most likely primary diagnosis?")
    else:
        sub_questions.append("General diagnostic inquiry.")
    return sub_questions

def integrate_multimodal_evidence(visual_features, textual_features, sub_question):
    combined_evidence = {}
    combined_evidence["visual_input"] = visual_features
    combined_evidence["textual_input"] = textual_features
    combined_evidence["sub_question"] = sub_question

    if "lung infection or inflammation" in sub_question:
        if visual_features[0] > 0.7 and textual_features[0] > 0.6:
            combined_evidence["integrated_finding"] = "Strong evidence of lung inflammation/infection."
        else:
            combined_evidence["integrated_finding"] = "Weak evidence of lung inflammation/infection."
    elif "chronic lung disease" in sub_question:
        if visual_features[1] > 0.5 and textual_features[1] > 0.2:
            combined_evidence["integrated_finding"] = "Some indication of chronic changes."
        else:
            combined_evidence["integrated_finding"] = "No clear signs of chronic disease."
    else:
        combined_evidence["integrated_finding"] = "General integrated finding."
    return combined_evidence

def build_conceptual_diagnostic_graph(evidence_list):
    graph = {
        "nodes": [],
        "edges": []
    }
    node_id_counter = 0
    node_map = {}

    for evidence in evidence_list:
        finding_node = f"finding_{node_id_counter}"
        graph["nodes"].append({"id": finding_node, "label": evidence["integrated_finding"]})
        node_map[evidence["integrated_finding"]] = finding_node
        node_id_counter += 1

        # Simulate connections based on conceptual reasoning
        if "Strong evidence of lung inflammation/infection." == evidence["integrated_finding"]:
            if "Pneumonia" not in node_map:
                pneumonia_node = f"diagnosis_{node_id_counter}"
                graph["nodes"].append({"id": pneumonia_node, "label": "Pneumonia"})
                node_map["Pneumonia"] = pneumonia_node
                node_id_counter += 1
            graph["edges"].append({"source": finding_node, "target": node_map["Pneumonia"], "type": "supports"})

        if "Some indication of chronic changes." == evidence["integrated_finding"]:
            if "COPD" not in node_map:
                copd_node = f"diagnosis_{node_id_counter}"
                graph["nodes"].append({"id": copd_node, "label": "COPD"})
                node_map["COPD"] = copd_node
                node_id_counter += 1
            graph["edges"].append({"source": finding_node, "target": node_map["COPD"], "type": "supports"})

    return graph

def reason_for_diagnosis(diagnostic_graph, sub_question_results):
    primary_diagnosis = "Undetermined"
    differential_diagnoses = []
    explanation_steps = []

    for node in diagnostic_graph["nodes"]:
        if "Pneumonia" == node["label"]:
            primary_diagnosis = "Pneumonia (likely bacterial)"
            explanation_steps.append("Evidence strongly points to acute lung inflammation consistent with pneumonia.")
        elif "COPD" == node["label"]:
            if primary_diagnosis == "Undetermined": # Prioritize acute over chronic for primary
                differential_diagnoses.append("COPD exacerbation (consider given smoking history).")
            else:
                differential_diagnoses.insert(0, "COPD exacerbation (consider given smoking history).")
            explanation_steps.append("Patient's smoking history and some chronic changes suggest underlying COPD.")

    if not explanation_steps:
        explanation_steps.append("Further investigation needed for a definitive diagnosis.")

    return primary_diagnosis, differential_diagnoses, explanation_steps

def generate_conceptual_visual_explanation(original_image_data, diagnosis, relevant_evidence):
    visual_explanation_description = ""
    if "Pneumonia" in diagnosis and "lung_opacity" in original_image_data:
        visual_explanation_description = "Highlighted region in upper right lung field shows increased opacity, consistent with consolidation seen in pneumonia."
    elif "COPD" in diagnosis:
        visual_explanation_description = "Subtle indications of hyperinflation and flattened diaphragm noted, often associated with COPD."
    else:
        visual_explanation_description = "No specific visual anomalies highlighted for this diagnosis."
    return visual_explanation_description

def format_diagnostic_report(diagnosis, explanation, visual_explanation_description):
    report = f"*** Multimodal AI Diagnostic Report ***\n\n"
    report += f"Primary Diagnosis: {diagnosis}\n\n"
    report += f"Reasoning Steps:\n"
    for step in explanation:
        report += f"- {step}\n"
    report += f"\nVisual Explanation:\n"
    report += f"{visual_explanation_description}\n\n"
    report += "Disclaimer: This is an AI-generated report and should not replace professional medical advice."
    return report

def run_diagnostic_assistant(image_id, patient_id, symptoms_query):
    # Data Ingestion
    image_data = load_mock_image_data(image_id)
    patient_history_text = load_mock_patient_history(patient_id)

    # Multimodal Feature Extraction
    visual_features = extract_conceptual_visual_features(image_data)
    textual_features = extract_conceptual_textual_features(patient_history_text)

    # Structured Reasoning Engine
    sub_questions = decompose_diagnostic_problem(symptoms_query)
    
    integrated_evidences = []
    sub_question_results = {}
    for i, sq in enumerate(sub_questions):
        evidence = integrate_multimodal_evidence(visual_features, textual_features, sq)
        integrated_evidences.append(evidence)
        sub_question_results[f"Sub-question {i+1}: {sq}"] = evidence["integrated_finding"]

    diagnostic_graph = build_conceptual_diagnostic_graph(integrated_evidences)
    primary_diagnosis, differential_diagnoses, reasoning_steps = reason_for_diagnosis(diagnostic_graph, sub_question_results)

    # Explainability and Output
    visual_explanation = generate_conceptual_visual_explanation(image_data, primary_diagnosis, integrated_evidences)
    
    full_explanation = reasoning_steps + [f"Differential Diagnoses: {', '.join(differential_diagnoses)}"] if differential_diagnoses else reasoning_steps

    diagnostic_report = format_diagnostic_report(primary_diagnosis, full_explanation, visual_explanation)

    return diagnostic_report

if __name__ == "__main__":
    # Example Usage
    report = run_diagnostic_assistant("img_001", "pat_001", "persistent cough and shortness of breath")
    print(report)

    print("\n" + "="*50 + "\n")

    report2 = run_diagnostic_assistant("img_002", "pat_002", "general checkup")
    print(report2)