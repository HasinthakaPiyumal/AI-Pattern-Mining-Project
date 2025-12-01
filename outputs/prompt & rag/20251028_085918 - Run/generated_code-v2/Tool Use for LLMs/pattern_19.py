
import json

def symptom_checker(symptoms: str) -> str:
    """
    Checks given symptoms and suggests potential conditions.
    Input: A comma-separated string of symptoms (e.g., "fever, cough, fatigue").
    Output: A JSON string with a list of potential conditions.
    """
    symptoms_list = [s.strip().lower() for s in symptoms.split(',')]
    
    possible_conditions = []

    if "fever" in symptoms_list and "cough" in symptoms_list and "fatigue" in symptoms_list:
        possible_conditions.append("Flu")
    if "headache" in symptoms_list and "nausea" in symptoms_list:
        possible_conditions.append("Migraine")
    if "abdominal pain" in symptoms_list and "jaundice" in symptoms_list:
        possible_conditions.append("Hepatitis")
    if "chest pain" in symptoms_list and "shortness of breath" in symptoms_list:
        possible_conditions.append("Possible Cardiac Issue")
    if not possible_conditions:
        possible_conditions.append("No common conditions directly match. Further investigation needed.")

    return json.dumps({"potential_conditions": possible_conditions})

def lab_result_interpreter(test_name: str, value: float, unit: str, reference_range: str) -> str:
    """
    Interprets a single lab test result by comparing it to a reference range.
    Input: test_name (str), value (float), unit (str), reference_range (str, e.g., "2.0-8.0").
    Output: A JSON string indicating if the result is normal, low, or high, and potential implications.
    """
    try:
        lower, upper = map(float, reference_range.split('-'))
        status = "Normal"
        implication = ""
        if value < lower:
            status = "Low"
            implication = f"{test_name} is low. Consider nutritional deficiencies or organ dysfunction."
        elif value > upper:
            status = "High"
            implication = f"{test_name} is high. Consider inflammation, infection, or organ stress."
        else:
            implication = f"{test_name} is within the normal range."

        return json.dumps({
            "test_name": test_name,
            "value": value,
            "unit": unit,
            "reference_range": reference_range,
            "status": status,
            "implication": implication
        })
    except ValueError:
        return json.dumps({"error": "Invalid reference range format. Expected 'lower-upper'."})

def medical_imaging_analysis(image_type: str, findings: str) -> str:
    """
    Simulates analysis of a medical imaging report.
    Input: image_type (str, e.g., "X-Ray", "MRI"), findings (str, descriptive text).
    Output: A JSON string summarizing the findings and potential next steps.
    """
    summary = f"Analysis of {image_type} reveals: {findings}."
    next_steps = []

    if "mass" in findings.lower() or "lesion" in findings.lower() or "tumor" in findings.lower():
        next_steps.append("Recommend further investigation (e.g., biopsy, follow-up imaging).")
    if "inflammation" in findings.lower():
        next_steps.append("Consider anti-inflammatory treatment or infection workup.")
    if not next_steps:
        next_steps.append("Findings appear non-specific or require clinical correlation.")

    return json.dumps({
        "image_type": image_type,
        "findings": findings,
        "summary": summary,
        "recommended_next_steps": next_steps
    })

def medical_literature_search(query: str, num_results: int = 3) -> str:
    """
    Simulates searching medical literature for relevant articles.
    Input: query (str, keywords for search), num_results (int, number of top results to return).
    Output: A JSON string with a list of simulated article titles and summaries.
    """
    simulated_articles = {
        "rare autoimmune disease": [
            {"title": "Novel Biomarkers for Autoimmune Encephalitis", "summary": "Discusses new diagnostic markers."}, 
            {"title": "Treatment Strategies for Refractory Autoimmune Conditions", "summary": "Review of advanced therapies."}, 
            {"title": "Case Study: Atypical Presentation of Systemic Lupus Erythematosus", "summary": "Detailed patient case."} 
        ],
        "unexplained fatigue": [
            {"title": "Chronic Fatigue Syndrome: Diagnostic Criteria and Management", "summary": "Guidelines for CFS."}, 
            {"title": "Role of Mitochondrial Dysfunction in Persistent Fatigue", "summary": "Research on cellular energy."}, 
            {"title": "Differential Diagnosis of Unexplained Fatigue", "summary": "Systematic approach to diagnosis."} 
        ],
        "pulmonary fibrosis": [
            {"title": "Idiopathic Pulmonary Fibrosis: Pathogenesis and Current Therapies", "summary": "Overview of IPF."}, 
            {"title": "Novel Anti-fibrotic Agents in Lung Disease", "summary": "Emerging drug treatments."}, 
            {"title": "High-Resolution CT in Diagnosis of Interstitial Lung Diseases", "summary": "Imaging techniques."} 
        ]
    }
    
    results = []
    for keyword, articles in simulated_articles.items():
        if keyword in query.lower():
            results.extend(articles)
    
    if not results:
        results.append({"title": "No direct matches found", "summary": "Try a different query or broaden your search."})

    return json.dumps({"search_results": results[:num_results]})
