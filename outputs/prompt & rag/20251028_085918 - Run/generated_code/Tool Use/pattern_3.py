import re

# --- Simulated Specialized Tools ---

def medical_imaging_analysis_tool(image_report: str) -> str:
    """
    Simulates a medical imaging analysis tool.
    In a real system, this would involve computer vision models.
    """
    if "infiltrate" in image_report.lower() or "consolidation" in image_report.lower():
        return "Imaging suggests potential pneumonia or other lung infection."
    elif "fracture" in image_report.lower():
        return "Imaging indicates a bone fracture."
    else:
        return "Imaging report reviewed, no specific abnormalities immediately identified relevant to current query."

def lab_test_interpretation_tool(lab_results: dict) -> str:
    """
    Simulates a lab test interpretation tool.
    In a real system, this would involve a knowledge base and potentially ML models.
    """
    interpretations = []
    if "WBC" in lab_results and lab_results["WBC"] > 10000:
        interpretations.append(f"Elevated White Blood Cell count ({lab_results['WBC']}), suggesting an infection or inflammatory process.")
    if "Hemoglobin" in lab_results and lab_results["Hemoglobin"] < 12:
        interpretations.append(f"Low Hemoglobin ({lab_results['Hemoglobin']}), indicating potential anemia.")
    if "Creatinine" in lab_results and lab_results["Creatinine"] > 1.2:
        interpretations.append(f"Elevated Creatinine ({lab_results['Creatinine']}), indicating potential kidney dysfunction.")
    
    if not interpretations:
        return "Lab results reviewed, no significant deviations found."
    return " ".join(interpretations)

def drug_interaction_checking_tool(medications: list, conditions: list) -> str:
    """
    Simulates a drug interaction checking tool.
    In a real system, this would query comprehensive drug databases.
    """
    interactions = []
    if "Warfarin" in medications:
        if "aspirin" in [m.lower() for m in medications] or any(c in [co.lower() for co in conditions] for c in ["bleeding disorder", "ulcer"]):
            interactions.append("Caution: Warfarin interacts with Aspirin (increased bleeding risk) and conditions like bleeding disorders.")
        if any(c in [co.lower() for co in conditions] for c in ["liver disease", "kidney disease"]):
            interactions.append("Caution: Warfarin dosage may need adjustment in patients with liver or kidney disease.")
    
    if not interactions:
        return "No significant drug interactions or contraindications found based on provided information."
    return " ".join(interactions)

def clinical_guideline_retrieval_tool(query: str) -> str:
    """
    Simulates a Retrieval-Augmented Generation (RAG) system for clinical guidelines.
    In a real system, this would use a vector database and embedding models.
    """
    guidelines_db = {
        "chest pain": "For acute chest pain with suspected cardiac origin, consider ECG, cardiac enzymes, and imaging. Refer to ACC/AHA guidelines.",
        "pneumonia": "Community-acquired pneumonia treatment often involves antibiotics like Azithromycin or Doxycycline. Consider severity assessment tools like CURB-65.",
        "warfarin management": "Regular INR monitoring is crucial for patients on Warfarin. Adjust dosage based on therapeutic range and patient specific factors. Avoid large fluctuations in Vitamin K intake.",
        "elevated wbc": "Elevated WBC typically indicates infection, inflammation, or stress. Further investigation may include differential count and clinical correlation."
    }
    
    # Simple keyword matching for demonstration
    for keyword, guideline in guidelines_db.items():
        if keyword in query.lower():
            return f"Relevant Clinical Guideline: {guideline}"
    
    return "No specific clinical guidelines directly matching your query were found in our simplified database."

# --- Orchestrator LLM Simulation ---

class MedicalDiagnosticAssistant:
    def __init__(self):
        self.tools = {
            "imaging_analysis": medical_imaging_analysis_tool,
            "lab_interpretation": lab_test_interpretation_tool,
            "drug_interaction": drug_interaction_checking_tool,
            "guideline_retrieval": clinical_guideline_retrieval_tool,
        }

    def _parse_query(self, query: str) -> dict:
        """
        Simulates LLM's ability to parse queries and extract relevant information.
        This is a simplified implementation using regex and keyword matching.
        """
        parsed_info = {
            "symptoms": [],
            "image_report": None,
            "lab_results": {},
            "medications": [],
            "conditions": [],
            "guideline_query": None
        }

        # Extract symptoms
        symptom_matches = re.findall(r"(chest pain|fever|cough|fatigue)", query, re.IGNORECASE)
        parsed_info["symptoms"] = list(set(symptom_matches))

        # Extract image report
        image_match = re.search(r"X-ray shows ([^.,]+)|CT scan shows ([^.,]+)|imaging report states ([^.,]+)", query, re.IGNORECASE)
        if image_match:
            parsed_info["image_report"] = image_match.group(1) or image_match.group(2) or image_match.group(3)

        # Extract lab results (simplified)
        wbc_match = re.search(r"WBC (\d{1,5}(?:\.\d+)?)", query, re.IGNORECASE)
        if wbc_match: parsed_info["lab_results"]["WBC"] = float(wbc_match.group(1))
        
        hgb_match = re.search(r"Hemoglobin (\d{1,2}(?:\.\d+)?)", query, re.IGNORECASE)
        if hgb_match: parsed_info["lab_results"]["Hemoglobin"] = float(hgb_match.group(1))

        creat_match = re.search(r"Creatinine (\d{1}(?:\.\d+)?)", query, re.IGNORECASE)
        if creat_match: parsed_info["lab_results"]["Creatinine"] = float(creat_match.group(1))

        # Extract medications
        med_match = re.findall(r"on (Warfarin|Aspirin|Insulin)", query, re.IGNORECASE)
        parsed_info["medications"] = list(set(med_match))

        # Extract conditions (simplified)
        condition_match = re.findall(r"with (diabetes|hypertension|liver disease|kidney disease|bleeding disorder)", query, re.IGNORECASE)
        parsed_info["conditions"] = list(set(condition_match))

        # Determine if a guideline query is present
        if "guideline" in query.lower() or "recommendation" in query.lower() or "best practice" in query.lower():
            parsed_info["guideline_query"] = query # Use the whole query for RAG

        return parsed_info

    def orchestrate(self, user_query: str) -> str:
        """
        Orchestrates tool use based on the parsed query and synthesizes results.
        """
        print(f"\n--- Processing Query: \"{user_query}\" ---")
        parsed_info = self._parse_query(user_query)
        print(f"Parsed Info: {parsed_info}")

        results = []

        # Invoke Imaging Analysis Tool if relevant
        if parsed_info["image_report"]:
            print("Invoking Medical Imaging Analysis Tool...")
            imaging_result = self.tools["imaging_analysis"](parsed_info["image_report"])
            results.append(imaging_result)

        # Invoke Lab Test Interpretation Tool if relevant
        if parsed_info["lab_results"]:
            print("Invoking Lab Test Interpretation Tool...")
            lab_result = self.tools["lab_interpretation"](parsed_info["lab_results"])
            results.append(lab_result)

        # Invoke Drug Interaction Checking Tool if relevant
        if parsed_info["medications"] or parsed_info["conditions"]:
            print("Invoking Drug Interaction Checking Tool...")
            drug_interaction_result = self.tools["drug_interaction"](parsed_info["medications"], parsed_info["conditions"])
            results.append(drug_interaction_result)

        # Invoke Clinical Guideline Retrieval Tool if relevant
        if parsed_info["guideline_query"]:
            print("Invoking Clinical Guideline Retrieval Tool...")
            guideline_result = self.tools["guideline_retrieval"](parsed_info["guideline_query"])
            results.append(guideline_result)

        # Synthesize results
        if not results:
            return "I couldn't find enough information or relevant tools to provide a specific diagnostic suggestion based on your query. Please provide more details."
        
        synthesis = "\n--- Diagnostic Assistant Summary ---\n"
        for res in results:
            synthesis += f"- {res}\n"
        
        # Simple reasoning based on collected information (simulated LLM reasoning)
        if "pneumonia" in synthesis.lower() and "elevated white blood cell" in synthesis.lower() and "infiltrate" in synthesis.lower():
            synthesis += "Based on the findings, pneumonia is a strong diagnostic possibility. Consider appropriate antibiotic therapy as per guidelines."
        elif "fracture" in synthesis.lower():
            synthesis += "The imaging suggests a fracture. Immobilization and orthopedic consultation are recommended."
        elif "elevated white blood cell" in synthesis.lower():
            synthesis += "Elevated WBC indicates infection or inflammation. Further investigation for the source is advised."
        
        synthesis += "\n--- End Summary ---"
        return synthesis

# --- Demonstration ---
if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()

    # Example 1: Comprehensive query
    query1 = "Patient with chest pain, X-ray shows infiltrate, recent WBC 15000. On Warfarin. What's the likely diagnosis and treatment recommendation?"
    print(assistant.orchestrate(query1))

    # Example 2: Lab results focus
    query2 = "Patient has fatigue, Hemoglobin 10.5, Creatinine 1.5. What's the interpretation of these lab results and what could be the underlying conditions?"
    print(assistant.orchestrate(query2))

    # Example 3: Drug interaction focus
    query3 = "Patient is on Warfarin and Aspirin, with a history of liver disease. Are there any drug interactions to be aware of?"
    print(assistant.orchestrate(query3))

    # Example 4: Guideline retrieval focus
    query4 = "What are the current clinical guidelines for managing community-acquired pneumonia?"
    print(assistant.orchestrate(query4))

    # Example 5: No specific tools triggered or less information
    query5 = "Patient has a headache. What should I do?"
    print(assistant.orchestrate(query5))
