class InputHandler:
    @staticmethod
    def process_inputs(image_data: bytes, text_description: str, lab_results: dict) -> dict:
        print("InputHandler: Processing multimodal inputs...")
        processed_image = f"Image data of size {len(image_data)} bytes"
        processed_text = f"Text description: {text_description[:50]}..."
        processed_lab_results = f"Lab results: {len(lab_results)} entries"

        return {
            "image_data": image_data,
            "text_data": text_description,
            "lab_data": lab_results,
            "processed_image_summary": processed_image,
            "processed_text_summary": processed_text,
            "processed_lab_summary": processed_lab_results
        }

class DecompositionEngine:
    @staticmethod
    def decompose_problem(inputs: dict) -> list[str]:
        print("DecompositionEngine: Decomposing the diagnostic problem into sub-questions...")
        sub_questions = [
            "What are the key findings from the patient's images (e.g., X-ray, MRI)?",
            "What are the primary symptoms and patient history described in the text?",
            "Are there any abnormal values or significant patterns in the lab results?",
            "Considering all findings, what are the most probable differential diagnoses?",
            "What further investigations might be required based on initial findings?"
        ]
        
        if "fever" in inputs.get("text_data", "").lower():
            sub_questions.insert(2, "Is there any indication of infection from text or lab data?")
        if inputs.get("lab_data", {}).get("White Blood Cell Count") and \
           inputs["lab_data"]["White Blood Cell Count"] > 10000:
            sub_questions.insert(3, "What is the significance of the elevated White Blood Cell Count?")

        print(f"Generated {len(sub_questions)} sub-questions.")
        return sub_questions

class ImageAnalysisModel:
    @staticmethod
    def analyze(image_data: bytes) -> str:
        print("ImageAnalysisModel: Analyzing image data...")
        if b"tumor" in image_data:
            return "Image analysis suggests presence of a suspicious mass in the lung, requiring further investigation."
        elif b"fracture" in image_data:
            return "Image analysis indicates a hairline fracture in the tibia."
        else:
            return "Image analysis shows no significant abnormalities, but subtle findings cannot be ruled out without expert review."

class NLPModel:
    @staticmethod
    def analyze(text_description: str) -> str:
        print("NLPModel: Analyzing text description...")
        findings = []
        if "fever" in text_description.lower():
            findings.append("Patient reports fever.")
        if "cough" in text_description.lower():
            findings.append("Patient reports persistent cough.")
        if "headache" in text_description.lower():
            findings.append("Patient reports headache.")
        if "diabetes" in text_description.lower():
            findings.append("Patient has a history of diabetes.")
        
        if not findings:
            return "NLP analysis found no specific concerning symptoms in the description."
        return "Text analysis reveals: " + "; ".join(findings) + "."

class DataAnalysisModel:
    @staticmethod
    def analyze(lab_results: dict) -> str:
        print("DataAnalysisModel: Analyzing lab results...")
        abnormalities = []
        if lab_results.get("White Blood Cell Count") and lab_results["White Blood Cell Count"] > 10000:
            abnormalities.append(f"Elevated White Blood Cell Count: {lab_results["White Blood Cell Count"]} (indicative of infection/inflammation).")
        if lab_results.get("Hemoglobin") and lab_results["Hemoglobin"] < 12.0:
            abnormalities.append(f"Low Hemoglobin: {lab_results["Hemoglobin"]} (suggesting anemia).")
        if lab_results.get("Glucose") and lab_results["Glucose"] > 125:
            abnormalities.append(f"High Glucose: {lab_results["Glucose"]} (potential hyperglycemia/diabetes).")

        if not abnormalities:
            return "Lab results are within normal limits for the analyzed parameters."
        return "Lab results analysis reveals: " + "; ".join(abnormalities) + "."

class SynthesisEngine:
    @staticmethod
    def synthesize(image_findings: str, text_findings: str, lab_findings: str, sub_questions: list[str]) -> str:
        print("SynthesisEngine: Synthesizing findings into a diagnostic suggestion...")
        
        synthesis_report = []
        synthesis_report.append("--- Comprehensive Diagnostic Suggestion ---")
        synthesis_report.append("\n1. Key Findings from Sub-questions:")
        synthesis_report.append(f"   - Image Analysis: {image_findings}")
        synthesis_report.append(f"   - Text Description: {text_findings}")
        synthesis_report.append(f"   - Lab Results: {lab_findings}")

        synthesis_report.append("\n2. Overall Interpretation:")
        overall_impression = []

        if "suspicious mass" in image_findings.lower() or \
           "elevated white blood cell count" in lab_findings.lower() or \
           "fever" in text_findings.lower():
            overall_impression.append("High suspicion for an inflammatory or infectious process, potentially with a focal lesion. Further investigation is strongly recommended.")
        elif "fracture" in image_findings.lower():
            overall_impression.append("Diagnosis of a fracture. Recommend immobilization and follow-up with orthopedics.")
        elif "low hemoglobin" in lab_findings.lower() and "fatigue" in text_findings.lower():
             overall_impression.append("Findings suggest anemia, potentially contributing to reported fatigue. Recommend further workup for anemia etiology.")
        elif "no significant abnormalities" in image_findings.lower() and \
             "within normal limits" in lab_findings.lower() and \
             "no specific concerning symptoms" in text_findings.lower():
            overall_impression.append("No significant abnormalities detected across modalities. Symptoms may be benign or require further observation/specialized testing if persistent.")
        else:
            overall_impression.append("Complex findings require integrated clinical judgment. Consider specific differentials based on individual findings.")
        
        synthesis_report.append("   " + " ".join(overall_impression))
        synthesis_report.append("\n3. Addressed Sub-questions:")
        for i, q in enumerate(sub_questions):
            synthesis_report.append(f"   {i+1}. {q}")

        synthesis_report.append("\n--- End of Suggestion ---")
        return "\n".join(synthesis_report)

def run_medical_diagnosis_assistant(
    image_data: bytes,
    text_description: str,
    lab_results: dict
) -> str:
    print("\n--- Starting Medical Diagnosis Assistant (DDCoT Pattern) ---")

    processed_inputs = InputHandler.process_inputs(image_data, text_description, lab_results)
    print(f"Processed Input Summary: {processed_inputs.get("processed_image_summary")}, "
          f"{processed_inputs.get("processed_text_summary")}, {processed_inputs.get("processed_lab_summary")}")

    sub_questions = DecompositionEngine.decompose_problem(processed_inputs)
    print("Sub-questions generated:")
    for i, q in enumerate(sub_questions):
        print(f"  {i+1}. {q}")

    print("\n--- Solving Sub-questions with Specialized AI Models ---")
    image_findings = ImageAnalysisModel.analyze(processed_inputs["image_data"])
    text_findings = NLPModel.analyze(processed_inputs["text_data"])
    lab_findings = DataAnalysisModel.analyze(processed_inputs["lab_data"])

    print(f"\nImage Model Output: {image_findings}")
    print(f"NLP Model Output: {text_findings}")
    print(f"Data Model Output: {lab_findings}")

    diagnostic_suggestion = SynthesisEngine.synthesize(
        image_findings, text_findings, lab_findings, sub_questions
    )

    print("\n--- Medical Diagnosis Assistant Finished ---")
    return diagnostic_suggestion

if __name__ == "__main__":
    mock_image_data_1 = b"image_bytes_with_tumor_indicator_for_analysis_by_model"
    mock_text_description_1 = "Patient presents with persistent cough for 2 months, mild fever, and fatigue. History of smoking for 10 years."
    mock_lab_results_1 = {"White Blood Cell Count": 15000, "Hemoglobin": 14.5, "Glucose": 95}
    
    mock_image_data_2 = b"image_bytes_indicating_no_major_issue"
    mock_text_description_2 = "Patient reports mild headache and occasional dizziness for a week. No significant medical history."
    mock_lab_results_2 = {"White Blood Cell Count": 7500, "Hemoglobin": 13.0, "Glucose": 110}

    mock_image_data_3 = b"image_bytes_with_fracture_indicator"
    mock_text_description_3 = "Patient fell and reports severe pain and swelling in the right ankle. Unable to bear weight."
    mock_lab_results_3 = {"White Blood Cell Count": 9000, "CRP": 5.0}

    print("\n\n=========== Running Diagnosis Case 1 ===========")
    diagnosis_output_1 = run_medical_diagnosis_assistant(
        mock_image_data_1, mock_text_description_1, mock_lab_results_1
    )
    print(diagnosis_output_1)

    print("\n\n=========== Running Diagnosis Case 2 ===========")
    diagnosis_output_2 = run_medical_diagnosis_assistant(
        mock_image_data_2, mock_text_description_2, mock_lab_results_2
    )
    print(diagnosis_output_2)

    print("\n\n=========== Running Diagnosis Case 3 ===========")
    diagnosis_output_3 = run_medical_diagnosis_assistant(
        mock_image_data_3, mock_text_description_3, mock_lab_results_3
    )
    print(diagnosis_output_3)