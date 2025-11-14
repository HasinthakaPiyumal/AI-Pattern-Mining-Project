import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any, Union

# --- models.py content ---
class PatientEducationMaterial(BaseModel):
    title: str = Field(..., description="Title of the patient education material.")
    target_audience: str = Field(..., description="Intended audience (e.g., 'general public', 'children', 'elderly').")
    key_takeaways: List[str] = Field(..., description="Bullet points of essential information for the patient.")
    explanation: str = Field(..., description="Detailed but easy-to-understand explanation of the medical condition or procedure.")
    common_questions: List[str] = Field(..., description="List of common questions and their concise answers.")
    disclaimer: str = Field("This information is for educational purposes only and not a substitute for professional medical advice.",
                            description="Standard medical disclaimer.")

class ClinicalSummary(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient.")
    summary_date: str = Field(..., description="Date when the summary was generated (YYYY-MM-DD).")
    diagnosis: List[str] = Field(..., description="List of primary and secondary diagnoses.")
    chief_complaint: str = Field(..., description="Patient's main reason for seeking medical attention.")
    medical_history_summary: str = Field(..., description="Concise summary of patient's relevant medical history.")
    treatment_plan_summary: str = Field(..., description="Overview of the proposed or ongoing treatment plan.")
    key_findings: List[str] = Field(..., description="Important findings from examinations, labs, or imaging.")
    recommendations: List[str] = Field(..., description="Future recommendations or follow-up actions.")

class ResearchAbstract(BaseModel):
    title: str = Field(..., description="Title of the research paper.")
    authors: List[str] = Field(..., description="List of authors.")
    introduction: str = Field(..., description="Brief background and purpose of the study.")
    methods: str = Field(..., description="Summary of the study design and methodology.")
    results: str = Field(..., description="Key findings and outcomes of the research.")
    conclusion: str = Field(..., description="Interpretation of results and their significance.")
    keywords: List[str] = Field(..., description="Relevant keywords for the research.")

# --- prompts.py content ---
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, FewShotPromptTemplate, PromptTemplate

def get_patient_education_prompt(medical_condition: str, examples: Optional[List[Dict[str, str]]] = None) -> ChatPromptTemplate:
    system_template = """You are an AI assistant specialized in creating clear, concise, and accurate patient education materials.
    Your goal is to explain complex medical information in an easy-to-understand manner for the general public, while ensuring factual accuracy and ethical guidelines.
    The output should be structured and contain a title, target audience, key takeaways, an explanation, common questions with answers, and a disclaimer."""

    human_template = """Generate patient education material about {medical_condition}.
    Focus on explaining the condition, its causes, symptoms, and general management in a simple language suitable for a layperson.
    Ensure to include common questions a patient might ask and provide concise answers."""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    if examples:
        example_prompt = PromptTemplate(
            input_variables=["title", "target_audience", "key_takeaways", "explanation", "common_questions", "disclaimer"],
            template="""Title: {title}
            Target Audience: {target_audience}
            Key Takeaways: {key_takeaways}
            Explanation: {explanation}
            Common Questions: {common_questions}
            Disclaimer: {disclaimer}"""
        )
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            suffix=human_template,
            input_variables=["medical_condition"],
            prefix=system_template + "\nHere are some examples of well-structured patient education materials:\n"
        )
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(few_shot_prompt.format(medical_condition=medical_condition))])
    else:
        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

def get_clinical_summary_prompt(patient_data: str, focus_area: Optional[str] = None) -> ChatPromptTemplate:
    system_template = """You are an AI assistant acting as a medical summarizer for healthcare professionals.
    Your task is to generate concise and accurate clinical summaries from patient data.
    Ensure the summary is factual, medically sound, and structured with patient ID, date, diagnosis, chief complaint, medical history, treatment plan, key findings, and recommendations.
    Adhere strictly to medical terminology and maintain patient confidentiality (though specific patient data is provided for summarization)."""

    focus_template = ""
    if focus_area:
        focus_template = f"Specifically, focus on the following aspect: {focus_area}. "

    human_template = f"""Generate a clinical summary for the following patient data:
    {patient_data}
    {focus_template}
    Ensure the output is structured to include patient_id, summary_date, diagnosis, chief_complaint, medical_history_summary, treatment_plan_summary, key_findings, and recommendations."""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

def get_research_abstract_prompt(research_paper_details: str) -> ChatPromptTemplate:
    system_template = """You are an AI assistant specialized in drafting concise and informative research abstracts.
    Your output should follow the standard structure of a scientific abstract: title, authors, introduction, methods, results, and conclusion.
    Ensure accuracy, clarity, and adherence to academic writing standards."""

    human_template = """Draft a research abstract based on the following research paper details:
    {research_paper_details}
    The abstract should include the study's purpose, methods, key findings, and main conclusions."""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# --- utils.py content ---
class MockLLM:
    """A mock LLM to simulate response generation."""
    def __init__(self, delay: float = 0.1):
        self.delay = delay

    def invoke(self, prompt_messages: Any) -> str:
        # Simulate LLM processing
        import time
        time.sleep(self.delay)

        # Extract the human message content to simulate response generation
        # In a real scenario, this would be passed to the LLM API
        human_message_content = ""
        for message in prompt_messages.messages:
            if message.type == "human":
                human_message_content = message.content
                break

        # Simple simulation based on keywords
        if "patient education material" in human_message_content.lower():
            return json.dumps({
                "title": "Understanding Hypertension (High Blood Pressure)",
                "target_audience": "General Public",
                "key_takeaways": ["Hypertension is high blood pressure.", "It often has no symptoms.", "Can lead to heart disease.", "Lifestyle changes and medication help."],
                "explanation": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. It's often called a 'silent killer' because it usually has no obvious symptoms.",
                "common_questions": ["What is normal blood pressure?", "How is hypertension diagnosed?", "How can I lower my blood pressure?", "What are the risks of untreated hypertension?"],
                "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
            })
        elif "clinical summary" in human_message_content.lower():
            return json.dumps({
                "patient_id": "P-789012",
                "summary_date": "2023-10-27",
                "diagnosis": ["Type 2 Diabetes Mellitus", "Essential Hypertension"],
                "chief_complaint": "Follow-up for diabetes management and blood pressure control.",
                "medical_history_summary": "55-year-old male with a history of Type 2 Diabetes diagnosed 5 years ago, managed with Metformin. Essential Hypertension diagnosed 2 years ago, controlled with Lisinopril. No known allergies. Family history significant for cardiovascular disease.",
                "treatment_plan_summary": "Continue current medications. Recommend dietary consultation for diabetes and hypertension management. Schedule HbA1c and lipid panel in 3 months. Follow-up appointment in 3 months.",
                "key_findings": ["HbA1c 7.2%", "BP 130/80 mmHg", "BMI 29.5 kg/m^2"],
                "recommendations": ["Dietary consultation", "Exercise regimen", "Smoking cessation (if applicable)"]
            })
        elif "research abstract" in human_message_content.lower():
            return json.dumps({
                "title": "Impact of Telemedicine on Patient Outcomes in Chronic Disease Management",
                "authors": ["J. Smith", "A. B. Johnson", "C. D. Williams"],
                "introduction": "Telemedicine has emerged as a crucial tool for healthcare delivery, especially in chronic disease management. This study investigates its impact on patient outcomes.",
                "methods": "A randomized controlled trial involving 300 patients with type 2 diabetes and hypertension was conducted over 12 months. Patients were randomized to either receive standard in-person care or telemedicine-supported care.",
                "results": "The telemedicine group showed significant improvements in HbA1c levels (p<0.01) and blood pressure control (p<0.05) compared to the control group. Patient satisfaction was also higher in the telemedicine group.",
                "conclusion": "Telemedicine significantly improves glycemic and blood pressure control in patients with chronic diseases and enhances patient satisfaction, suggesting its potential as an effective care model.",
                "keywords": ["telemedicine", "chronic disease", "diabetes", "hypertension", "patient outcomes"]
            })
        return "Simulated LLM response for: " + human_message_content[:100] + "..."

def dynamic_prompt_selector(content_type: str, *args, **kwargs) -> ChatPromptTemplate:
    """Dynamically selects and prepares the appropriate prompt."""

    if content_type == "patient_education":
        # Example few-shot prompts
        few_shot_examples: List[Dict[str, str]] = [
            {
                "title": "Understanding the Common Cold",
                "target_audience": "General Public",
                "key_takeaways": "['Caused by viruses', 'Symptoms include runny nose, cough, sore throat', 'No cure, treat symptoms', 'Prevent spread by handwashing']",
                "explanation": "The common cold is a viral infection of your nose and throat. It's generally harmless, although it might not feel that way. Many types of viruses can cause a common cold. Adults typically get two or three colds a year. Infants and young children may get even more.",
                "common_questions": "['How long does a cold last?', 'When should I see a doctor for a cold?', 'Are antibiotics effective for a cold?']",
                "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
            }
        ]
        return get_patient_education_prompt(medical_condition=kwargs.get("medical_condition", "a medical condition"), examples=few_shot_examples)
    elif content_type == "clinical_summary":
        return get_clinical_summary_prompt(patient_data=kwargs.get("patient_data", "No patient data provided."), focus_area=kwargs.get("focus_area"))
    elif content_type == "research_abstract":
        return get_research_abstract_prompt(research_paper_details=kwargs.get("research_paper_details", "No research details provided."))
    else:
        raise ValueError(f"Unknown content type: {content_type}")

def parse_llm_json_output(llm_output: str, pydantic_model: BaseModel) -> Union[BaseModel, Dict[str, Any]]:
    """Parses LLM string output into a Pydantic model or returns raw if parsing fails."""
    try:
        data = json.loads(llm_output)
        return pydantic_model.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Could not parse LLM output into Pydantic model {pydantic_model.__name__}. Error: {e}")
        print(f"Raw LLM output: {llm_output[:200]}...") # Print first 200 chars
        return {"parse_error": str(e), "raw_output": llm_output}

# --- evaluators.py content ---
from guardrails.hub import Toxicity, ProfanityFree
from guardrails import Guard

# --- Constitutional AI / Guardrails based Validation ---
def get_medical_content_guard(content_type: str) -> Guard:
    """
    Defines a Guardrails Guard for validating medical content.
    Incorporates ethical alignment and quality checks.
    """
    if content_type == "patient_education":
        return Guard.from_string(
            validators=[
                Toxicity(threshold=0.5, on_fail="fix"), # Ensure non-toxic language
                ProfanityFree(on_fail="fix"), # Ensure no profanity
                # Custom validator for factual accuracy (conceptual for now, would need a knowledge base)
            ],
        )
    elif content_type == "clinical_summary":
        return Guard.from_string(
            validators=[
                # Guardrails for clinical summaries can be more specific, e.g., checking for PII redaction,
                # consistency with structured EHR data (if integrated with a knowledge base).
            ]
        )
    elif content_type == "research_abstract":
        return Guard.from_string(
            validators=[
                # Guardrails for research abstracts could include checks for academic style, conciseness,
                # and factual claims against scientific databases.
            ]
        )
    return Guard.from_string(validators=[]) # Default empty guard

# --- LLM-based Autorating (Mocked) ---
class LLMEvaluator:
    """Simulates an LLM-based autorating system for medical content."""
    def __init__(self, llm: MockLLM):
        self.llm = llm

    def evaluate(self, generated_content: str, content_type: str, criteria: Dict[str, str]) -> Dict[str, Any]:
        """
        Evaluates generated content based on specific criteria using a mock LLM.
        In a real scenario, this would involve prompting another LLM with the content and criteria.
        """
        print(f"\n--- Simulating LLM-based Autorating for {content_type} ---")
        # Mock LLM response for evaluation
        mock_evaluation_response = {
            "scores": {
                "medical_accuracy": 4.5,
                "clarity": 4.0,
                "completeness": 3.8,
                "bias_mitigation": 4.2
            },
            "overall_feedback": f"The {content_type} content is generally accurate and clear. Some areas for completeness could be improved. Bias mitigation appears effective."
        }
        print(f"Mock LLM Evaluation Result: {json.dumps(mock_evaluation_response, indent=2)}")
        return mock_evaluation_response

# --- Round-trip Consistency Check (Conceptual) ---
def round_trip_consistency_check(original_data: str, summarized_content: ClinicalSummary, llm: MockLLM) -> Dict[str, Any]:
    """
    Conceptually checks round-trip consistency for a clinical summary.
    In a real scenario, this would involve generating a summary, then expanding it
    back to compare with original data, or checking if key information from
    original data is present in the summary and vice-versa.
    """
    print("\n--- Simulating Round-trip Consistency Check ---")

    missing_keys_in_summary = []
    if not summarized_content.diagnosis: missing_keys_in_summary.append("diagnosis")
    if not summarized_content.chief_complaint: missing_keys_in_summary.append("chief_complaint")
    if not summarized_content.medical_history_summary: missing_keys_in_summary.append("medical_history_summary")
    if not summarized_content.treatment_plan_summary: missing_keys_in_summary.append("treatment_plan_summary")

    if not missing_keys_in_summary:
        print("Basic consistency check passed: Key fields are present in the summary.")
        return {"status": "passed", "details": "Key summary fields are present."}
    else:
        print(f"Basic consistency check failed: Missing or empty fields in summary: {", ".join(missing_keys_in_summary)}")
        return {"status": "failed", "details": f"Missing or empty fields: {", ".join(missing_keys_in_summary)}"}

# --- Adversarial Evaluation for Truthfulness/Misinformation (Guardrails with specific validators) ---
def adversarial_misinformation_check(generated_content: str, content_type: str) -> Dict[str, Any]:
    """
    Applies adversarial evaluation conceptually using Guardrails for specific checks
    like factual consistency (if a custom validator existed) or identifying red flags.
    Here, we use `Toxicity` and `ProfanityFree` to catch potentially harmful or misleading tones.
    """
    print("\n--- Simulating Adversarial Misinformation Check ---")
    try:
        guard = Guard.from_string(
            validators=[
                Toxicity(threshold=0.5, on_fail="fix"), # Identify and fix toxic content which might stem from misinformation
                ProfanityFree(on_fail="fix"),
                # For factual checks, you'd need custom validators integrated with a knowledge graph or trusted data
            ]
        )
        validated_output = guard.validate(generated_content)
        if validated_output.validation_passed:
            print("Adversarial Check Status: Passed. No immediate red flags for misinformation/toxicity detected.")
            return {"status": "passed", "details": "Content appears free of toxicity and profanity. (Further factual checks would require a knowledge base)."}
        else:
            print(f"Adversarial Check Status: Failed. Issues found: {validated_output.validation_results}")
            return {"status": "failed", "details": validated_output.validation_results}
    except Exception as e:
        print(f"Error during adversarial check: {e}")
        return {"status": "error", "details": str(e)}

# --- main.py content ---
def run_medical_content_generation_and_validation_system():
    print("--- Medical Content Generation and Validation System Started ---")

    llm = MockLLM(delay=0.5) # Initialize mock LLM

    # --- Scenario 1: Generate Patient Education Material ---
    print("\n\n--- Scenario 1: Generating Patient Education Material ---")
    content_type_pe = "patient_education"
    medical_condition = "Type 2 Diabetes"

    # Dynamic prompt selection (including few-shot examples via utils)
    pe_prompt = dynamic_prompt_selector(content_type_pe, medical_condition=medical_condition)
    print(f"Generated Patient Education Prompt:\n{pe_prompt.format_messages()}")

    # Generate content
    raw_pe_output = llm.invoke(pe_prompt)
    print(f"\nRaw LLM Output for Patient Education:\n{raw_pe_output[:200]}...")

    # Parse and validate with Pydantic
    patient_education_content: Union[PatientEducationMaterial, Dict[str, Any]]
    try:
        patient_education_content = parse_llm_json_output(raw_pe_output, PatientEducationMaterial)
        if isinstance(patient_education_content, PatientEducationMaterial):
            print("\nParsed Patient Education Content (Pydantic):\n", patient_education_content.model_dump_json(indent=2))
        else:
            print("\nPatient Education Content Parsing Failed:\n", patient_education_content)
    except ValidationError as e:
        print(f"\nPydantic Validation Error for Patient Education: {e}")
        patient_education_content = {"validation_error": str(e), "raw_output": raw_pe_output}

    # Apply Quality Assurance (LLM-based Autorating & Adversarial Check)
    if isinstance(patient_education_content, PatientEducationMaterial):
        llm_evaluator = LLMEvaluator(llm)
        pe_evaluation_criteria = {
            "medical_accuracy": "How factually correct is the information?",
            "clarity": "Is the language easy for a layperson to understand?",
            "completeness": "Does it cover essential aspects of the condition?",
            "bias_mitigation": "Is the content neutral and free from bias?"
        }
        llm_evaluator.evaluate(patient_education_content.model_dump_json(), content_type_pe, pe_evaluation_criteria)

        adversarial_result_pe = adversarial_misinformation_check(patient_education_content.model_dump_json(), content_type_pe)
        print(f"Adversarial Check Result (Patient Education): {json.dumps(adversarial_result_pe, indent=2)}")

        # Constitutional AI / Guardrails
        pe_guard = get_medical_content_guard(content_type_pe)
        guarded_output_pe = pe_guard.validate(patient_education_content.model_dump_json())
        print(f"\nGuardrails (Constitutional AI) Validation Result for Patient Education:")
        print(f"  Passed: {guarded_output_pe.validation_passed}")
        if not guarded_output_pe.validation_passed:
            print(f"  Failures: {guarded_output_pe.validation_results}")
            # If fix is applied, guarded_output_pe.fixed_output contains the corrected version
        else:
            print("  No issues detected by Guardrails.")

    # --- Scenario 2: Generate Clinical Summary ---
    print("\n\n--- Scenario 2: Generating Clinical Summary ---")
    content_type_cs = "clinical_summary"
    patient_data = """
    Patient Name: Jane Doe, DOB: 1968-05-15, MRN: P-789012
    Visit Date: 2023-10-26
    Chief Complaint: Patient presents for routine follow-up of Type 2 Diabetes and Hypertension. Reports occasional fatigue but generally well.
    Diagnoses:
    1. Type 2 Diabetes Mellitus (E11.9) - On Metformin 1000mg BID
    2. Essential (Primary) Hypertension (I10) - On Lisinopril 20mg daily
    Medical History: Childhood asthma (resolved), no surgeries.
    Social History: Non-smoker, occasional alcohol. Works as an accountant.
    Medications: Metformin 1000mg BID, Lisinopril 20mg daily.
    Allergies: Penicillin (Rash)
    Physical Exam: BP 132/82, HR 72, Temp 98.6F. Lungs clear, heart regular rhythm. No edema.
    Labs (recent): HbA1c 7.1%, Creatinine 0.9 mg/dL, GFR >60.
    Assessment: Stable Type 2 DM and HTN.
    Plan: Continue current medications. Recommend dietician consult for improved glycemic control. Schedule follow-up in 3 months with labs.
    """
    focus_area = "key findings and treatment plan"

    cs_prompt = dynamic_prompt_selector(content_type_cs, patient_data=patient_data, focus_area=focus_area)
    print(f"\nGenerated Clinical Summary Prompt:\n{cs_prompt.format_messages()}")

    raw_cs_output = llm.invoke(cs_prompt)
    print(f"\nRaw LLM Output for Clinical Summary:\n{raw_cs_output[:200]}...")

    clinical_summary_content: Union[ClinicalSummary, Dict[str, Any]]
    try:
        clinical_summary_content = parse_llm_json_output(raw_cs_output, ClinicalSummary)
        if isinstance(clinical_summary_content, ClinicalSummary):
            print("\nParsed Clinical Summary Content (Pydantic):\n", clinical_summary_content.model_dump_json(indent=2))
        else:
            print("\nClinical Summary Content Parsing Failed:\n", clinical_summary_content)
    except ValidationError as e:
        print(f"\nPydantic Validation Error for Clinical Summary: {e}")
        clinical_summary_content = {"validation_error": str(e), "raw_output": raw_cs_output}


    # Apply Quality Assurance (Round-trip & Adversarial Check)
    if isinstance(clinical_summary_content, ClinicalSummary):
        round_trip_result = round_trip_consistency_check(patient_data, clinical_summary_content, llm)
        print(f"Round-trip Consistency Result: {json.dumps(round_trip_result, indent=2)}")

        adversarial_result_cs = adversarial_misinformation_check(clinical_summary_content.model_dump_json(), content_type_cs)
        print(f"Adversarial Check Result (Clinical Summary): {json.dumps(adversarial_result_cs, indent=2)}")

        # Constitutional AI / Guardrails (less specific for summary in this mock)
        cs_guard = get_medical_content_guard(content_type_cs)
        guarded_output_cs = cs_guard.validate(clinical_summary_content.model_dump_json())
        print(f"\nGuardrails (Constitutional AI) Validation Result for Clinical Summary:")
        print(f"  Passed: {guarded_output_cs.validation_passed}")
        if not guarded_output_cs.validation_passed:
            print(f"  Failures: {guarded_output_cs.validation_results}")
        else:
            print("  No issues detected by Guardrails.")

    # --- Scenario 3: Generate Research Abstract ---
    print("\n\n--- Scenario 3: Generating Research Abstract ---")
    content_type_ra = "research_abstract"
    research_details = """
    Study Title: A Novel AI-driven Diagnostic Tool for Early Detection of Glaucoma
    Authors: Dr. E. Chen, Prof. L. Garcia, Dr. S. Patel
    Background: Glaucoma is a leading cause of irreversible blindness. Early detection is critical. Current methods have limitations.
    Methods: We developed a deep learning model (CNN-based) trained on a dataset of 10,000 OCT scans from patients with early glaucoma and healthy controls. The model achieves automated classification.
    Results: The model demonstrated a sensitivity of 92% and specificity of 89% in identifying early glaucomatous changes, outperforming current clinical markers.
    Conclusion: This AI tool shows significant promise for improving early glaucoma diagnosis, potentially preserving vision in many patients.
    """
    ra_prompt = dynamic_prompt_selector(content_type_ra, research_paper_details=research_details)
    print(f"\nGenerated Research Abstract Prompt:\n{ra_prompt.format_messages()}")

    raw_ra_output = llm.invoke(ra_prompt)
    print(f"\nRaw LLM Output for Research Abstract:\n{raw_ra_output[:200]}...")

    research_abstract_content: Union[ResearchAbstract, Dict[str, Any]]
    try:
        research_abstract_content = parse_llm_json_output(raw_ra_output, ResearchAbstract)
        if isinstance(research_abstract_content, ResearchAbstract):
            print("\nParsed Research Abstract Content (Pydantic):\n", research_abstract_content.model_dump_json(indent=2))
        else:
            print("\nResearch Abstract Content Parsing Failed:\n", research_abstract_content)
    except ValidationError as e:
        print(f"\nPydantic Validation Error for Research Abstract: {e}")
        research_abstract_content = {"validation_error": str(e), "raw_output": raw_ra_output}


    # Apply Quality Assurance (Adversarial Check)
    if isinstance(research_abstract_content, ResearchAbstract):
        llm_evaluator = LLMEvaluator(llm)
        ra_evaluation_criteria = {
            "academic_accuracy": "How factually correct is the scientific information?",
            "conciseness": "Is the abstract to the point and within typical length limits?",
            "clarity_of_methods": "Are the methods clearly described?",
            "significance_of_results": "Does it highlight the importance of the findings?"
        }
        llm_evaluator.evaluate(research_abstract_content.model_dump_json(), content_type_ra, ra_evaluation_criteria)

        adversarial_result_ra = adversarial_misinformation_check(research_abstract_content.model_dump_json(), content_type_ra)
        print(f"Adversarial Check Result (Research Abstract): {json.dumps(adversarial_result_ra, indent=2)}")

        ra_guard = get_medical_content_guard(content_type_ra)
        guarded_output_ra = ra_guard.validate(research_abstract_content.model_dump_json())
        print(f"\nGuardrails (Constitutional AI) Validation Result for Research Abstract:")
        print(f"  Passed: {guarded_output_ra.validation_passed}")
        if not guarded_output_ra.validation_passed:
            print(f"  Failures: {guarded_output_ra.validation_results}")
        else:
            print("  No issues detected by Guardrails.")

    print("\n--- Medical Content Generation and Validation System Finished ---")

if __name__ == "__main__":
    run_medical_content_generation_and_validation_system()
