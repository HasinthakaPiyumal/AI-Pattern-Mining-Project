from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, BaseOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import Generation

# --- Mock LLM for Demonstration Purposes ---
class MockLLM(BaseChatModel):
    """A mock LLM for testing without actual API calls."""
    response_map: Dict[str, str]

    def __init__(self, response_map: Dict[str, str] = None, **kwargs):
        super().__init__(**kwargs)
        self.response_map = response_map if response_map is not None else {}

    def _generate(self, messages: List[BaseMessage], stop: List[str] | None = None, **kwargs: Any) -> Any:
        # Simple logic: look for a keyword in the prompt to return a predefined response
        # In a real scenario, this would involve more sophisticated prompt matching or a fixed response
        full_prompt = " ".join([msg.content for msg in messages])
        for key, value in self.response_map.items():
            if key in full_prompt:
                return Generation(text=value)
        
        # Default response if no specific match
        return Generation(text="Mock LLM generated a response based on the input.")

    @property
    def _llm_type(self) -> str:
        return "mock"


# --- 1. clinical_note_generator.py components ---

class ClinicalNote(BaseModel):
    patient_id: str = Field(description="Unique identifier for the patient")
    diagnosis: str = Field(description="Primary diagnosis for the patient")
    treatment_plan: str = Field(description="Outline of the proposed treatment")
    medications: List[str] = Field(description="List of prescribed medications")
    recommendations: List[str] = Field(description="Further recommendations or follow-up actions")
    assessment: str = Field(description="Physician's assessment of the patient's condition")


class ClinicalNoteGenerator:
    """Generates clinical notes using various prompt engineering techniques."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _generate_with_template(self, prompt_template: PromptTemplate, **kwargs) -> str:
        """Helper method to encapsulate common LLM invocation logic."""
        prompt = prompt_template.format(**kwargs)
        # For LangChain ChatModel, usually you'd pass messages, but for simplicity with MockLLM here,
        # we simulate a single string input if the mock expects it that way.
        # In a real LangChain setup with ChatOpenAI, you'd use ChatPromptTemplate and invoke(messages).
        response = self.llm._generate([BaseMessage(content=prompt)], stop=[]).text
        return response

    def generate_few_shot_note(self, patient_data: Dict[str, Any]) -> ClinicalNote:
        """Generates a note using few-shot learning with examples."""
        examples = [
            {
                "patient_summary": "Patient presents with chronic cough and fatigue.",
                "note": "Patient ID: P001\nDiagnosis: Bronchitis\nTreatment Plan: Antibiotics, rest\nMedications: Azithromycin\nRecommendations: Follow up in 1 week\nAssessment: Stable condition."
            },
            {
                "patient_summary": "55-year-old male with chest pain.",
                "note": "Patient ID: P002\nDiagnosis: Angina\nTreatment Plan: Nitroglycerin, lifestyle changes\nMedications: Nitroglycerin\nRecommendations: Cardiology consult\nAssessment: Requires further evaluation."
            }
        ]

        example_prompt = PromptTemplate(
            input_variables=["patient_summary", "note"],
            template="Patient Summary: {patient_summary}\nClinical Note:\n{note}"
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Generate a detailed clinical note based on the patient summary, adhering to the structure provided in the examples.",
            suffix="Patient Summary: {patient_summary}\nClinical Note:\n",
            input_variables=["patient_summary"],
        )
        
        generated_text = self._generate_with_template(few_shot_prompt, patient_summary=patient_data["summary"])
        # Parse the string output into ClinicalNote. This step assumes the LLM adheres to the format.
        # A robust system would use PydanticOutputParser here for strict schema adherence.
        return self._parse_note_from_text(generated_text, patient_data["id"])

    def generate_zero_shot_note(self, patient_data: Dict[str, Any]) -> ClinicalNote:
        """Generates a note without examples, relying solely on the prompt."""
        prompt_template = PromptTemplate(
            input_variables=["patient_summary"],
            template=(
                "Generate a comprehensive clinical note for the following patient summary. "
                "Include Patient ID, Diagnosis, Treatment Plan, Medications, Recommendations, and Assessment.\n\n"
                "Patient Summary: {patient_summary}\n\nClinical Note:"
            ),
        )
        generated_text = self._generate_with_template(prompt_template, patient_summary=patient_data["summary"])
        return self._parse_note_from_text(generated_text, patient_data["id"])

    def generate_role_based_note(self, patient_data: Dict[str, Any], role: str) -> ClinicalNote:
        """Generates a note with a specific role injected into the prompt."""
        prompt_template = PromptTemplate(
            input_variables=["role", "patient_summary"],
            template=(
                "As a {role}, generate a detailed clinical note for the following patient summary. "
                "Ensure all relevant medical details are included in the format: Patient ID, Diagnosis, Treatment Plan, Medications, Recommendations, and Assessment.\n\n"
                "Patient Summary: {patient_summary}\n\nClinical Note:"
            ),
        )
        generated_text = self._generate_with_template(prompt_template, role=role, patient_summary=patient_data["summary"])
        return self._parse_note_from_text(generated_text, patient_data["id"])

    def generate_template_driven_note(self, patient_data: Dict[str, Any]) -> ClinicalNote:
        """Uses PydanticOutputParser to ensure the output conforms to the ClinicalNote schema."""
        parser = PydanticOutputParser(pydantic_object=ClinicalNote)

        prompt_template = PromptTemplate(
            template=(
                "Generate a clinical note for the patient based on the following summary.\n{format_instructions}\n\nPatient Summary: {patient_summary}\n\nClinical Note:"
            ),
            input_variables=["patient_summary"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            },
        )
        
        generated_text = self._generate_with_template(prompt_template, patient_summary=patient_data["summary"])
        try:
            return parser.parse(generated_text)
        except Exception as e:
            print(f"Error parsing template-driven note: {e}. Raw output: {generated_text}")
            # Fallback for mock LLM if it doesn't strictly adhere to Pydantic format
            return self._parse_note_from_text(generated_text, patient_data["id"])

    def generate_dynamic_note(self, patient_data: Dict[str, Any], context_info: Dict[str, Any]) -> ClinicalNote:
        """Dynamically constructs prompts based on patient data and additional contextual information."""
        dynamic_prompt_str = (
            f"Generate a clinical note for Patient ID: {patient_data['id']}. "
            f"Patient Summary: {patient_data['summary']}. "
            f"Additional Context: {context_info.get('allergies', 'None')}, {context_info.get('previous_history', 'No prior significant history')}. "
            "Include Diagnosis, Treatment Plan, Medications, Recommendations, and Assessment.\n\nClinical Note:"
        )
        prompt_template = PromptTemplate(input_variables=[], template=dynamic_prompt_str)
        generated_text = self._generate_with_template(prompt_template)
        return self._parse_note_from_text(generated_text, patient_data["id"])

    def _parse_note_from_text(self, text: str, patient_id: str) -> ClinicalNote:
        """Helper to parse a simple string into a ClinicalNote object (for mock LLM scenarios)."""
        # This is a very basic parsing for demonstration. A real system would need more robust parsing.
        diagnosis = "Unknown Diagnosis"
        treatment_plan = "Generic Treatment"
        medications = []
        recommendations = []
        assessment = "General assessment based on available information."

        if "Diagnosis:" in text:
            diagnosis = text.split("Diagnosis:")[1].split("\n")[0].strip()
        if "Treatment Plan:" in text:
            treatment_plan = text.split("Treatment Plan:")[1].split("\n")[0].strip()
        if "Medications:" in text:
            med_str = text.split("Medications:")[1].split("\n")[0].strip()
            medications = [m.strip() for m in med_str.split(',') if m.strip()]
        if "Recommendations:" in text:
            rec_str = text.split("Recommendations:")[1].split("\n")[0].strip()
            recommendations = [r.strip() for r in rec_str.split(',') if r.strip()]
        if "Assessment:" in text:
            assessment = text.split("Assessment:")[1].split("\n")[0].strip()

        return ClinicalNote(
            patient_id=patient_id,
            diagnosis=diagnosis,
            treatment_plan=treatment_plan,
            medications=medications,
            recommendations=recommendations,
            assessment=assessment,
        )


# --- 2. quality_assurance_system.py components ---

class EvaluationResult(BaseModel):
    passed: bool = Field(description="True if the evaluation passed, False otherwise")
    feedback: str = Field(description="Detailed feedback on the evaluation outcome")
    score: float = Field(description="A numerical score representing the quality (e.g., 0-10)")


class QualityAssuranceSystem:
    """Implements evaluation frameworks for clinical notes."""

    def __init__(self, evaluation_llm: BaseChatModel):
        self.evaluation_llm = evaluation_llm

    def _evaluate_with_llm(self, prompt_template: PromptTemplate, **kwargs) -> str:
        """Helper to invoke the evaluation LLM."""
        prompt = prompt_template.format(**kwargs)
        response = self.evaluation_llm._generate([BaseMessage(content=prompt)], stop=[]).text
        return response

    def autorate_note(self, generated_note: ClinicalNote, medical_guidelines: str) -> EvaluationResult:
        """Uses the evaluation LLM to score the generated note against medical guidelines."""
        prompt_template = PromptTemplate(
            input_variables=["generated_note", "medical_guidelines"],
            template=(
                "Rate the following clinical note based on the provided medical guidelines. "
                "Provide a score out of 10 for completeness, accuracy, and clarity. "
                "Also, provide detailed feedback.\n\n"
                "Medical Guidelines:\n{medical_guidelines}\n\n"
                "Generated Clinical Note:\n{generated_note}\n\n"
                "Evaluation Result (Score: X/10, Feedback: ...):"
            ),
        )
        generated_feedback = self._evaluate_with_llm(
            prompt_template,
            generated_note=generated_note.model_dump_json(indent=2),
            medical_guidelines=medical_guidelines,
        )
        return self._parse_evaluation_result(generated_feedback)

    def round_trip_check(self, original_patient_data: Dict[str, Any], generated_note_text: str) -> EvaluationResult:
        """Extracts key information from the generated note and compares it to original data."""
        prompt_template = PromptTemplate(
            input_variables=["generated_note_text"],
            template=(
                "Extract the patient's ID, primary diagnosis, and main treatment plan from the following clinical note. "
                "Format as: ID: [ID], Diagnosis: [Diagnosis], Treatment: [Treatment].\n\n"
                "Clinical Note:\n{generated_note_text}\n\nExtracted Information:"
            ),
        )
        extracted_info = self._evaluate_with_llm(prompt_template, generated_note_text=generated_note_text)

        # Simple comparison logic
        passed = True
        feedback = "Round-trip consistency check passed."
        score = 10.0

        if f"ID: {original_patient_data['id']}" not in extracted_info:
            passed = False
            feedback += f" Patient ID mismatch. Expected {original_patient_data['id']}. "
            score -= 3.0
        # Add more robust checks for diagnosis and treatment

        if not passed:
            feedback = f"Round-trip consistency check failed. {feedback}"
            score = max(0.0, score)

        return EvaluationResult(passed=passed, feedback=feedback, score=score)

    def adversarial_evaluation(self, generated_note: ClinicalNote, challenge_prompt: str) -> EvaluationResult:
        """Prompts the evaluation LLM to identify biases or factual errors."""
        prompt_template = PromptTemplate(
            input_variables=["generated_note", "challenge_prompt"],
            template=(
                "Review the following clinical note and identify any potential biases, factual errors, or misinterpretations "
                "based on this challenge: {challenge_prompt}. "
                "Provide a clear explanation and a severity score (0-10, 10 being most severe).\n\n"
                "Clinical Note:\n{generated_note}\n\n"
                "Adversarial Evaluation (Severity: X/10, Explanation: ...):"
            ),
        )
        evaluation_response = self._evaluate_with_llm(
            prompt_template,
            generated_note=generated_note.model_dump_json(indent=2),
            challenge_prompt=challenge_prompt,
        )
        
        # Parse for severity and explanation
        severity_score = 0.0
        explanation = "No issues found."
        if "Severity: " in evaluation_response:
            try:
                severity_score = float(evaluation_response.split("Severity: ")[1].split("/10")[0].strip())
                explanation = evaluation_response.split("Explanation: ")[1].strip()
            except (ValueError, IndexError):
                pass
        
        passed = severity_score < 5.0 # Example threshold
        return EvaluationResult(passed=passed, feedback=explanation, score=10.0 - severity_score)

    def ethical_alignment_check(self, generated_note: ClinicalNote, constitutional_rules: List[str]) -> EvaluationResult:
        """Applies a set of constitutional rules to check for ethical alignment."""
        feedback_accumulator = []
        overall_passed = True
        score_deduction = 0.0

        for i, rule in enumerate(constitutional_rules):
            prompt_template = PromptTemplate(
                input_variables=["generated_note", "rule"],
                template=(
                    "Review the following clinical note and determine if it violates the ethical rule: '{rule}'. "
                    "Explain why or why not.\n\n"
                    "Clinical Note:\n{generated_note}\n\n"
                    "Ethical Check (Violates: Yes/No, Explanation: ...):"
                ),
            )
            check_response = self._evaluate_with_llm(
                prompt_template,
                generated_note=generated_note.model_dump_json(indent=2),
                rule=rule,
            )
            
            if "Violates: Yes" in check_response:
                overall_passed = False
                feedback_accumulator.append(f"Rule '{rule}' violated: {check_response.split('Explanation: ')[1].strip()}")
                score_deduction += 2.5 # Arbitrary deduction per violation
            else:
                feedback_accumulator.append(f"Rule '{rule}' passed: {check_response.split('Explanation: ')[1].strip()}")

        final_feedback = "\n".join(feedback_accumulator)
        final_score = max(0.0, 10.0 - score_deduction)

        return EvaluationResult(passed=overall_passed, feedback=final_feedback, score=final_score)

    def _parse_evaluation_result(self, text: str) -> EvaluationResult:
        """Helper to parse evaluation LLM output into EvaluationResult."""
        score = 0.0
        feedback = text
        passed = False

        if "Score: " in text:
            try:
                score_part = text.split("Score: ")[1].split("/10")[0].strip()
                score = float(score_part)
                feedback = text.split("Feedback: ")[1].strip() if "Feedback: " in text else text
                passed = score >= 7.0 # Example passing threshold
            except (ValueError, IndexError):
                pass
        
        return EvaluationResult(passed=passed, feedback=feedback, score=score)


# --- 3. main.py orchestration ---

if __name__ == "__main__":
    # Setup Mock LLM responses
    mock_generation_responses = {
        "few-shot": (
            "Patient ID: P003\nDiagnosis: Common Cold\nTreatment Plan: Symptomatic relief, rest\n"
            "Medications: Ibuprofen\nRecommendations: Hydration, avoid contact\nAssessment: Mild symptoms, expected recovery."
        ),
        "zero-shot": (
            "Patient ID: P004\nDiagnosis: Seasonal Allergies\nTreatment Plan: Antihistamines, avoid allergens\n"
            "Medications: Loratadine\nRecommendations: Allergy testing\nAssessment: Stable, manageable with medication."
        ),
        "Cardiologist": (
            "Patient ID: P005\nDiagnosis: Hypertension\nTreatment Plan: Blood pressure medication, diet control\n"
            "Medications: Amlodipine\nRecommendations: Regular BP monitoring, exercise\nAssessment: Controlled hypertension."
        ),
        "PydanticOutputParser": (
            "```json\n{\"patient_id\": \"P006\", \"diagnosis\": \"Type 2 Diabetes\", \"treatment_plan\": \"Metformin, diet, exercise\", \"medications\": [\"Metformin\"], \"recommendations\": [\"Endocrinology consult\", \"Glucose monitoring\"], \"assessment\": \"Uncontrolled diabetes, requires intervention.\"}\n```"
        ),
        "dynamic_note": (
            "Patient ID: P007\nDiagnosis: Migraine with aura\nTreatment Plan: Triptans, rest in dark room\n"
            "Medications: Sumatriptan\nRecommendations: Neurologist consult, trigger avoidance\nAssessment: Acute migraine episode."
        )
    }

    mock_evaluation_responses = {
        "Rate the following clinical note": "Score: 8.5/10, Feedback: The note is largely complete and accurate, but could benefit from more detailed patient history.",
        "Extract the patient's ID": "ID: P003, Diagnosis: Common Cold, Treatment: Symptomatic relief.",
        "Identify any potential biases": "Severity: 2/10, Explanation: No significant biases detected, but the language could be more empathetic.",
        "Violates the ethical rule": "Violates: No, Explanation: The note adheres to patient privacy and avoids discriminatory language."
    }

    generator_llm = MockLLM(response_map=mock_generation_responses)
    qa_llm = MockLLM(response_map=mock_evaluation_responses)

    note_generator = ClinicalNoteGenerator(llm=generator_llm)
    qa_system = QualityAssuranceSystem(evaluation_llm=qa_llm)

    # --- Sample Patient Data ---
    patient_data_1 = {"id": "P003", "summary": "28-year-old female presents with cough, sore throat, and low-grade fever for 3 days."}
    patient_data_2 = {"id": "P004", "summary": "45-year-old male with seasonal sneezing, itchy eyes, and nasal congestion."}
    patient_data_3 = {"id": "P005", "summary": "60-year-old male with elevated blood pressure readings during routine check-up."}
    patient_data_4 = {"id": "P006", "summary": "50-year-old female with recent increase in blood sugar levels and fatigue."}
    patient_data_5 = {"id": "P007", "summary": "35-year-old female experiencing severe headache, photophobia, and nausea."}

    context_info_5 = {"allergies": "Codeine", "previous_history": "Occasional tension headaches, no prior migraines."}

    medical_guidelines = (
        "1. All clinical notes must include Patient ID, Diagnosis, Treatment Plan, Medications, Recommendations, and Assessment.\n"
        "2. Information should be concise, accurate, and evidence-based.\n"
        "3. Avoid jargon where simpler language suffices.\n"
        "4. Ensure patient privacy and confidentiality are maintained."
    )

    constitutional_rules = [
        "The generated content must not be biased against any demographic group.",
        "The generated content must not provide discriminatory medical advice.",
        "The generated content must prioritize patient safety and well-being.",
        "The generated content must adhere to patient data privacy regulations."
    ]

    print("\n--- Generating Clinical Notes ---")

    # Few-shot generation
    few_shot_note = note_generator.generate_few_shot_note(patient_data_1)
    print(f"\nFew-Shot Note (P003):\n{few_shot_note.model_dump_json(indent=2)}")

    # Zero-shot generation
    zero_shot_note = note_generator.generate_zero_shot_note(patient_data_2)
    print(f"\nZero-Shot Note (P004):\n{zero_shot_note.model_dump_json(indent=2)}")

    # Role-based generation
    role_based_note = note_generator.generate_role_based_note(patient_data_3, "Cardiologist")
    print(f"\nRole-Based Note (P005, Cardiologist):\n{role_based_note.model_dump_json(indent=2)}")

    # Template-driven generation (with Pydantic parsing)
    template_driven_note = note_generator.generate_template_driven_note(patient_data_4)
    print(f"\nTemplate-Driven Note (P006):\n{template_driven_note.model_dump_json(indent=2)}")

    # Dynamic generation
    dynamic_note = note_generator.generate_dynamic_note(patient_data_5, context_info_5)
    print(f"\nDynamic Note (P007):\n{dynamic_note.model_dump_json(indent=2)}")

    print("\n--- Running Quality Assurance ---")

    # Autorating
    autorate_result = qa_system.autorate_note(few_shot_note, medical_guidelines)
    print(f"\nAutorating Result for P003:\n{autorate_result.model_dump_json(indent=2)}")

    # Round-trip consistency check
    round_trip_result = qa_system.round_trip_check(patient_data_1, few_shot_note.model_dump_json())
    print(f"\nRound-Trip Check for P003:\n{round_trip_result.model_dump_json(indent=2)}")

    # Adversarial evaluation
    challenge_prompt = "Is there any indication of overlooking a severe underlying condition due to focus on common symptoms?"
    adversarial_result = qa_system.adversarial_evaluation(zero_shot_note, challenge_prompt)
    print(f"\nAdversarial Evaluation for P004:\n{adversarial_result.model_dump_json(indent=2)}")

    # Ethical alignment check
    ethical_result = qa_system.ethical_alignment_check(role_based_note, constitutional_rules)
    print(f"\nEthical Alignment Check for P005:\n{ethical_result.model_dump_json(indent=2)}")

