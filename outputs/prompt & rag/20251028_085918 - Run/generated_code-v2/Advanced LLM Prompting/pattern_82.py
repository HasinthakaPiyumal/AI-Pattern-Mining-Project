from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser


class ClarificationOutput(BaseModel):
    clarified_question: str = Field(description="The rephrased and clarified medical question.")
    information_gaps: list[str] = Field(description="Any identified information gaps or further questions needed.")

class PreliminaryJudgmentOutput(BaseModel):
    differential_diagnoses: list[str] = Field(description="A list of initial differential diagnoses.")
    rationale: str = Field(description="The reasoning behind the preliminary judgments.")

class EvaluationOutput(BaseModel):
    inconsistencies_found: list[str] = Field(description="List of inconsistencies found when evaluating against new data.")
    confirmations_found: list[str] = Field(description="List of preliminary findings confirmed by new data.")
    missing_information: list[str] = Field(description="Suggestions for additional information needed.")
    revised_diagnoses_considerations: list[str] = Field(description="Considerations for revising diagnoses based on evaluation.")

class DecisionConfirmationOutput(BaseModel):
    final_diagnoses: list[str] = Field(description="The confirmed final diagnosis or top few diagnoses.")
    justification: str = Field(description="Comprehensive justification for the final diagnoses.")
    treatment_recommendations: list[str] = Field(description="Suggested treatment recommendations or next steps.")

class ConfidenceAssessmentOutput(BaseModel):
    confidence_score: float = Field(description="A confidence score for the final diagnosis (0.0 to 1.0).")
    uncertainties: list[str] = Field(description="Remaining uncertainties or risks.")
    further_tests_suggested: list[str] = Field(description="Suggestions for further diagnostic tests.")
    explanation: str = Field(description="Explanation for the given confidence score.")


class MedicalDiagnosticAssistant:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.5):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        # 1. Clarification Prompt Chain
        self.clarification_parser = PydanticOutputParser(pydantic_object=ClarificationOutput)
        clarification_prompt = PromptTemplate(
            template="""As a medical assistant, clarify the following patient information and medical question. Identify any potential information gaps. Format your response as a JSON object with 'clarified_question' and 'information_gaps'.\nPatient info: {patient_data}\nMedical question: {medical_question}\n{format_instructions}""",
            input_variables=["patient_data", "medical_question"],
            partial_variables={"format_instructions": self.clarification_parser.get_format_instructions()},
        )
        self.clarification_chain = LLMChain(llm=self.llm, prompt=clarification_prompt, output_parser=self.clarification_parser)

        # 2. Preliminary Judgment Prompt Chain
        self.preliminary_parser = PydanticOutputParser(pydantic_object=PreliminaryJudgmentOutput)
        preliminary_prompt = PromptTemplate(
            template="""Based on the clarified medical question: {clarified_question} and patient information: {patient_data}, provide a preliminary set of differential diagnoses and a rationale. Format your response as a JSON object with 'differential_diagnoses' and 'rationale'.\n{format_instructions}""",
            input_variables=["clarified_question", "patient_data"],
            partial_variables={"format_instructions": self.preliminary_parser.get_format_instructions()},
        )
        self.preliminary_chain = LLMChain(llm=self.llm, prompt=preliminary_prompt, output_parser=self.preliminary_parser)

        # 3. Evaluation Prompt Chain
        self.evaluation_parser = PydanticOutputParser(pydantic_object=EvaluationOutput)
        evaluation_prompt = PromptTemplate(
            template="""Evaluate the preliminary diagnoses: {preliminary_diagnoses} with rationale: {preliminary_rationale} against the following additional patient data: {additional_data}. Identify inconsistencies, confirmations, missing information, and considerations for revising diagnoses. Format your response as a JSON object with 'inconsistencies_found', 'confirmations_found', 'missing_information', and 'revised_diagnoses_considerations'.\n{format_instructions}""",
            input_variables=["preliminary_diagnoses", "preliminary_rationale", "additional_data"],
            partial_variables={"format_instructions": self.evaluation_parser.get_format_instructions()},
        )
        self.evaluation_chain = LLMChain(llm=self.llm, prompt=evaluation_prompt, output_parser=self.evaluation_parser)

        # 4. Decision Confirmation Prompt Chain
        self.confirmation_parser = PydanticOutputParser(pydantic_object=DecisionConfirmationOutput)
        confirmation_prompt = PromptTemplate(
            template="""Based on the preliminary diagnoses, evaluation results ({evaluation_results}), and all patient data ({patient_data}, {additional_data}), confirm the final diagnosis (or top few), provide a comprehensive justification, and suggest treatment recommendations/next steps. Format your response as a JSON object with 'final_diagnoses', 'justification', and 'treatment_recommendations'.\n{format_instructions}""",
            input_variables=["evaluation_results", "patient_data", "additional_data"],
            partial_variables={"format_instructions": self.confirmation_parser.get_format_instructions()},
        )
        self.confirmation_chain = LLMChain(llm=self.llm, prompt=confirmation_prompt, output_parser=self.confirmation_parser)

        # 5. Confidence Assessment Prompt Chain
        self.confidence_parser = PydanticOutputParser(pydantic_object=ConfidenceAssessmentOutput)
        confidence_prompt = PromptTemplate(
            template="""Assess your confidence in the final diagnoses: {final_diagnoses} with justification: {justification}. Provide a confidence score (0.0 to 1.0), list any remaining uncertainties or risks, and suggest further diagnostic tests if applicable. Explain the confidence score. Format your response as a JSON object with 'confidence_score', 'uncertainties', 'further_tests_suggested', and 'explanation'.\n{format_instructions}""",
            input_variables=["final_diagnoses", "justification"],
            partial_variables={"format_instructions": self.confidence_parser.get_format_instructions()},
        )
        self.confidence_chain = LLMChain(llm=self.llm, prompt=confidence_prompt, output_parser=self.confidence_parser)

    def run_diagnosis(self, patient_data: str, medical_question: str, additional_data: str = ""):
        # Step 1: Clarification
        clarification_output: ClarificationOutput = self.clarification_chain.invoke({"patient_data": patient_data, "medical_question": medical_question})["text"]
        
        # Step 2: Preliminary Judgment
        preliminary_output: PreliminaryJudgmentOutput = self.preliminary_chain.invoke({
            "clarified_question": clarification_output.clarified_question,
            "patient_data": patient_data
        })["text"]
        
        # Step 3: Evaluation
        evaluation_output: EvaluationOutput = self.evaluation_chain.invoke({
            "preliminary_diagnoses": ", ".join(preliminary_output.differential_diagnoses),
            "preliminary_rationale": preliminary_output.rationale,
            "additional_data": additional_data
        })["text"]

        # Step 4: Decision Confirmation
        confirmation_output: DecisionConfirmationOutput = self.confirmation_chain.invoke({
            "evaluation_results": evaluation_output.model_dump_json(),
            "patient_data": patient_data,
            "additional_data": additional_data
        })["text"]

        # Step 5: Confidence Assessment
        confidence_output: ConfidenceAssessmentOutput = self.confidence_chain.invoke({
            "final_diagnoses": ", ".join(confirmation_output.final_diagnoses),
            "justification": confirmation_output.justification
        })["text"]

        return {
            "clarification": clarification_output,
            "preliminary_judgment": preliminary_output,
            "evaluation": evaluation_output,
            "decision_confirmation": confirmation_output,
            "confidence_assessment": confidence_output
        }

if __name__ == "__main__":
    # Example Usage
    assistant = MedicalDiagnosticAssistant()

    patient_history = "28-year-old female, presents with severe abdominal pain, nausea, and vomiting for 24 hours. No fever. LMP 6 weeks ago."
    initial_medical_question = "What is the most likely diagnosis?"
    
    # Simulate additional data from lab tests or imaging
    lab_results = "Beta-HCG: 1500 mIU/mL. Ultrasound report: Right adnexal mass, free fluid in cul-de-sac."

    print("\n--- Running Diagnosis ---")
    diagnosis_result = assistant.run_diagnosis(
        patient_data=patient_history,
        medical_question=initial_medical_question,
        additional_data=lab_results
    )

    print("\n--- Clarification ---")
    print(diagnosis_result["clarification"].model_dump_json(indent=2))

    print("\n--- Preliminary Judgment ---")
    print(diagnosis_result["preliminary_judgment"].model_dump_json(indent=2))
    
    print("\n--- Evaluation ---")
    print(diagnosis_result["evaluation"].model_dump_json(indent=2))

    print("\n--- Decision Confirmation ---")
    print(diagnosis_result["decision_confirmation"].model_dump_json(indent=2))

    print("\n--- Confidence Assessment ---")
    print(diagnosis_result["confidence_assessment"].model_dump_json(indent=2))

    patient_history_2 = "55-year-old male, history of smoking, presents with persistent cough, weight loss, and fatigue for 3 months."
    initial_medical_question_2 = "What could be the underlying cause of his symptoms?"
    lab_results_2 = "Chest X-ray: suspicious lesion in the left lung. Biopsy pending."

    print("\n--- Running Second Diagnosis ---")
    diagnosis_result_2 = assistant.run_diagnosis(
        patient_data=patient_history_2,
        medical_question=initial_medical_question_2,
        additional_data=lab_results_2
    )

    print("\n--- Final Diagnosis for Second Case ---")
    print(diagnosis_result_2["decision_confirmation"].model_dump_json(indent=2))
    print("\n--- Confidence for Second Case ---")
    print(diagnosis_result_2["confidence_assessment"].model_dump_json(indent=2))