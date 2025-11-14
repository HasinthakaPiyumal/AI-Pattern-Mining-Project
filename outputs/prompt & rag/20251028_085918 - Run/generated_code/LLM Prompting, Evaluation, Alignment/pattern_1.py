
import os
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain

# --- 0. Configuration and Utilities ---

load_dotenv() # Load environment variables from .env file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure OPENAI_API_KEY is set
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please create a .env file or set it manually.")

llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7) # Using a more capable model for clinical text

# --- 1. Data Models (Pydantic) ---

class PatientData(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient.")
    name: str = Field(..., description="Full name of the patient.")
    age: int = Field(..., description="Age of the patient in years.")
    gender: str = Field(..., description="Gender of the patient.")
    chief_complaint: str = Field(..., description="The primary reason for the patient's visit.")
    diagnosis: List[str] = Field(default_factory=list, description="List of diagnoses for the patient.")
    medications: List[str] = Field(default_factory=list, description="List of current medications.")
    allergies: List[str] = Field(default_factory=list, description="List of known allergies.")
    medical_history: List[str] = Field(default_factory=list, description="Relevant past medical history.")
    social_history: List[str] = Field(default_factory=list, description="Relevant social history (e.g., smoking, alcohol).")
    vitals: Dict[str, str] = Field(default_factory=dict, description="Recent vital signs (e.g., BP, HR, Temp).")

class ClinicalDocument(BaseModel):
    doc_type: str = Field(..., description="Type of the clinical document (e.g., 'patient_summary', 'discharge_instructions').")
    patient_id: str = Field(..., description="Unique identifier of the patient this document pertains to.")
    content: str = Field(..., description="The generated clinical document content.")
    qa_results: Dict[str, Any] = Field(default_factory=dict, description="Results of quality assurance checks.")

# --- 2. Prompt Engineering Module ---

class PromptEngineer:
    def __init__(self):
        self.few_shot_examples = {
            "patient_summary": [
                {
                    "input": {"name": "John Doe", "age": 65, "diagnosis": ["Hypertension"], "chief_complaint": "Routine checkup"},
                    "output": "Patient John Doe, 65, presents for a routine checkup. Diagnosed with hypertension, currently stable on medication."
                },
                {
                    "input": {"name": "Jane Smith", "age": 42, "diagnosis": ["Type 2 Diabetes"], "chief_complaint": "Follow-up on blood sugar"},
                    "output": "Patient Jane Smith, 42, with Type 2 Diabetes, seen for blood sugar follow-up. Current glucose levels being monitored."
                },
            ],
            "discharge_instructions": [
                {
                    "input": {"name": "Alice Brown", "diagnosis": ["Pneumonia"], "medications": ["Amoxicillin"]},
                    "output": "Discharge Instructions for Alice Brown: Diagnosis: Pneumonia. Medications: Continue Amoxicillin as prescribed. Follow-up with PCP in 7 days. Avoid strenuous activity."
                }
            ]
        }

        self.templates = {
            "patient_summary": "Create a concise patient summary for {name}, age {age}, presenting with {chief_complaint}. Key diagnoses: {diagnosis}. Medications: {medications}. Medical History: {medical_history}. Social History: {social_history}. Vitals: {vitals}.\n\nSummary:",
            "discharge_instructions": "Generate clear and actionable discharge instructions for patient {name} (ID: {patient_id}) with diagnosis(es): {diagnosis}. Include information about medications: {medications}, and follow-up plans. Also, mention any specific activity restrictions or important advice. Allergies: {allergies}.\n\nDischarge Instructions:",
            "referral_letter": "Draft a referral letter from {referring_doctor} to {specialist} for patient {name} (DOB: {dob}, ID: {patient_id}). Chief Complaint: {chief_complaint}. Diagnosis: {diagnosis}. Relevant history: {medical_history}. Current medications: {medications}. Purpose of referral: {purpose_of_referral}.\n\nReferral Letter:"
        }

    def _format_patient_data(self, patient_data: PatientData) -> Dict[str, Any]:
        # Convert lists to comma-separated strings for prompt injection
        formatted_data = patient_data.dict()
        for key in ['diagnosis', 'medications', 'allergies', 'medical_history', 'social_history']:
            formatted_data[key] = ", ".join(formatted_data[key]) if formatted_data[key] else "None"
        formatted_data['vitals'] = ", ".join([f"{k}: {v}" for k, v in formatted_data['vitals'].items()]) if formatted_data['vitals'] else "None"
        formatted_data['dob'] = "Unknown" # Placeholder for referral letter if not in PatientData
        formatted_data['referring_doctor'] = "Dr. General Practitioner" # Placeholder
        formatted_data['specialist'] = "Dr. Specialist" # Placeholder
        formatted_data['purpose_of_referral'] = "Further evaluation and management" # Placeholder

        return formatted_data

    def generate_prompt(self, strategy: str, doc_type: str, patient_data: PatientData) -> ChatPromptTemplate:
        formatted_patient_data = self._format_patient_data(patient_data)

        system_message = f"You are a highly accurate and ethical medical documentation AI assistant. Generate professional clinical documents based on the provided patient information. Ensure factual consistency, avoid hallucinations, and adhere to medical best practices and ethical guidelines."
        system_prompt = SystemMessagePromptTemplate.from_template(system_message)

        if strategy == "zero_shot":
            human_template = self.templates.get(doc_type, "Generate a clinical document of type {doc_type} for patient {name} with the following details: {patient_details}.")
            human_prompt = HumanMessagePromptTemplate.from_template(human_template)
            chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
            return chat_prompt.partial(**formatted_patient_data)

        elif strategy == "few_shot":
            examples = self.few_shot_examples.get(doc_type, [])
            if not examples:
                logger.warning(f"No few-shot examples for {doc_type}. Falling back to zero-shot.")
                return self.generate_prompt("zero_shot", doc_type, patient_data)

            example_prompts = []
            for example in examples:
                example_prompts.append(HumanMessagePromptTemplate.from_template(self.templates.get(doc_type, "").format(**example["input"])))
                example_prompts.append(SystemMessagePromptTemplate.from_template(example["output"]))

            human_template = self.templates.get(doc_type, "Generate a clinical document of type {doc_type} for patient {name} with the following details: {patient_details}.")
            human_prompt = HumanMessagePromptTemplate.from_template(human_template)
            chat_prompt = ChatPromptTemplate.from_messages([system_prompt] + example_prompts + [human_prompt])
            return chat_prompt.partial(**formatted_patient_data)

        elif strategy == "role_based":
            role_system_message = f"You are a senior physician specializing in {doc_type.replace('_', ' ')}. Your task is to generate precise and comprehensive clinical documents. " + system_message
            role_system_prompt = SystemMessagePromptTemplate.from_template(role_system_message)
            human_template = self.templates.get(doc_type, "Generate a clinical document of type {doc_type} for patient {name} with the following details: {patient_details}.")
            human_prompt = HumanMessagePromptTemplate.from_template(human_template)
            chat_prompt = ChatPromptTemplate.from_messages([role_system_prompt, human_prompt])
            return chat_prompt.partial(**formatted_patient_data)

        elif strategy == "template_driven":
            template_str = self.templates.get(doc_type)
            if not template_str:
                raise ValueError(f"No template found for document type: {doc_type}")
            human_prompt = HumanMessagePromptTemplate.from_template(template_str)
            chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
            return chat_prompt.partial(**formatted_patient_data)

        elif strategy == "dynamic": # Simple dynamic example: adjusts based on doc_type
            if doc_type == "patient_summary":
                dynamic_system = f"You are an expert in summarizing complex patient cases for quick review. {system_message}"
            elif doc_type == "discharge_instructions":
                dynamic_system = f"You are an expert in providing clear and empathetic discharge instructions. {system_message}"
            else:
                dynamic_system = system_message
            dynamic_system_prompt = SystemMessagePromptTemplate.from_template(dynamic_system)

            human_template = self.templates.get(doc_type, "Generate a clinical document of type {doc_type} for patient {name} with the following details: {patient_details}.")
            human_prompt = HumanMessagePromptTemplate.from_template(human_template)
            chat_prompt = ChatPromptTemplate.from_messages([dynamic_system_prompt, human_prompt])
            return chat_prompt.partial(**formatted_patient_data)

        else:
            raise ValueError(f"Unknown prompt strategy: {strategy}")

# --- 3. Documentation Generation Engine ---

class DocumentationGenerator:
    def __init__(self, llm: ChatOpenAI, prompt_engineer: PromptEngineer):
        self.llm = llm
        self.prompt_engineer = prompt_engineer

    def generate_clinical_document(self, patient_data: PatientData, doc_type: str, prompt_strategy: str = "template_driven") -> ClinicalDocument:
        logger.info(f"Generating {doc_type} for patient {patient_data.name} using {prompt_strategy} strategy.")
        try:
            prompt = self.prompt_engineer.generate_prompt(strategy=prompt_strategy, doc_type=doc_type, patient_data=patient_data)
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.run()
            return ClinicalDocument(doc_type=doc_type, patient_id=patient_data.patient_id, content=response)
        except Exception as e:
            logger.error(f"Error generating document: {e}")
            return ClinicalDocument(doc_type=doc_type, patient_id=patient_data.patient_id, content=f"Error generating document: {e}", qa_results={"generation_error": True, "error_message": str(e)})

# --- 4. Quality Assurance & Evaluation Framework ---

class QualityAssuranceFramework:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.autorating_prompt_template = PromptTemplate(
            template="""You are an expert medical reviewer. Evaluate the following clinical document for clarity, completeness, factual correctness, and professionalism. Assign a score from 1 (poor) to 5 (excellent) for each criterion and provide a brief justification. Also, identify any potential factual inconsistencies or hallucinations.

Document Type: {doc_type}
Patient ID: {patient_id}
Document Content:
```
{document_content}
```

Evaluation Criteria:
- Clarity (1-5):
- Completeness (1-5):
- Factual Correctness (1-5):
- Professionalism (1-5):

Justification and Potential Issues:
""",
            input_variables=["doc_type", "patient_id", "document_content"]
        )

        self.adversarial_prompt_template = PromptTemplate(
            template="""As a skeptical medical expert, review the following clinical document and try to find any potential factual errors, contradictions, or misleading statements. Focus on challenging the truthfulness of the content against general medical knowledge. If you find issues, describe them concisely.

Document Type: {doc_type}
Patient ID: {patient_id}
Document Content:
```
{document_content}
```

Adversarial Review Findings (e.g., inconsistencies, potential hallucinations):
""",
            input_variables=["doc_type", "patient_id", "document_content"]
        )

        self.ethical_alignment_prompt_template = PromptTemplate(
            template="""Review the following clinical document for adherence to ethical principles in healthcare, focusing on potential biases, privacy concerns (if applicable in context), or inappropriate language. Does it maintain a respectful and unbiased tone? Suggest improvements if any ethical concerns are found.

Document Type: {doc_type}
Patient ID: {patient_id}
Document Content:
```
{document_content}
```

Ethical Alignment Review Findings (e.g., bias, privacy, tone):
""",
            input_variables=["doc_type", "patient_id", "document_content"]
        )

    def evaluate_document_autorating(self, document: ClinicalDocument) -> Dict[str, Any]:
        logger.info(f"Performing LLM-based autorating for {document.doc_type} (ID: {document.patient_id}).")
        try:
            chain = LLMChain(llm=self.llm, prompt=self.autorating_prompt_template)
            response = chain.run(doc_type=document.doc_type, patient_id=document.patient_id, document_content=document.content)
            # Simple parsing of the response to extract scores and justification
            results = {"raw_llm_response": response}
            for line in response.split('\n'):
                if 'Clarity' in line: results['clarity'] = line.strip()
                elif 'Completeness' in line: results['completeness'] = line.strip()
                elif 'Factual Correctness' in line: results['factual_correctness'] = line.strip()
                elif 'Professionalism' in line: results['professionalism'] = line.strip()
                elif 'Justification' in line: results['justification'] = line.replace('Justification and Potential Issues:', '').strip()
            return results
        except Exception as e:
            logger.error(f"Error in autorating: {e}")
            return {"error": str(e), "autorating_failed": True}

    def evaluate_document_consistency(self, original_patient_data: PatientData, document: ClinicalDocument) -> Dict[str, Any]:
        logger.info(f"Performing consistency check for {document.doc_type} (ID: {document.patient_id}).")
        consistency_issues = []

        # Check if key patient data is mentioned in the document
        if original_patient_data.name not in document.content:
            consistency_issues.append(f"Patient name '{original_patient_data.name}' not explicitly found.")
        if not any(diag in document.content for diag in original_patient_data.diagnosis):
            consistency_issues.append(f"Some diagnoses ({', '.join(original_patient_data.diagnosis)}) might be missing or not clearly stated.")
        if original_patient_data.chief_complaint and original_patient_data.chief_complaint not in document.content:
            consistency_issues.append(f"Chief complaint '{original_patient_data.chief_complaint}' not explicitly found.")

        # A more robust check would involve extracting structured data from the generated document
        # and comparing it to the original PatientData object.
        # This is a simplified keyword-based check.

        return {"is_consistent": not bool(consistency_issues), "issues": consistency_issues}

    def evaluate_document_adversarial(self, document: ClinicalDocument) -> Dict[str, Any]:
        logger.info(f"Performing adversarial evaluation for {document.doc_type} (ID: {document.patient_id}).")
        try:
            chain = LLMChain(llm=self.llm, prompt=self.adversarial_prompt_template)
            response = chain.run(doc_type=document.doc_type, patient_id=document.patient_id, document_content=document.content)
            return {"raw_llm_response": response, "adversarial_findings": response.strip()}
        except Exception as e:
            logger.error(f"Error in adversarial evaluation: {e}")
            return {"error": str(e), "adversarial_failed": True}

    def evaluate_document_ethical_alignment(self, document: ClinicalDocument) -> Dict[str, Any]:
        logger.info(f"Performing ethical alignment check for {document.doc_type} (ID: {document.patient_id}).")
        try:
            chain = LLMChain(llm=self.llm, prompt=self.ethical_alignment_prompt_template)
            response = chain.run(doc_type=document.doc_type, patient_id=document.patient_id, document_content=document.content)
            return {"raw_llm_response": response, "ethical_findings": response.strip()}
        except Exception as e:
            logger.error(f"Error in ethical alignment evaluation: {e}")
            return {"error": str(e), "ethical_check_failed": True}

    def perform_full_qa(self, original_patient_data: PatientData, generated_document: ClinicalDocument) -> ClinicalDocument:
        logger.info(f"Starting full QA for document (ID: {generated_document.patient_id}).")
        qa_results = {}
        qa_results['autorating'] = self.evaluate_document_autorating(generated_document)
        qa_results['consistency'] = self.evaluate_document_consistency(original_patient_data, generated_document)
        qa_results['adversarial'] = self.evaluate_document_adversarial(generated_document)
        qa_results['ethical_alignment'] = self.evaluate_document_ethical_alignment(generated_document)

        generated_document.qa_results = qa_results
        logger.info(f"Full QA completed for document (ID: {generated_document.patient_id}).")
        return generated_document

# --- 5. User Interface (Streamlit Prototype) ---

# To run this, you'll need Streamlit installed: pip install streamlit
# Save this file as clinical_doc_assistant.py and run: streamlit run clinical_doc_assistant.py

import streamlit as st

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="AI Clinical Doc Assistant")
    st.title("🤖 AI-Powered Clinical Documentation Assistant")
    st.subheader("Generate and Quality Assure Clinical Documents")

    # Initialize components
    prompt_engineer = PromptEngineer()
    doc_generator = DocumentationGenerator(llm=llm, prompt_engineer=prompt_engineer)
    qa_framework = QualityAssuranceFramework(llm=llm)

    with st.sidebar:
        st.header("Patient Information")
        patient_id = st.text_input("Patient ID", "P12345")
        patient_name = st.text_input("Patient Name", "Alice Wonderland")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=30)
        patient_gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        chief_complaint = st.text_area("Chief Complaint", "Severe headache and blurred vision for 2 days.")

        diagnosis_input = st.text_area("Diagnoses (comma-separated)", "Migraine with aura, Possible hypertension")
        medications_input = st.text_area("Medications (comma-separated)", "Ibuprofen 600mg PRN, Metoprolol 25mg BID")
        allergies_input = st.text_area("Allergies (comma-separated)", "Penicillin")
        medical_history_input = st.text_area("Medical History (comma-separated)", "History of migraines, Family history of hypertension")
        social_history_input = st.text_area("Social History (comma-separated)", "Non-smoker, Social alcohol use")
        vitals_input = st.text_area("Vitals (e.g., BP: 140/90, HR: 85, Temp: 99.2F)", "BP: 140/90 mmHg, HR: 85 bpm, Temp: 99.2F")

        doc_type_options = list(prompt_engineer.templates.keys())
        document_type = st.selectbox("Document Type", doc_type_options)

        prompt_strategy_options = ["template_driven", "zero_shot", "few_shot", "role_based", "dynamic"]
        prompt_strategy = st.selectbox("Prompt Strategy", prompt_strategy_options)

        generate_button = st.button("Generate Document")

    st.header("Generated Clinical Document")

    if generate_button:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Error: OPENAI_API_KEY is not set. Please set it in your environment variables or in a .env file.")
        else:
            with st.spinner("Generating document and performing QA..."):
                try:
                    # Parse inputs
                    diagnosis = [d.strip() for d in diagnosis_input.split(',') if d.strip()]
                    medications = [m.strip() for m in medications_input.split(',') if m.strip()]
                    allergies = [a.strip() for a in allergies_input.split(',') if a.strip()]
                    medical_history = [h.strip() for h in medical_history_input.split(',') if h.strip()]
                    social_history = [s.strip() for s in social_history_input.split(',') if s.strip()]
                    vitals = {}
                    for item in vitals_input.split(','):
                        if ':' in item:
                            key, value = item.split(':', 1)
                            vitals[key.strip()] = value.strip()

                    patient_data = PatientData(
                        patient_id=patient_id,
                        name=patient_name,
                        age=patient_age,
                        gender=patient_gender,
                        chief_complaint=chief_complaint,
                        diagnosis=diagnosis,
                        medications=medications,
                        allergies=allergies,
                        medical_history=medical_history,
                        social_history=social_history,
                        vitals=vitals
                    )

                    # Generate Document
                    generated_doc = doc_generator.generate_clinical_document(
                        patient_data=patient_data,
                        doc_type=document_type,
                        prompt_strategy=prompt_strategy
                    )

                    st.markdown("### Document Content")
                    st.text_area("", generated_doc.content, height=400)

                    # Perform QA
                    if not generated_doc.qa_results.get("generation_error"):
                        final_doc = qa_framework.perform_full_qa(patient_data, generated_doc)
                        st.markdown("### Quality Assurance Results")

                        st.subheader("LLM-based Autorating")
                        autorating = final_doc.qa_results.get('autorating', {})
                        st.json(autorating)

                        st.subheader("Consistency Check")
                        consistency = final_doc.qa_results.get('consistency', {})
                        st.json(consistency)

                        st.subheader("Adversarial Evaluation")
                        adversarial = final_doc.qa_results.get('adversarial', {})
                        st.json(adversarial)

                        st.subheader("Ethical Alignment Review")
                        ethical = final_doc.qa_results.get('ethical_alignment', {})
                        st.json(ethical)
                    else:
                        st.error(f"Document generation failed: {generated_doc.qa_results.get('error_message')}")

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    logger.exception("Error during document generation or QA in Streamlit app.")
    else:
        st.info("Enter patient information and click 'Generate Document' to begin.")

