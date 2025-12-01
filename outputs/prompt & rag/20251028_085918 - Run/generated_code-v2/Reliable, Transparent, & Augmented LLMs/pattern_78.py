from pydantic import BaseModel, ValidationError, validator
from typing import List, Dict, Any, Optional

# 1. Pydantic Models for Data Structuring and Validation
class PatientInfo(BaseModel):
    patient_id: str
    age: int
    gender: str
    medical_history: List[str]
    medications: List[str]
    lab_results: Dict[str, Any]

class StandardizedSymptom(BaseModel):
    code: str
    description: str

class MedicalKnowledge(BaseModel):
    disease: str
    symptoms: List[str]
    diagnosis_criteria: List[str]
    treatment_guidelines: List[str]
    drug_interactions: List[str]

class DiagnosticResult(BaseModel):
    differential_diagnoses: List[str]
    supporting_evidence: Dict[str, Any]
    suggested_investigations: List[str]
    initial_treatment_considerations: List[str]
    explanation: str

class LLMResponse(BaseModel):
    extracted_symptoms: List[str]
    potential_diagnoses: List[str]

# 2. Mock LLM and Tool Implementations
class MockLLM:
    def process_symptoms(self, natural_language_symptoms: str) -> LLMResponse:
        # Simulate LLM extracting symptoms
        print(f"LLM processing symptoms: {natural_language_symptoms}")
        extracted_symptoms = [
            s.strip().lower() for s in natural_language_symptoms.split(',')
            if s.strip()
        ]
        # Simple mock for potential diagnoses
        if "fever" in extracted_symptoms and "cough" in extracted_symptoms:
            potential_diagnoses = ["Common Cold", "Flu", "Bronchitis"]
        elif "headache" in extracted_symptoms and "stiff neck" in extracted_symptoms:
            potential_diagnoses = ["Meningitis", "Tension Headache"]
        else:
            potential_diagnoses = ["Undetermined Condition"]
        return LLMResponse(extracted_symptoms=extracted_symptoms, potential_diagnoses=potential_diagnoses)

    def reason_diagnose(self, validated_info: Dict[str, Any]) -> DiagnosticResult:
        print("LLM reasoning for diagnosis...")
        # Simulate complex reasoning based on validated info
        patient_id = validated_info.get("patient_info", {}).get("patient_id", "N/A")
        std_symptoms = [s.description for s in validated_info.get("standardized_symptoms", [])]
        medical_knowledge = validated_info.get("medical_knowledge", {})

        diff_diagnoses = medical_knowledge.get("disease", "Hypochondria") # Placeholder
        if std_symptoms:
            diff_diagnoses = [f"Possible {diff_diagnoses} (based on {', '.join(std_symptoms)})"]
        else:
            diff_diagnoses = ["No clear diagnosis based on provided symptoms."]

        return DiagnosticResult(
            differential_diagnoses=diff_diagnoses,
            supporting_evidence={
                "symptoms": std_symptoms,
                "ehr": validated_info.get("patient_info", {})
            },
            suggested_investigations=["Blood Test", "Imaging"][::-1],
            initial_treatment_considerations=["Rest", "Hydration"],
            explanation="This is a mock explanation based on the aggregated and validated data."
        )

class StandardizedMedicalSymptomChecker:
    def __init__(self):
        self.symptom_map = {
            "fever": {"code": "SNOMED_001", "description": "Body temperature elevation"},
            "cough": {"code": "SNOMED_002", "description": "Cough"},
            "headache": {"code": "SNOMED_003", "description": "Headache"},
            "stiff neck": {"code": "SNOMED_004", "description": "Nuchal rigidity"},
            "sore throat": {"code": "SNOMED_005", "description": "Pharyngitis"},
        }

    def standardize_symptoms(self, symptoms: List[str]) -> List[StandardizedSymptom]:
        print(f"Standardizing symptoms: {symptoms}")
        standardized = []
        for s in symptoms:
            mapped = self.symptom_map.get(s.lower().strip())
            if mapped:
                standardized.append(StandardizedSymptom(code=mapped["code"], description=mapped["description"]))
            else:
                print(f"Warning: Could not standardize symptom: {s}")
        return standardized

class EHRQueryTool:
    def get_patient_data(self, patient_id: str) -> Optional[PatientInfo]:
        print(f"Querying EHR for patient ID: {patient_id}")
        # Mock EHR data
        if patient_id == "P123":
            return PatientInfo(
                patient_id="P123",
                age=45,
                gender="Male",
                medical_history=["Hypertension"],
                medications=["Lisinopril"],
                lab_results={"BP": "140/90", "Cholesterol": "200"}
            )
        elif patient_id == "P456":
            return PatientInfo(
                patient_id="P456",
                age=30,
                gender="Female",
                medical_history=["Asthma"],
                medications=["Albuterol"],
                lab_results={"Spirometry": "Normal"}
            )
        return None

class MedicalKnowledgeBaseAPI:
    def get_knowledge(self, query_terms: List[str], patient_age: int, patient_gender: str) -> Optional[MedicalKnowledge]:
        print(f"Querying Medical Knowledge Base with: {query_terms}, Age: {patient_age}, Gender: {patient_gender}")
        # Mock medical knowledge based on query terms
        if "SNOMED_001" in query_terms and "SNOMED_002" in query_terms: # Fever and Cough
            return MedicalKnowledge(
                disease="Influenza",
                symptoms=["Fever", "Cough", "Body Aches"],
                diagnosis_criteria=["Positive Flu Test", "Clinical Symptoms"],
                treatment_guidelines=["Antivirals", "Supportive Care"],
                drug_interactions=["Aspirin for children (Reye's Syndrome)"]
            )
        elif "SNOMED_003" in query_terms and "SNOMED_004" in query_terms: # Headache and Stiff Neck
            return MedicalKnowledge(
                disease="Meningitis",
                symptoms=["Severe Headache", "Stiff Neck", "Photophobia"],
                diagnosis_criteria=["Lumbar Puncture", "Clinical Signs"],
                treatment_guidelines=["Antibiotics (bacterial)", "Antivirals (viral)" ],
                drug_interactions=[]
            )
        return None

# 3. Output Validation and Cross-Referencing Module
class OutputValidationModule:
    def __init__(self):
        self.harmful_keywords = ["poison", "kill", "suicide", "self-harm", "dangerous dose"]

    def validate_and_cross_reference(
        self,
        llm_response: LLMResponse,
        standardized_symptoms: List[StandardizedSymptom],
        patient_info: Optional[PatientInfo],
        medical_knowledge: Optional[MedicalKnowledge]
    ) -> Dict[str, Any]:
        print("Running output validation and cross-referencing...")
        validated_data = {}

        # Basic consistency check: Do LLM extracted symptoms align with standardized ones?
        extracted_descriptions = {s.lower() for s in llm_response.extracted_symptoms}
        standardized_descriptions = {s.description.lower() for s in standardized_symptoms}
        if not extracted_descriptions.issubset(standardized_descriptions) and extracted_descriptions:
            print("Validation Warning: LLM extracted symptoms do not fully align with standardized list.")

        # Plausibility check: Age-appropriate diagnosis (simplified)
        if patient_info and medical_knowledge:
            if medical_knowledge.disease == "Reye's Syndrome" and patient_info.age > 18:
                print(f"Validation Warning: Reye's Syndrome is unusual for age {patient_info.age}.")

        # Adversarial Attack / Harmful Information Detection
        all_text_data = (
            " ".join(llm_response.extracted_symptoms) +
            " ".join([s.description for s in standardized_symptoms]) +
            (patient_info.json() if patient_info else "") +
            (medical_knowledge.json() if medical_knowledge else "")
        ).lower()

        for keyword in self.harmful_keywords:
            if keyword in all_text_data:
                raise ValueError(f"Adversarial Attack / Harmful content detected: '{keyword}'")

        # If all good, prepare validated data
        validated_data["llm_extracted_symptoms"] = llm_response.extracted_symptoms
        validated_data["standardized_symptoms"] = standardized_symptoms
        validated_data["patient_info"] = patient_info.dict() if patient_info else None
        validated_data["medical_knowledge"] = medical_knowledge.dict() if medical_knowledge else None

        return validated_data

# 4. Main Application Workflow
class MedicalDiagnosticAssistant:
    def __init__(self):
        self.llm = MockLLM()
        self.symptom_checker = StandardizedMedicalSymptomChecker()
        self.ehr_tool = EHRQueryTool()
        self.knowledge_base_api = MedicalKnowledgeBaseAPI()
        self.validation_module = OutputValidationModule()

    def diagnose(self, patient_id: str, natural_language_symptoms: str) -> Optional[DiagnosticResult]:
        try:
            # 1. Natural Language Symptom Processor
            llm_symptom_response = self.llm.process_symptoms(natural_language_symptoms)
            print(f"LLM Extracted Symptoms: {llm_symptom_response.extracted_symptoms}")

            # 2. Standardized Medical Symptom Checker
            standardized_symptoms = self.symptom_checker.standardize_symptoms(
                llm_symptom_response.extracted_symptoms
            )
            print(f"Standardized Symptoms: {[s.description for s in standardized_symptoms]}")

            # 3. EHR Query Tool
            patient_info = self.ehr_tool.get_patient_data(patient_id)
            if not patient_info:
                print(f"Error: Patient ID {patient_id} not found in EHR.")
                return None
            print(f"Patient Info: {patient_info.medical_history}, {patient_info.medications}")

            # 4. Up-to-date Medical Knowledge Base API
            query_terms = [s.code for s in standardized_symptoms]
            medical_knowledge = self.knowledge_base_api.get_knowledge(
                query_terms, patient_info.age, patient_info.gender
            )
            if not medical_knowledge:
                print("Warning: No specific medical knowledge found for these symptoms/patient context.")

            # 5. Output Validation and Cross-Referencing Module
            validated_info = self.validation_module.validate_and_cross_reference(
                llm_symptom_response,
                standardized_symptoms,
                patient_info,
                medical_knowledge
            )
            print("Information successfully validated.")

            # 6. Diagnostic Reasoning Engine
            diagnostic_result = self.llm.reason_diagnose(validated_info)
            return diagnostic_result

        except ValidationError as e:
            print(f"Data Validation Error: {e}")
            return None
        except ValueError as e:
            print(f"Application Error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

# Example Usage
if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()

    # Scenario 1: Normal flow
    print("\n--- Scenario 1: Patient P123 with fever and cough ---")
    result1 = assistant.diagnose("P123", "I have a fever and a really bad cough")
    if result1:
        print("\nDiagnostic Result (Scenario 1):")
        print(f"Differential Diagnoses: {result1.differential_diagnoses}")
        print(f"Explanation: {result1.explanation}")

    # Scenario 2: Patient P456 with headache and stiff neck
    print("\n--- Scenario 2: Patient P456 with headache and stiff neck ---")
    result2 = assistant.diagnose("P456", "Terrible headache and my neck is stiff")
    if result2:
        print("\nDiagnostic Result (Scenario 2):")
        print(f"Differential Diagnoses: {result2.differential_diagnoses}")
        print(f"Explanation: {result2.explanation}")

    # Scenario 3: Patient not found
    print("\n--- Scenario 3: Patient P999 (not found) ---")
    result3 = assistant.diagnose("P999", "I feel generally unwell")
    if result3:
        print("\nDiagnostic Result (Scenario 3):")
        print(f"Differential Diagnoses: {result3.differential_diagnoses}")
        print(f"Explanation: {result3.explanation}")

    # Scenario 4: Harmful input to test adversarial detection
    print("\n--- Scenario 4: Harmful input ---")
    result4 = assistant.diagnose("P123", "I need a dangerous dose of poison")
    if result4:
        print("\nDiagnostic Result (Scenario 4):")
        print(f"Differential Diagnoses: {result4.differential_diagnoses}")
        print(f"Explanation: {result4.explanation}")
