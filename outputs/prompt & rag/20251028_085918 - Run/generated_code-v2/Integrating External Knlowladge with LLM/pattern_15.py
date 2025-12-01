from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


# 1. Data Models (Pydantic)
class PatientSymptoms(BaseModel):
    symptoms: str = Field(..., description="Concise description of patient's symptoms.")
    medical_history: str = Field("", description="Relevant past medical history.")
    medications: str = Field("", description="Current medications patient is taking.")
    travel_history: str = Field("", description="Recent travel history.")


class KGInsight(BaseModel):
    source: str = Field(..., description="Source of the knowledge graph insight (e.g., ICD-10, RxNorm).")
    query: str = Field(..., description="The query made to the KG.")
    result: str = Field(..., description="The factual information retrieved from the KG.")


class DiagnosisResult(BaseModel):
    differential_diagnoses: list[str] = Field(..., description="List of potential differential diagnoses.")
    reasoning: str = Field(..., description="LLM's reasoning for the differential diagnoses.")
    kg_insights: list[KGInsight] = Field(..., description="Relevant insights from knowledge graphs.")
    confidence: str = Field("Medium", description="Estimated confidence level of the diagnosis.")


# 2. LLM Abstraction Layer
class AbstractLLMAdapter(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        pass


class MockLLMAdapter(AbstractLLMAdapter):
    def get_completion(self, prompt: str) -> str:
        if "meningitis" in prompt.lower() and "stiff neck" in prompt.lower():
            return "Based on symptoms, consider Meningitis, Dengue Fever, Japanese Encephalitis. Meningitis is highly probable given stiff neck, fever, photophobia. Dengue and Japanese Encephalitis are relevant due to travel history. Need to confirm specific lab markers for differentiation. Focus on ruling out bacterial meningitis first."  # noqa: E501
        elif "headache" in prompt.lower():
            return "Based on the symptoms, potential diagnoses include tension headache, migraine, or a viral infection. Further details needed for refinement."  # noqa: E501
        return "I am a mock LLM. Based on the input, I can suggest some generic possibilities. Further details would improve my diagnostic capabilities."


# 3. Knowledge Graph (KG) Abstraction Layer
class AbstractKGAdapter(ABC):
    @abstractmethod
    def query(self, query_string: str) -> str:
        pass


class MockICD10KG(AbstractKGAdapter):
    def query(self, query_string: str) -> str:
        if "meningitis" in query_string.lower():
            return "ICD-10 Code: G03.9 - Meningitis, unspecified. Key diagnostic criteria include CSF analysis, neurological examination, and imaging. Bacterial meningitis is a medical emergency."  # noqa: E501
        elif "dengue fever" in query_string.lower():
            return "ICD-10 Code: A90 - Dengue fever [classical dengue]. Characterized by fever, rash, muscle and joint pain. Endemic in tropical and subtropical regions."  # noqa: E501
        return f"No specific ICD-10 information found for '{query_string}'."


class MockRxNormKG(AbstractKGAdapter):
    def query(self, query_string: str) -> str:
        if "acetaminophen" in query_string.lower() and "drug interactions" in query_string.lower():  # noqa: E501
            return "RxNorm: Acetaminophen can interact with alcohol (increased liver risk) and warfarin (increased bleeding risk)."
        return f"No specific RxNorm drug interaction information found for '{query_string}'."


class MockTravelKG(AbstractKGAdapter):
    def query(self, query_string: str) -> str:
        if "southeast asia" in query_string.lower() and "symptoms" in query_string.lower():  # noqa: E501
            return "Travel KG: Common diseases in Southeast Asia with fever, headache, stiff neck include Dengue Fever, Japanese Encephalitis, and Typhoid Fever. Consider malaria screening too."
        return f"No specific travel health information found for '{query_string}'."


# 4. Core Orchestration Framework
class DiagnosticAssistant:
    def __init__(self, llm_adapter: AbstractLLMAdapter, kg_adapters: dict[str, AbstractKGAdapter]):  # noqa: E501
        self.llm = llm_adapter
        self.kges = kg_adapters

    def assist_diagnosis(self, patient_symptoms: PatientSymptoms) -> DiagnosisResult:
        # Step 1: LLM generates initial differential diagnoses and potential KG queries
        llm_prompt = f"Patient symptoms: {patient_symptoms.symptoms}. Medical history: {patient_symptoms.medical_history}. Medications: {patient_symptoms.medications}. Travel history: {patient_symptoms.travel_history}. Based on this, suggest potential differential diagnoses and key areas for knowledge graph queries (e.g., specific diseases, drug interactions, regional prevalence)."
        llm_response = self.llm.get_completion(llm_prompt)

        # Parse LLM response (simplified for mock)
        differential_diagnoses = []
        reasoning_start_idx = llm_response.find("Based on symptoms,")
        if reasoning_start_idx != -1:
            reasoning_text = llm_response[reasoning_start_idx:].strip()
            if "meningitis" in reasoning_text.lower():
                differential_diagnoses.append("Meningitis")
            if "dengue fever" in reasoning_text.lower():
                differential_diagnoses.append("Dengue Fever")
            if "japanese encephalitis" in reasoning_text.lower():
                differential_diagnoses.append("Japanese Encephalitis")
            if "tension headache" in reasoning_text.lower():
                differential_diagnoses.append("Tension Headache")
            if "migraine" in reasoning_text.lower():
                differential_diagnoses.append("Migraine")
            if "viral infection" in reasoning_text.lower():
                differential_diagnoses.append("Viral Infection")
            llm_reasoning = reasoning_text
        else:
            llm_reasoning = llm_response

        # Step 2: Query KGs based on LLM's initial thoughts and patient info
        kg_insights = []
        for diag in differential_diagnoses:
            if "icd10" in self.kges:
                icd10_result = self.kges["icd10"].query(diag)
                kg_insights.append(KGInsight(source="ICD-10", query=diag, result=icd10_result))
        
        if patient_symptoms.medications and "rxnorm" in self.kges:
            rxnorm_result = self.kges["rxnorm"].query(f"{patient_symptoms.medications} drug interactions")
            kg_insights.append(KGInsight(source="RxNorm", query=f"{patient_symptoms.medications} drug interactions", result=rxnorm_result))
        
        if patient_symptoms.travel_history and "travelkg" in self.kges:
            travel_result = self.kges["travelkg"].query(f"{patient_symptoms.travel_history} symptoms")
            kg_insights.append(KGInsight(source="TravelKG", query=f"{patient_symptoms.travel_history} symptoms", result=travel_result))

        # Step 3: Synthesize results (simplified - in a real app, LLM might do final synthesis)
        final_reasoning = f"{llm_reasoning}\n\nKnowledge Graph Insights:\n"
        for insight in kg_insights:
            final_reasoning += f"- {insight.source} ({insight.query}): {insight.result}\n"
        
        return DiagnosisResult(
            differential_diagnoses=differential_diagnoses,
            reasoning=final_reasoning,
            kg_insights=kg_insights,
            confidence="High" if len(differential_diagnoses) > 1 and len(kg_insights) > 0 else "Medium"
        )


# 5. Streamlit UI Simulation
def run_streamlit_app_simulation():
    print("--- Intelligent Medical Diagnostic Assistant (Simulation) ---")
    print("\nInput Patient Symptoms:")
    symptoms = input("Symptoms (e.g., 'severe headache, stiff neck, fever, photophobia'): ")
    medical_history = input("Medical History (optional): ")
    medications = input("Current Medications (optional, e.g., 'acetaminophen'): ")
    travel_history = input("Recent Travel History (optional, e.g., 'Southeast Asia'): ")

    patient_data = PatientSymptoms(
        symptoms=symptoms,
        medical_history=medical_history,
        medications=medications,
        travel_history=travel_history
    )

    # Initialize LLM and KG Adapters
    llm_adapter = MockLLMAdapter()
    kg_adapters = {
        "icd10": MockICD10KG(),
        "rxnorm": MockRxNormKG(),
        "travelkg": MockTravelKG()
    }

    assistant = DiagnosticAssistant(llm_adapter=llm_adapter, kg_adapters=kg_adapters)

    print("\nProcessing patient data...")
    result = assistant.assist_diagnosis(patient_data)

    print("\n--- Diagnostic Assistance Result ---")
    print(f"Differential Diagnoses: {', '.join(result.differential_diagnoses)}")
    print(f"Confidence: {result.confidence}")
    print("\nDetailed Reasoning and KG Insights:")
    print(result.reasoning)


if __name__ == "__main__":
    run_streamlit_app_simulation()
