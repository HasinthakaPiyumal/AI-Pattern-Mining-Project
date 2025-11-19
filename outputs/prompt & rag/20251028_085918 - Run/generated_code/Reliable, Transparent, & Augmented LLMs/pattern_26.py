from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool


class PatientData(BaseModel):
    patient_id: str
    symptoms: List[str]
    medical_history: List[str]
    lab_results: Dict[str, Any]


class DiagnosticOutput(BaseModel):
    initial_diagnosis: str
    confidence_score: float
    reasoning_path: str
    evidence_citations: List[str]
    recommendations: List[str]
    factual_verified: bool
    disclosed_stage: int = 0 # 0: initial, 1: reasoning, 2: recommendations


# --- Simulated External Connectors ---
class KnowledgeBaseConnector:
    def query(self, topic: str) -> str:
        if "pneumonia" in topic.lower():
            return "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. Symptoms include cough with phlegm or pus, fever, chills, and difficulty breathing. Common causes are bacteria, viruses, and fungi. Diagnosis often involves chest X-ray and sputum tests. Treatment depends on the cause but usually involves antibiotics, antivirals, or antifungals."
        elif "diabetes" in topic.lower():
            return "Diabetes is a chronic condition that affects how your body turns food into energy. Most of the food you eat is broken down into sugar (glucose) and released into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin. Insulin acts like a key to let blood sugar into your body's cells for use as energy. With diabetes, your body doesn't make enough insulin or can't use the insulin it makes as well as it should."
        return f"No specific medical knowledge found for '{topic}'."


class EHRConnector:
    def get_patient_data(self, patient_id: str) -> Dict[str, Any]:
        if patient_id == "P001":
            return {
                "id": "P001",
                "name": "Alice Smith",
                "age": 45,
                "gender": "Female",
                "history": ["Hypertension (controlled)", "Seasonal allergies"],
                "medications": ["Lisinopril"],
                "allergies": ["Penicillin"],
                "recent_lab_results": {"WBC": 12.5, "CRP": 15.0, "Blood Sugar": 90}
            }
        return {"error": "Patient not found"}


knowledge_base = KnowledgeBaseConnector()
ehr_system = EHRConnector()


@tool
def get_medical_knowledge(topic: str) -> str:
    return knowledge_base.query(topic)


@tool
def get_ehr_data(patient_id: str) -> str:
    data = ehr_system.get_patient_data(patient_id)
    if "error" in data:
        raise ValueError(data["error"])
    return str(data)


# --- Trustworthiness & Transparency Module ---
def generate_reasoning_path(llm_output: str) -> str:
    parts = llm_output.split("\n\n")
    for part in parts:
        if "Reasoning:" in part:
            return part.replace("Reasoning:", "").strip()
    return "Reasoning path could not be extracted."


def assign_confidence_score(llm_output: str) -> float:
    # Mock confidence score extraction
    if "Confidence:" in llm_output:
        try:
            score_str = llm_output.split("Confidence:")[1].split("\n")[0].strip()
            return float(score_str) / 100.0 # Assuming LLM gives 0-100
        except ValueError:
            pass
    return 0.75  # Default confidence


def get_evidence_citations(llm_output: str) -> List[str]:
    citations = []
    if "Citations:" in llm_output:
        citation_block = llm_output.split("Citations:")[1].split("\n\n")[0].strip()
        citations = [c.strip() for c in citation_block.split("\n") if c.strip()]
    return citations


def progressive_response_disclosure(full_output: DiagnosticOutput, stage: int) -> DiagnosticOutput:
    if stage == 0:
        return DiagnosticOutput(
            initial_diagnosis=full_output.initial_diagnosis,
            confidence_score=full_output.confidence_score,
            reasoning_path="(Hidden until stage 1)",
            evidence_citations=[],
            recommendations=[],
            factual_verified=False,
            disclosed_stage=0
        )
    elif stage == 1:
        return DiagnosticOutput(
            initial_diagnosis=full_output.initial_diagnosis,
            confidence_score=full_output.confidence_score,
            reasoning_path=full_output.reasoning_path,
            evidence_citations=full_output.evidence_citations,
            recommendations=["(Hidden until stage 2)"],
            factual_verified=False,
            disclosed_stage=1
        )
    else: # stage >= 2
        return full_output.model_copy(update={"disclosed_stage": stage})


# --- Quality Control & Evaluation Module ---
def factual_verification(diagnosis: str, evidence: List[str]) -> bool:
    if "pneumonia" in diagnosis.lower() and "infection" in evidence[0].lower():
        return True
    if "diabetes" in diagnosis.lower() and "insulin" in evidence[0].lower():
        return True
    return False


def prompt_sanitizer(prompt: str) -> str:
    # Simple sanitization to prevent basic injection attempts
    sanitized_prompt = prompt.replace("\"", "")
    sanitized_prompt = sanitized_prompt.replace("\n", " ")
    return sanitized_prompt


# --- LLM Orchestration (LangChain) ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

langchain_tools = [get_medical_knowledge, get_ehr_data]

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a medical diagnostic assistant. Your goal is to provide differential diagnoses, reasoning, confidence scores (0-100), evidence citations, and recommendations based on patient data and medical knowledge. Always state confidence in percentage. Format your output clearly: Initial Diagnosis: [Diagnosis]\nConfidence: [Score%]\nReasoning: [Detailed reasoning]\nCitations: [List of citations]\nRecommendations: [List of recommendations]"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, langchain_tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=langchain_tools, verbose=False)


# --- FastAPI Application ---
app = FastAPI()


@app.post("/diagnose", response_model=DiagnosticOutput)
async def diagnose_patient(patient_data: PatientData):
    sanitized_symptoms = [prompt_sanitizer(s) for s in patient_data.symptoms]
    sanitized_medical_history = [prompt_sanitizer(h) for h in patient_data.medical_history]

    input_prompt = (
        f"Patient ID: {patient_data.patient_id}\n"
        f"Symptoms: {', '.join(sanitized_symptoms)}\n"
        f"Medical History: {', '.join(sanitized_medical_history)}\n"
        f"Lab Results: {patient_data.lab_results}\n"
        "Provide a differential diagnosis, reasoning, confidence (0-100%), evidence citations, and recommendations."
    )

    try:
        # Simulate LangChain agent interaction and tool usage
        # In a real scenario, agent_executor.invoke would call tools.
        # For this mock, we'll manually simulate the data flow.
        ehr_info = ehr_system.get_patient_data(patient_data.patient_id)
        if "error" in ehr_info:
            raise HTTPException(status_code=404, detail=ehr_info["error"])
        
        # Mock LLM response for a specific case
        mock_llm_response = (
            "Initial Diagnosis: Possible Bacterial Pneumonia\n"
            "Confidence: 85%\n"
            "Reasoning: The patient presents with symptoms common to pneumonia, such as cough and difficulty breathing. Elevated WBC and CRP levels in lab results suggest an active infection. Medical history and current medications are noted. Consultation of medical knowledge indicates bacterial pneumonia as a strong possibility given these factors. EHR data shows no immediate contraindications.\n"
            "Citations:\n"
            "1. PubMed Article: Clinical presentation and diagnosis of community-acquired pneumonia.\n"
            "2. CDC Guidelines: Management of acute respiratory infections.\n"
            "Recommendations:\n"
            "1. Order a chest X-ray to confirm lung consolidation.\n"
            "2. Perform sputum culture to identify causative organism.\n"
            "3. Start empiric broad-spectrum antibiotics (e.g., Azithromycin + Amoxicillin-Clavulanate) pending culture results.\n"
            "4. Monitor oxygen saturation and provide supportive care."
        )

        # If input suggests diabetes, provide a diabetes response
        if any("high blood sugar" in s.lower() for s in sanitized_symptoms) or patient_data.lab_results.get("Blood Sugar", 0) > 125:
            mock_llm_response = (
                "Initial Diagnosis: Type 2 Diabetes Mellitus\n"
                "Confidence: 92%\n"
                "Reasoning: Patient's symptoms (if any were present indicating high blood sugar) combined with a high blood sugar lab result strongly suggest Type 2 Diabetes. Medical knowledge confirms these indicators. Further evaluation is needed to differentiate from other types.\n"
                "Citations:\n"
                "1. ADA Standards of Medical Care in Diabetes.\n"
                "2. WHO Report: Global Diabetes Factsheet.\n"
                "Recommendations:\n"
                "1. Order HbA1c, fasting blood glucose, and oral glucose tolerance test.\n"
                "2. Advise lifestyle modifications (diet, exercise).\n"
                "3. Consider Metformin if confirmed, based on clinical guidelines.\n"
                "4. Refer to endocrinologist for specialized management."
            )

        # Simulate LLM processing and extraction
        initial_diagnosis_text = mock_llm_response.split("Initial Diagnosis:")[1].split("\n")[0].strip()
        confidence = assign_confidence_score(mock_llm_response)
        reasoning = generate_reasoning_path(mock_llm_response)
        citations = get_evidence_citations(mock_llm_response)

        recommendations_block = mock_llm_response.split("Recommendations:")[1].strip()
        recommendations_list = [r.strip() for r in recommendations_block.split("\n") if r.strip()]

        # Factual verification
        is_verified = factual_verification(initial_diagnosis_text, [knowledge_base.query(initial_diagnosis_text)])

        full_diagnostic_output = DiagnosticOutput(
            initial_diagnosis=initial_diagnosis_text,
            confidence_score=confidence,
            reasoning_path=reasoning,
            evidence_citations=citations,
            recommendations=recommendations_list,
            factual_verified=is_verified
        )

        # Progressively disclose response (e.g., return initial, then on subsequent calls return more)
        # For this single API call, we'll return the full, but the structure supports stages.
        return progressive_response_disclosure(full_diagnostic_output, 2) # Returning full disclosure for demonstration

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")