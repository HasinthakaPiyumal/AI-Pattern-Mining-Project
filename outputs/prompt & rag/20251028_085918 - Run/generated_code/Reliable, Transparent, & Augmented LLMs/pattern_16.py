import datetime
import json
from typing import List, Dict, Union, Optional

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from loguru import logger

# --- 1. FastAPI Application Setup ---
app = FastAPI(title="Medical Diagnostic and Patient Support AI Assistant")

# --- 7. Data Storage (Simplified) ---
# In-memory storage for demonstration
PATIENT_RECORDS = {} # patient_id: {"name": ..., "history": [...], "age": ..., "gender": ...}
AI_DIAGNOSES = {}    # diagnosis_id: {"patient_id": ..., "diagnosis": ..., "reasoning": ..., "confidence": ..., "timestamp": ...}
HUMAN_FEEDBACK = {}  # feedback_id: {"diagnosis_id": ..., "feedback": ..., "corrections": ..., "timestamp": ...}
FEEDBACK_ID_COUNTER = 0

# --- Pydantic Models ---
class SymptomInput(BaseModel):
    patient_id: str = Field(..., example="patient_123")
    symptoms: List[str] = Field(..., example=["fever", "cough", "fatigue"])
    age: Optional[int] = Field(None, ge=0, example=35)
    gender: Optional[str] = Field(None, example="male")
    medical_history: Optional[List[str]] = Field(None, example=["asthma", "seasonal allergies"])

class AIConsultationResponse(BaseModel):
    diagnosis_id: str
    patient_id: str
    diagnosis: str
    reasoning: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    abstain: bool = False
    recommendations: List[str]
    timestamp: datetime.datetime

class HumanFeedback(BaseModel):
    diagnosis_id: str = Field(..., example="diag_1")
    feedback: str = Field(..., example="AI diagnosis was accurate but missed a rare differential.")
    corrections: Optional[Dict[str, str]] = Field(None, example={
        "diagnosis": "Flu with complications",
        "reasoning": "Patient also presented with dehydration."
    })
    verified: bool = Field(False, description="Whether the human professional verified the AI's diagnosis as correct.")

# --- 3. External Tools (Placeholder/Simulated) ---
# These functions simulate external API calls or database lookups
def medical_knowledge_base_tool(query: str) -> str:
    """Simulates querying a medical knowledge base."""
    logger.info(f"Tool Call: Medical Knowledge Base with query: {query}")
    query_lower = query.lower()
    if "fever" in query_lower and "cough" in query_lower:
        return "Common cold symptoms include fever, cough, fatigue, and runny nose. Flu can present similarly but often with more severe body aches and higher fever. Pneumonia often involves a productive cough and shortness of breath."
    elif "diabetes" in query_lower:
        return "Diabetes Mellitus is a metabolic disease that causes high blood sugar. Type 1 is autoimmune, Type 2 is insulin resistance. Symptoms include frequent urination, increased thirst, and unexplained weight loss."
    return f"Information about '{query}' from medical knowledge base (simulated)."

def symptom_checker_tool(symptoms: List[str], age: Optional[int] = None, gender: Optional[str] = None) -> Dict[str, Union[str, float, List[str]]]:
    """Simulates a symptom checker, returning probabilistic diagnoses."""
    logger.info(f"Tool Call: Symptom Checker with symptoms: {symptoms}, age: {age}, gender: {gender}")
    symptoms_str = " ".join(symptoms).lower()
    if "fever" in symptoms_str and "cough" in symptoms_str and "fatigue" in symptoms_str:
        return {"diagnosis": "Common Cold", "probability": 0.6, "differential": ["Flu", "Bronchitis"]}
    elif "chest pain" in symptoms_str and "shortness of breath" in symptoms_str:
        return {"diagnosis": "Potential Cardiac Event", "probability": 0.8, "differential": ["Anxiety attack", "Pneumonia"], "urgent_care": True}
    elif "headache" in symptoms_str and "stiff neck" in symptoms_str:
        return {"diagnosis": "Possible Meningitis", "probability": 0.75, "differential": ["Severe tension headache", "Migraine"], "urgent_care": True}
    return {"diagnosis": "Undetermined", "probability": 0.4, "differential": [], "urgent_care": False}

def lab_result_analyzer_tool(lab_results: Dict[str, Union[float, str]]) -> str:
    """Simulates interpreting lab results and flagging abnormalities."""
    logger.info(f"Tool Call: Lab Result Analyzer with results: {lab_results}")
    feedback = []
    if "glucose" in lab_results and isinstance(lab_results["glucose"], (int, float)):
        if lab_results["glucose"] > 125:
            feedback.append(f"High glucose level ({lab_results['glucose']} mg/dL), indicative of hyperglycemia.")
        elif lab_results["glucose"] < 70:
            feedback.append(f"Low glucose level ({lab_results['glucose']} mg/dL), indicative of hypoglycemia.")
    if "white_blood_cells" in lab_results and isinstance(lab_results["white_blood_cells"], (int, float)):
        if lab_results["white_blood_cells"] > 11.0:
            feedback.append(f"Elevated white blood cell count ({lab_results['white_blood_cells']} x10^9/L), suggesting infection or inflammation.")
    return ". ".join(feedback) if feedback else "Lab results appear within normal ranges or no specific flags were identified (simulated)."

def drug_interaction_database_tool(drugs: List[str]) -> str:
    """Simulates checking for adverse drug interactions."""
    logger.info(f"Tool Call: Drug Interaction Database with drugs: {drugs}")
    drugs_lower = [d.lower() for d in drugs]
    if "warfarin" in drugs_lower and "ibuprofen" in drugs_lower:
        return "Significant interaction: Ibuprofen can increase the risk of bleeding when taken with Warfarin."
    if "simvastatin" in drugs_lower and "grapefruit juice" in drugs_lower:
        return "Moderate interaction: Grapefruit juice can increase Simvastatin levels, leading to muscle problems."
    return "No significant drug interactions found for the given medications (simulated)."

# --- LangChain Components (Simplified/Mocked for demonstration) ---
# In a real scenario, you'd initialize a proper LLM from langchain.chat_models
# and use AgentExecutor for dynamic tool calling. For this self-contained example,
# we use a MockLLM and manually orchestrate tool calls.
class MockLLM:
    """A mock LLM to simulate responses without actual API calls."""
    def invoke(self, prompt: str) -> str:
        logger.info(f"MockLLM: Invoking with prompt snippet: {prompt[:200]}...")
        # Simple rule-based mock for demonstration, returning JSON string
        if "chest pain" in prompt.lower() and "shortness of breath" in prompt.lower():
            return json.dumps({
                "diagnosis": "Requires Immediate Medical Attention",
                "reasoning": "The patient reports severe symptoms (chest pain, shortness of breath) which could indicate a serious cardiac or respiratory event. The symptom checker tool flagged a potential cardiac event with high probability. Given the critical nature of these symptoms, immediate consultation with a human medical professional is strongly recommended.",
                "confidence": 0.95,
                "abstain": False,
                "recommendations": [
                    "Seek emergency medical care immediately.",
                    "Do not self-medicate.",
                    "Inform medical professionals about all symptoms and medical history."
                ]
            })
        elif "fever" in prompt.lower() and "cough" in prompt.lower() and "fatigue" in prompt.lower():
            return json.dumps({
                "diagnosis": "Likely Common Cold",
                "reasoning": "Based on reported symptoms (fever, cough, fatigue) which are highly consistent with the common cold. The symptom checker tool also suggested this with a moderate probability.",
                "confidence": 0.7,
                "abstain": False,
                "recommendations": [
                    "Rest and stay hydrated.",
                    "Over-the-counter cold medication if symptoms are bothersome (e.g., acetaminophen for fever, cough syrup).",
                    "Consult a doctor if symptoms worsen or persist for more than a week."
                ]
            })
        elif "undetermined" in prompt.lower() or "insufficient data" in prompt.lower() or "vague" in prompt.lower():
            return json.dumps({
                "diagnosis": "Undetermined - Insufficient Data/High Uncertainty",
                "reasoning": "The provided information is insufficient to confidently form a diagnosis, or symptoms are too vague/contradictory. The symptom checker tool returned 'Undetermined'. Abstaining from a definitive diagnosis at this time.",
                "confidence": 0.2,
                "abstain": True,
                "recommendations": [
                    "Provide more detailed symptoms or additional medical history.",
                    "Consult a human doctor for a thorough examination and further diagnostic tests."
                ]
            })
        return json.dumps({
            "diagnosis": "General Health Advice",
            "reasoning": "Based on general understanding of symptoms provided.",
            "confidence": 0.5,
            "abstain": False,
            "recommendations": [
                "Monitor your symptoms closely.",
                "If symptoms worsen or new symptoms appear, seek medical advice.",
                "Maintain a healthy lifestyle with adequate rest and nutrition."
            ]
        })

mock_llm = MockLLM()

# This function simulates the LangChain agent's orchestration logic
def medical_agent_orchestrator(patient_input: SymptomInput) -> Dict:
    """
    Simulates the LangChain agent orchestrating LLM and tool calls.
    In a full LangChain setup, an AgentExecutor would handle tool selection dynamically.
    Here, we manually decide which tools to call first for demonstration.
    """
    logger.info(f"Agent Orchestrator: Processing input for patient {patient_input.patient_id}")

    # --- 6. Robust Processing & Input Validation (Pydantic handles initial validation) ---
    # Additional simple adversarial check (placeholder)
    if any(keyword in ' '.join(patient_input.symptoms).lower() for keyword in ["attack system", "bypass security"]):
        logger.warning(f"Potential adversarial input detected for patient {patient_input.patient_id}")
        return {
            "diagnosis": "Security Alert",
            "reasoning": "Input detected as potentially malicious or an attempt to bypass system safeguards. Cannot proceed with diagnosis.",
            "confidence": 0.0,
            "abstain": True,
            "recommendations": ["Input was flagged for security concerns. Please provide valid medical symptoms."]
        }

    # Step 1: Call Symptom Checker tool initially, as it's often the first step in diagnosis
    symptom_check_result = symptom_checker_tool(
        symptoms=patient_input.symptoms,
        age=patient_input.age,
        gender=patient_input.gender
    )
    logger.debug(f"Symptom Checker Result: {symptom_check_result}")

    # Example of calling another tool if needed (e.g., if medical history mentioned specific drugs)
    # For simplicity, this example primarily relies on symptom checker and then LLM.
    # if patient_input.medical_history and any("medication" in h.lower() for h in patient_input.medical_history):
    #     drug_interaction_info = drug_interaction_database_tool(["example_drug1", "example_drug2"])
    #     logger.debug(f"Drug Interaction Info: {drug_interaction_info}")

    # --- 4. Reasoning & Transparency Module ---
    # Step 2: Formulate prompt for LLM, including tool outputs and desired output format
    prompt = f"""
    You are a highly reliable, robust, and transparent AI medical assistant. Your goal is to provide accurate and understandable medical insights, including reasoning, confidence, and to abstain when highly uncertain. You are interacting with a patient.

    The patient (ID: {patient_input.patient_id}) presents with the following symptoms: {', '.join(patient_input.symptoms)}.
    Age: {patient_input.age if patient_input.age else 'Not provided'}.
    Gender: {patient_input.gender if patient_input.gender else 'Not provided'}.
    Known Medical History: {', '.join(patient_input.medical_history) if patient_input.medical_history else 'None'}.

    Here is the primary output from a symptom checker tool:
    Diagnosis Suggestion: {symptom_check_result.get('diagnosis', 'N/A')}
    Probability (from tool): {symptom_check_result.get('probability', 'N/A')}
    Differential Diagnoses: {', '.join(symptom_check_result.get('differential', []))}
    Urgent Care Indication: {symptom_check_result.get('urgent_care', 'N/A')}

    Based on all available information and your medical knowledge, perform the following steps:
    1. Provide a concise, most probable diagnosis.
    2. Explain your detailed reasoning, integrating information from the symptoms and the symptom checker tool.
    3. Estimate your confidence in this diagnosis as a floating-point number between 0.0 (very uncertain) and 1.0 (very certain).
    4. If your confidence is below 0.3, or if the case is complex/critical and requires human medical expertise, set 'abstain' to true. Otherwise, set it to false.
    5. Offer actionable recommendations, which may include next steps like consulting a doctor, monitoring symptoms, or lifestyle advice.

    Your response MUST be a valid JSON object with the following keys:
    {{
        "diagnosis": "<string>",
        "reasoning": "<string>",
        "confidence": <float_0.0_to_1.0>,
        "abstain": <boolean>,
        "recommendations": ["<string_recommendation_1>", "<string_recommendation_2>", ...]
    }}
    Ensure the JSON is well-formed and contains only the specified keys.
    """
    logger.debug(f"LLM Prompt (partial): {prompt[:500]}...")

    # Step 3: Call LLM
    llm_raw_response = mock_llm.invoke(prompt)
    logger.debug(f"LLM Raw Response: {llm_raw_response}")

    # --- 4. Output Parsing ---
    try:
        parsed_llm_output = json.loads(llm_raw_response)
        # Basic validation for required keys and types from LLM output
        if not all(k in parsed_llm_output for k in ["diagnosis", "reasoning", "confidence", "abstain", "recommendations"]):
             raise ValueError("LLM response missing required keys.")
        if not isinstance(parsed_llm_output["confidence"], (int, float)) or not (0.0 <= parsed_llm_output["confidence"] <= 1.0):
             raise ValueError("Confidence must be a float between 0.0 and 1.0.")
        if not isinstance(parsed_llm_output["abstain"], bool):
             raise ValueError("Abstain must be a boolean.")
        if not isinstance(parsed_llm_output["recommendations"], list):
             raise ValueError("Recommendations must be a list.")
        if not isinstance(parsed_llm_output["diagnosis"], str) or not isinstance(parsed_llm_output["reasoning"], str):
             raise ValueError("Diagnosis and reasoning must be strings.")

        return parsed_llm_output
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse LLM output: {e}. Raw output: {llm_raw_response}")
        # Fallback for malformed LLM response, with abstention
        return {
            "diagnosis": "Error in AI processing - Malformed Response",
            "reasoning": f"Could not parse AI's response due to format issues: {e}. Original response: {llm_raw_response}",
            "confidence": 0.0,
            "abstain": True,
            "recommendations": ["An internal error occurred while processing. Please try again or consult a human medical professional."]
        }

# --- 1. API Layer (FastAPI Endpoints) ---
@app.post("/diagnose", response_model=AIConsultationResponse, summary="Get AI-powered medical diagnosis and support")
async def get_diagnosis(symptom_input: SymptomInput):
    """
    Submits patient symptoms and other relevant information to the AI assistant
    to receive an AI-generated diagnosis, detailed reasoning, confidence score,
    and actionable recommendations. The AI will abstain if it's highly uncertain.
    """
    # (Simplified) Add/update patient to records
    if symptom_input.patient_id not in PATIENT_RECORDS:
        PATIENT_RECORDS[symptom_input.patient_id] = {
            "name": f"Patient_{symptom_input.patient_id}", # Placeholder name
            "history": [],
            "age": symptom_input.age,
            "gender": symptom_input.gender
        }
    # Update age/gender if provided in current request (simple update)
    if symptom_input.age is not None: PATIENT_RECORDS[symptom_input.patient_id]["age"] = symptom_input.age
    if symptom_input.gender is not None: PATIENT_RECORDS[symptom_input.patient_id]["gender"] = symptom_input.gender

    # Simulate LangChain agent orchestration
    ai_output = medical_agent_orchestrator(symptom_input)

    diagnosis_id = f"diag_{len(AI_DIAGNOSES) + 1}"
    current_time = datetime.datetime.now()

    diagnosis_record = {
        "diagnosis_id": diagnosis_id,
        "patient_id": symptom_input.patient_id,
        "diagnosis": ai_output["diagnosis"],
        "reasoning": ai_output["reasoning"],
        "confidence": ai_output["confidence"],
        "abstain": ai_output["abstain"],
        "recommendations": ai_output["recommendations"],
        "timestamp": current_time
    }
    AI_DIAGNOSES[diagnosis_id] = diagnosis_record
    PATIENT_RECORDS[symptom_input.patient_id]["history"].append(diagnosis_id)

    logger.info(f"Generated diagnosis {diagnosis_id} for patient {symptom_input.patient_id} with confidence {ai_output['confidence']}. Abstain: {ai_output['abstain']}")
    return AIConsultationResponse(**diagnosis_record)


@app.post("/feedback", summary="Submit human medical professional feedback on an AI diagnosis")
async def submit_feedback(feedback_input: HumanFeedback):
    """
    Allows healthcare professionals to provide feedback, corrections, and verification
    for AI-generated diagnoses, which is crucial for the Human-in-the-Loop mechanism
    and system improvement. This feedback can be used for retraining or knowledge base updates.
    """
    global FEEDBACK_ID_COUNTER

    if feedback_input.diagnosis_id not in AI_DIAGNOSES:
        raise HTTPException(status_code=404, detail=f"Diagnosis ID '{feedback_input.diagnosis_id}' not found.")

    FEEDBACK_ID_COUNTER += 1
    feedback_id = f"feedback_{FEEDBACK_ID_COUNTER}"
    current_time = datetime.datetime.now()

    feedback_record = {
        "feedback_id": feedback_id,
        "diagnosis_id": feedback_input.diagnosis_id,
        "feedback": feedback_input.feedback,
        "corrections": feedback_input.corrections,
        "verified": feedback_input.verified,
        "timestamp": current_time
    }
    HUMAN_FEEDBACK[feedback_id] = feedback_record

    logger.info(f"Received feedback {feedback_id} for diagnosis {feedback_input.diagnosis_id}. Verified: {feedback_input.verified}")
    return {"message": "Feedback submitted successfully.", "feedback_id": feedback_id}

@app.get("/diagnosis/{diagnosis_id}", response_model=AIConsultationResponse, summary="Retrieve a specific AI diagnosis")
async def get_single_diagnosis(diagnosis_id: str):
    """Retrieves the detailed record of a specific AI-generated diagnosis by its ID."""
    if diagnosis_id not in AI_DIAGNOSES:
        raise HTTPException(status_code=404, detail=f"Diagnosis ID '{diagnosis_id}' not found.")
    return AI_DIAGNOSES[diagnosis_id]

@app.get("/patient_history/{patient_id}", summary="Retrieve a patient's diagnosis history")
async def get_patient_history(patient_id: str) -> Dict[str, Union[str, List[AIConsultationResponse]]]:
    """Retrieves all AI diagnoses associated with a specific patient ID, ordered by timestamp."""
    if patient_id not in PATIENT_RECORDS:
        raise HTTPException(status_code=404, detail=f"Patient ID '{patient_id}' not found.")
    
    history_diagnosis_ids = PATIENT_RECORDS[patient_id]["history"]
    # Retrieve full diagnosis details and sort by timestamp
    history_details = sorted(
        [AIConsultationResponse(**AI_DIAGNOSES[diag_id]) for diag_id in history_diagnosis_ids if diag_id in AI_DIAGNOSES],
        key=lambda x: x.timestamp
    )
    return {"patient_id": patient_id, "diagnoses": history_details}

@app.get("/all_feedback", summary="Retrieve all human feedback records (for administrative purposes)")
async def get_all_feedback() -> Dict[str, HumanFeedback]:
    """Retrieves all human feedback records submitted to the system. (Admin/Debug endpoint)"""
    return HUMAN_FEEDBACK

@app.get("/all_diagnoses", summary="Retrieve all AI diagnosis records (for administrative purposes)")
async def get_all_diagnoses() -> Dict[str, AIConsultationResponse]:
    """Retrieves all AI diagnosis records generated by the system. (Admin/Debug endpoint)"""
    return AI_DIAGNOSES

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application. Run with: uvicorn main:app --reload")
    uvicorn.run(app, host="0.0.0.0", port=8000)
