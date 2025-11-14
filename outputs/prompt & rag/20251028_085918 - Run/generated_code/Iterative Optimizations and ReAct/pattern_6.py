#!/usr/bin/env python3

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

# --- Mock LLM and External Services (Simulations) ---

class MockLLM:
    """Simulates an LLM for reasoning and decision-making."""
    async def chat(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None, 
                   tool_choice: Optional[str] = None) -> Dict[str, Any]:
        prompt = messages[-1]["content"]
        print(f"[MockLLM] Processing prompt: {prompt[:100]}...")

        # Simple tool calling logic
        if "query_medical_db" in prompt and tools:
            print("[MockLLM] Decided to call query_medical_db.")
            return {"tool_calls": [{"function": {"name": "query_medical_db", "arguments": "{\"query\": \"diabetes symptoms\"}"}}]}
        elif "analyze_lab_results" in prompt and tools:
            print("[MockLLM] Decided to call analyze_lab_results.")
            return {"tool_calls": [{"function": {"name": "analyze_lab_results", "arguments": "{\"results_data\": \"glucose: 150 mg/dL\"}"}}]}
        elif "access_ehr" in prompt and tools:
            print("[MockLLM] Decided to call access_ehr.")
            return {"tool_calls": [{"function": {"name": "access_ehr", "arguments": "{\"patient_id\": \"P123\", \"data_type\": \"medications\"}"}}]}
        elif "retrieve_knowledge" in prompt and tools:
            print("[MockLLM] Decided to call retrieve_knowledge.")
            return {"tool_calls": [{"function": {"name": "retrieve_knowledge", "arguments": "{\"query\": \"insulin resistance guidelines\"}"}}]}
        elif "suggest diagnosis" in prompt:
            print("[MockLLM] Generating diagnosis.")
            return {"content": "Based on the available data, a preliminary diagnosis of Type 2 Diabetes is suggested. Further tests recommended: HbA1c, oral glucose tolerance test. Consider Metformin as initial treatment."}
        elif "reflect and correct" in prompt:
            print("[MockLLM] Reflecting and self-correcting.")
            return {"content": "Reflection complete. Identified a potential oversight in considering patient's diet history. Will re-evaluate after accessing EHR diet logs."}
        elif "evaluate termination" in prompt:
            print("[MockLLM] Evaluating termination conditions.")
            if "confident in diagnosis" in prompt:
                return {"content": "Confidence in diagnosis is high. All critical information gathered. Ready to terminate."}
            else:
                return {"content": "Not yet confident. Missing diet history from EHR. Continue."}
        
        # Default response
        return {"content": f"Understood: {prompt}. I need more information or a specific tool to proceed."}


class MockMedicalDB:
    """Simulates a medical database (e.g., RxNorm, PubMed)."""
    async def query(self, query: str) -> Dict[str, Any]:
        print(f"[MockMedicalDB] Querying for: {query}")
        if "diabetes symptoms" in query.lower():
            return {"source": "MedicalDB", "result": "Common symptoms of diabetes include frequent urination, increased thirst, unexplained weight loss, fatigue, blurred vision, slow-healing sores."}
        if "metformin side effects" in query.lower():
            return {"source": "MedicalDB", "result": "Metformin side effects include nausea, diarrhea, stomach upset. Rare but serious: lactic acidosis."}
        return {"source": "MedicalDB", "result": f"No specific data found for '{query}'."}


class MockLabAnalyzer:
    """Simulates a lab result parsing and analysis service."""
    async def analyze(self, results_data: str) -> Dict[str, Any]:
        print(f"[MockLabAnalyzer] Analyzing: {results_data}")
        parsed_results = {"source": "LabAnalyzer", "results": {}}
        if "glucose: 150 mg/dL" in results_data:
            parsed_results["results"]["glucose"] = {"value": 150, "unit": "mg/dL", "status": "high"}
            parsed_results["analysis"] = "Elevated glucose level, indicative of hyperglycemia."
        return parsed_results


class MockImagingAnalysisService:
    """Simulates an external imaging analysis microservice."""
    async def analyze_image(self, image_id: str) -> Dict[str, Any]:
        print(f"[MockImagingAnalysisService] Analyzing image: {image_id}")
        # In a real scenario, this would send an image to an ML model
        if "chest_xray_p123" in image_id:
            return {"source": "ImagingService", "image_id": image_id, "findings": "No acute cardiopulmonary abnormalities. Small nodule in right upper lobe (requires follow-up)."}
        return {"source": "ImagingService", "image_id": image_id, "findings": "Analysis not available for this image."}


class MockEHRSystem:
    """Simulates an Electronic Health Record (EHR) system."""
    async def access_patient_data(self, patient_id: str, data_type: str) -> Dict[str, Any]:
        print(f"[MockEHRSystem] Accessing EHR for patient {patient_id}, data_type: {data_type}")
        if patient_id == "P123":
            if data_type == "medications":
                return {"source": "EHR", "patient_id": patient_id, "data": {"current_meds": ["Lisinopril", "Simvastatin"], "allergies": ["Penicillin"]}}
            if data_type == "history":
                return {"source": "EHR", "patient_id": patient_id, "data": {"past_conditions": ["Hypertension", "Hyperlipidemia"], "family_history": "Diabetes"}}
            if data_type == "diet_logs":
                return {"source": "EHR", "patient_id": patient_id, "data": {"diet": "High carbohydrate diet reported over the last 6 months."}}
        return {"source": "EHR", "patient_id": patient_id, "data": f"No '{data_type}' data for patient {patient_id}."}


class MockVectorDB:
    """Simulates a vector database for knowledge retrieval."""
    def __init__(self):
        self.knowledge_base = {
            "diabetes_guidelines": "Clinical guidelines for managing Type 2 Diabetes recommend lifestyle changes, Metformin as first-line, and consideration of SGLT2 inhibitors or GLP-1 receptor agonists if targets not met.",
            "insulin_resistance": "Insulin resistance is a condition in which the body's cells don't respond well to insulin and can't easily take up glucose from your blood.",
            "patient_case_1": "Anonymized case: 55-year-old male, elevated glucose, family history of diabetes, responded well to Metformin and diet changes."
        }

    async def search(self, query: str) -> List[Dict[str, str]]:
        print(f"[MockVectorDB] Searching knowledge base for: {query}")
        results = []
        for key, value in self.knowledge_base.items():
            if query.lower() in key.lower() or query.lower() in value.lower():
                results.append({"id": key, "content": value})
        return results


# --- Tool Definitions ---

def tool_query_medical_db(query: str) -> Dict[str, Any]:
    """Queries an external medical database for information."""
    return {"function": "query_medical_db", "args": {"query": query}}

def tool_analyze_lab_results(results_data: str) -> Dict[str, Any]:
    """Analyzes raw lab result data."""
    return {"function": "analyze_lab_results", "args": {"results_data": results_data}}

def tool_imaging_analysis(image_id: str) -> Dict[str, Any]:
    """Submits an image for specialized analysis."""
    return {"function": "imaging_analysis", "args": {"image_id": image_id}}

def tool_access_ehr(patient_id: str, data_type: str) -> Dict[str, Any]:
    """Accesses patient data from the EHR system."""
    return {"function": "access_ehr", "args": {"patient_id": patient_id, "data_type": data_type}}

def tool_retrieve_knowledge(query: str) -> Dict[str, Any]:
    """Retrieves relevant information from the internal knowledge base."""
    return {"function": "retrieve_knowledge", "args": {"query": query}}


# --- Agent State Management ---

class AgentState(Dict[str, Any]):
    """Represents the current state of the diagnostic agent."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setdefault("patient_id", None)
        self.setdefault("chat_history", [])
        self.setdefault("observations", [])
        self.setdefault("hypotheses", [])
        self.setdefault("feedback_log", [])
        self.setdefault("diagnosis_confidence", 0.0)
        self.setdefault("termination_reason", None)
        self.setdefault("current_plan", [])


# --- Core Agent Logic ---

class MedicalDiagnosticAgent:
    """Intelligent Medical Diagnostic Assistant agent."""
    def __init__(self):
        self.llm = MockLLM()
        self.medical_db = MockMedicalDB()
        self.lab_analyzer = MockLabAnalyzer()
        self.imaging_service = MockImagingAnalysisService()
        self.ehr_system = MockEHRSystem()
        self.vector_db = MockVectorDB()
        self.available_tools = {
            "query_medical_db": self.medical_db.query,
            "analyze_lab_results": self.lab_analyzer.analyze,
            "imaging_analysis": self.imaging_service.analyze_image,
            "access_ehr": self.ehr_system.access_patient_data,
            "retrieve_knowledge": self.vector_db.search
        }
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "query_medical_db",
                    "description": "Queries an external medical database for information about diseases, drugs, or symptoms.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "The query string for the medical database."}},
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_lab_results",
                    "description": "Analyzes raw lab result data to extract key values and interpretations.",
                    "parameters": {
                        "type": "object",
                        "properties": {"results_data": {"type": "string", "description": "Raw text data of lab results."}},
                        "required": ["results_data"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "imaging_analysis",
                    "description": "Submits a medical image (e.g., X-ray, MRI) for specialized AI analysis and retrieves findings.",
                    "parameters": {
                        "type": "object",
                        "properties": {"image_id": {"type": "string", "description": "Unique identifier for the medical image to be analyzed."}},
                        "required": ["image_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "access_ehr",
                    "description": "Accesses specific patient data from the Electronic Health Record (EHR) system.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {"type": "string", "description": "The ID of the patient."}, 
                            "data_type": {"type": "string", "description": "The type of data to retrieve (e.g., medications, history, diet_logs)."}
                        },
                        "required": ["patient_id", "data_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_knowledge",
                    "description": "Retrieves relevant medical literature, guidelines, or past case information from the internal knowledge base.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "The query for the knowledge base."}},
                        "required": ["query"]
                    }
                }
            }
        ]

    async def _call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Executes a registered tool."""
        if tool_name not in self.available_tools:
            return {"error": f"Tool '{tool_name}' not found."}
        print(f"[Agent] Calling tool: {tool_name} with args: {kwargs}")
        try:
            result = await self.available_tools[tool_name](**kwargs)
            print(f"[Agent] Tool '{tool_name}' returned: {result}")
            return result
        except Exception as e:
            print(f"[Agent] Error calling tool '{tool_name}': {e}")
            return {"error": f"Error executing tool '{tool_name}': {str(e)}"}

    async def _process_llm_response(self, state: AgentState, llm_response: Dict[str, Any]) -> AgentState:
        """Processes the LLM's response, including tool calls and content."""
        if "tool_calls" in llm_response:
            for tool_call in llm_response["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                tool_output = await self._call_tool(function_name, **function_args)
                state["observations"].append({"tool": function_name, "args": function_args, "output": tool_output, "timestamp": datetime.now().isoformat()})
                state["chat_history"].append({"role": "tool", "content": json.dumps(tool_output)})
        elif "content" in llm_response:
            state["chat_history"].append({"role": "assistant", "content": llm_response["content"]})
            # Update hypotheses or other state elements based on LLM content
            if "preliminary diagnosis" in llm_response["content"].lower():
                state["hypotheses"].append(llm_response["content"])
        return state

    async def _reason_and_act(self, state: AgentState) -> AgentState:
        """The core reasoning loop: LLM decides next action (tool call or direct response)."""
        current_prompt = f"Patient ID: {state['patient_id']}\n"
        current_prompt += f"Current Observations: {json.dumps(state['observations'][-3:]) if state['observations'] else 'None'}\n"
        current_prompt += f"Current Hypotheses: {state['hypotheses'][-1] if state['hypotheses'] else 'None'}\n"
        current_prompt += "Based on the above, what is the next step? Do I need to use a tool? If so, which one and with what arguments? Or provide a preliminary diagnosis/reflection if enough info is gathered.\n"
        current_prompt += "Consider the following tools: " + ", ".join(self.available_tools.keys()) + ".\n"

        state["chat_history"].append({"role": "user", "content": current_prompt})
        llm_response = await self.llm.chat(
            messages=state["chat_history"],
            tools=self.tool_definitions,
            tool_choice="auto"
        )
        return await self._process_llm_response(state, llm_response)

    async def _reflect_and_self_correct(self, state: AgentState) -> AgentState:
        """LLM reflects on recent actions and self-corrects if necessary."""
        reflection_prompt = f"Review the latest observations and hypotheses:\n"
        reflection_prompt += f"Observations: {json.dumps(state['observations'][-5:])}\n"
        reflection_prompt += f"Hypotheses: {json.dumps(state['hypotheses'][-1:])}\n"
        reflection_prompt += "Are there any inconsistencies or missing pieces of information? Should I adjust my strategy or query more data? Identify and resolve knowledge conflicts."
        
        state["chat_history"].append({"role": "user", "content": reflection_prompt})
        llm_response = await self.llm.chat(
            messages=state["chat_history"],
            tools=self.tool_definitions,
            tool_choice="auto"
        )
        return await self._process_llm_response(state, llm_response)

    async def _evaluate_termination_condition(self, state: AgentState) -> bool:
        """LLM evaluates if the diagnostic process can be terminated."""
        eval_prompt = f"Based on all gathered observations ({len(state['observations'])} entries) and current hypotheses ({len(state['hypotheses'])} entries), am I confident in a diagnosis and a recommendation? If so, why? If not, what is missing? Output 'TERMINATE' if ready, otherwise 'CONTINUE'.\n"
        eval_prompt += f"Current diagnosis confidence: {state['diagnosis_confidence']}.\n"
        eval_prompt += f"Last hypothesis: {state['hypotheses'][-1] if state['hypotheses'] else 'None'}"
        
        state["chat_history"].append({"role": "user", "content": eval_prompt})
        llm_response = await self.llm.chat(messages=state["chat_history"])
        response_content = llm_response.get("content", "").upper()
        state["chat_history"].append({"role": "assistant", "content": llm_response.get("content", "")})

        if "TERMINATE" in response_content or state["diagnosis_confidence"] >= 0.95: # Simple confidence threshold
            state["termination_reason"] = response_content
            return True
        return False

    async def run_diagnostic_cycle(self, initial_patient_info: Dict[str, Any], max_iterations: int = 10) -> AgentState:
        """Runs the iterative diagnostic process."""
        state = AgentState(
            patient_id=initial_patient_info.get("patient_id"),
            chat_history=[{"role": "user", "content": f"Initial patient data: {json.dumps(initial_patient_info)}"}],
            observations=[{"initial_input": initial_patient_info, "timestamp": datetime.now().isoformat()}],
            hypotheses=[f"Initial assessment based on input: {initial_patient_info.get('symptoms', 'None')}"]
        )

        for i in range(max_iterations):
            print(f"\n--- Diagnostic Cycle {i+1}/{max_iterations} ---")
            
            # 1. Reasoning and Action (Tool Selection/Execution)
            state = await self._reason_and_act(state)

            # 2. Reflection and Self-Correction
            state = await self._reflect_and_self_correct(state)

            # 3. Evaluate Termination
            if await self._evaluate_termination_condition(state):
                print("--- Agent decided to terminate ---")
                break

            # Simulate feedback if available (e.g., from a user UI)
            # In a real system, this would be an external input.
            # For demonstration, let's assume some internal feedback mechanism
            if i == 2: # Example: simulate a physician providing feedback after 3 cycles
                feedback_msg = "Physician feedback: Consider family history more closely. Patient's father also had early-onset diabetes."
                state["feedback_log"].append({"type": "physician_feedback", "content": feedback_msg, "timestamp": datetime.now().isoformat()})
                state["chat_history"].append({"role": "user", "content": feedback_msg})
                state["diagnosis_confidence"] = 0.6 # Adjust confidence based on new info
                print(f"[Agent] Received simulated feedback: {feedback_msg}")
            
            # Example of how LLM might update confidence (simplified)
            if "preliminary diagnosis" in state["chat_history"][-1].get("content", "").lower():
                state["diagnosis_confidence"] = min(0.9, state["diagnosis_confidence"] + 0.1)

        return state

    def ingest_feedback(self, feedback_data: Dict[str, Any]):
        """Simulates ingesting feedback from external sources."""
        self.vector_db.knowledge_base[f"feedback_case_{uuid.uuid4().hex}"] = json.dumps(feedback_data)
        print(f"[Agent] Ingested feedback: {feedback_data.get('type')}")
        # In a real system, this would trigger model fine-tuning or knowledge base updates


# --- FastAPI Backend (Simplified) ---

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Medical Diagnostic Assistant API")
agent_instance = MedicalDiagnosticAgent() # Initialize the agent globally

class PatientInput(BaseModel):
    patient_id: str
    symptoms: str
    lab_results: Optional[str] = None
    imaging_report_id: Optional[str] = None
    additional_info: Optional[str] = None

class FeedbackInput(BaseModel):
    case_id: str
    feedback_type: str # e.g., "diagnosis_confirmation", "correction", "outcome"
    details: str


@app.post("/diagnose", response_model=Dict[str, Any])
async def diagnose_patient(patient_input: PatientInput):
    """Initiates a diagnostic session for a new patient."""
    initial_info = patient_input.dict()
    try:
        final_state = await agent_instance.run_diagnostic_cycle(initial_info)
        return {
            "status": "completed" if final_state["termination_reason"] else "max_iterations_reached",
            "patient_id": final_state["patient_id"],
            "final_diagnosis_summary": final_state["hypotheses"][-1] if final_state["hypotheses"] else "No clear diagnosis.",
            "observations": final_state["observations"],
            "chat_history_summary": [msg for msg in final_state["chat_history"] if msg["role"] == "assistant" or msg["role"] == "user"],
            "termination_reason": final_state["termination_reason"],
            "diagnosis_confidence": final_state["diagnosis_confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic process failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(feedback_input: FeedbackInput):
    """Submits feedback on a diagnostic case."""
    agent_instance.ingest_feedback(feedback_input.dict())
    return {"message": "Feedback received and processed.", "case_id": feedback_input.case_id}

@app.get("/health")
async def health_check():
    return {"status": "ok", "agent_ready": True}


# --- Streamlit UI (Conceptual - run separately if needed) ---
# To run this, you would need to save it as a separate .py file and run `streamlit run your_streamlit_app.py`
# For simplicity, including as comments here.

"""
# streamlit_app.py (conceptual)

import streamlit as st
import requests
import json

# Assuming FastAPI is running on http://127.0.0.1:8000
FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Intelligent Medical Diagnostic Assistant")
st.title("🩺 Intelligent Medical Diagnostic Assistant")
st.markdown("---\n")

# --- Patient Diagnosis Section ---
st.header("New Patient Diagnosis")

with st.form("diagnosis_form"):
    patient_id = st.text_input("Patient ID", value="P123")
    symptoms = st.text_area("Symptoms", "Patient reports fatigue, increased thirst, and frequent urination for 2 months.")
    lab_results = st.text_area("Lab Results (raw data)", "Glucose: 150 mg/dL, HbA1c: 7.2%, Cholesterol: 200 mg/dL")
    imaging_report_id = st.text_input("Imaging Report ID (e.g., chest_xray_p123)", "")
    additional_info = st.text_area("Additional Info", "Family history of Type 2 Diabetes.")

    submitted = st.form_submit_button("Run Diagnosis")
    if submitted:
        payload = {
            "patient_id": patient_id,
            "symptoms": symptoms,
            "lab_results": lab_results if lab_results else None,
            "imaging_report_id": imaging_report_id if imaging_report_id else None,
            "additional_info": additional_info if additional_info else None,
        }
        st.json(payload)
        try:
            response = requests.post(f"{FASTAPI_URL}/diagnose", json=payload)
            response.raise_for_status() # Raise an exception for HTTP errors
            diagnosis_data = response.json()
            st.success("Diagnosis Process Completed!")
            st.subheader("Final Diagnosis Summary")
            st.write(diagnosis_data.get("final_diagnosis_summary", "N/A"))
            st.subheader("Confidence")
            st.write(f"{diagnosis_data.get('diagnosis_confidence', 0)*100:.1f}%")
            st.subheader("Termination Reason")
            st.write(diagnosis_data.get("termination_reason", "N/A"))

            st.subheader("Agent Chat History")
            for msg in diagnosis_data.get("chat_history_summary", []):
                st.json(msg)

            st.subheader("Detailed Observations")
            st.json(diagnosis_data.get("observations", []))

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Is it running?")
        except requests.exceptions.RequestException as e:
            st.error(f"Error during diagnosis: {e}")
            if response.status_code:
                st.error(f"Response: {response.json()}")

st.markdown("---\n")

# --- Feedback Section ---
st.header("Submit Feedback")
with st.form("feedback_form"):
    feedback_case_id = st.text_input("Case ID for Feedback", "")
    feedback_type = st.selectbox("Feedback Type", ["diagnosis_confirmation", "correction", "outcome", "new_information"])
    feedback_details = st.text_area("Details of Feedback")
    
    feedback_submitted = st.form_submit_button("Submit Feedback")
    if feedback_submitted:
        if not feedback_case_id:
            st.error("Please provide a Case ID for feedback.")
        else:
            feedback_payload = {
                "case_id": feedback_case_id,
                "feedback_type": feedback_type,
                "details": feedback_details
            }
            try:
                response = requests.post(f"{FASTAPI_URL}/feedback", json=feedback_payload)
                response.raise_for_status()
                st.success("Feedback submitted successfully!")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Is it running?")
            except requests.exceptions.RequestException as e:
                st.error(f"Error submitting feedback: {e}")
                if response.status_code:
                    st.error(f"Response: {response.json()}")


"""

# To run the FastAPI application:
# 1. Save this file as `medical_diagnostic_agent.py`.
# 2. Run `uvicorn medical_diagnostic_agent:app --reload` in your terminal.
# 3. Access the API documentation at `http://127.0.0.1:8000/docs`.

# To run the Streamlit UI (if uncommented and saved separately as e.g., `streamlit_app.py`):
# 1. Ensure `streamlit` is installed (`pip install streamlit requests`)
# 2. Run `streamlit run streamlit_app.py` in a separate terminal. (Ensure FastAPI is running first)


# Example of how to interact with the agent directly (for testing without FastAPI):
async def main_direct_test():
    print("\n--- Direct Agent Test --- ")
    agent = MedicalDiagnosticAgent()
    initial_patient_data = {
        "patient_id": "P123",
        "symptoms": "Patient reports fatigue, increased thirst, and frequent urination for 2 months.",
        "lab_results": "Glucose: 150 mg/dL, HbA1c: 7.2%",
        "additional_info": "Family history of Type 2 Diabetes. Patient is 45 years old."
    }
    
    final_state = await agent.run_diagnostic_cycle(initial_patient_data)
    print("\n--- Final Agent State (Direct Test) ---")
    print(f"Diagnosis Summary: {final_state['hypotheses'][-1] if final_state['hypotheses'] else 'No clear diagnosis.'}")
    print(f"Termination Reason: {final_state['termination_reason']}")
    print(f"Confidence: {final_state['diagnosis_confidence']}")
    # print("Full chat history:")
    # for msg in final_state["chat_history"]:
    #     print(msg)

if __name__ == "__main__":
    # To run the FastAPI application, use `uvicorn medical_diagnostic_agent:app --reload`
    # This 'if __name__ == "__main__"' block is for direct testing of the agent logic.
    # For the full application, run FastAPI and optionally Streamlit separately.
    asyncio.run(main_direct_test())
