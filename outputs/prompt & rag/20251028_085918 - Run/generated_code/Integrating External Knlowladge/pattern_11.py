
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from llm_medical_assistant import MedicalDiagnosticAssistant

app = FastAPI(
    title="Medical Diagnostic Assistant API",
    description="An AI-powered assistant for healthcare professionals to aid in diagnostics and treatment planning."
)

# Initialize the medical assistant outside the request scope to avoid re-initialization
try:
    medical_assistant = MedicalDiagnosticAssistant()
except Exception as e:
    raise RuntimeError(f"Failed to initialize MedicalDiagnosticAssistant: {e}")

class DiagnosticQuery(BaseModel):
    symptoms: List[str]
    patient_history: str = ""
    additional_info: str = ""

class ToolUseQuery(BaseModel):
    query: str

@app.post("/diagnose", summary="Get a diagnostic aid based on patient information")
async def get_diagnostic_aid(query: DiagnosticQuery):
    """Simulates a diagnostic aid request, using RAG and LLM to provide insights."""
    full_query = f"Patient Symptoms: {', '.join(query.symptoms)}. Patient History: {query.patient_history}. Additional Info: {query.additional_info}"
    
    try:
        response = medical_assistant.run_diagnostic_chain(full_query)
        return {"diagnostic_aid": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during diagnosis: {str(e)}")

@app.post("/tool_query", summary="Query the assistant to use its integrated tools")
async def query_with_tools(tool_query: ToolUseQuery):
    """Allows direct querying to leverage the assistant's integrated tools (e.g., medical search, drug checker)."""
    try:
        response = medical_assistant.run_tool_chain(tool_query.query)
        return {"tool_response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error using tools: {str(e)}")

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Checks if the API is running and the medical assistant is initialized."""
    return {"status": "healthy", "message": "Medical Diagnostic Assistant API is operational."}

