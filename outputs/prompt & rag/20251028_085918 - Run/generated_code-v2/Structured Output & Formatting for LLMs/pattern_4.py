from fastapi import FastAPI
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

# Patch the OpenAI client with instructor
client = instructor.patch(OpenAI())

# 1. Define the MedicalReport Pydantic model
class MedicalReport(BaseModel):
    diagnoses: list[str] = Field(..., description="List of diagnoses identified in the medical note.")
    medications: list[str] = Field(..., description="List of medications prescribed or mentioned in the medical note.")
    allergies: list[str] = Field(..., description="List of allergies identified in the medical note.")
    treatment_plans: list[str] = Field(..., description="List of treatment plans outlined in the medical note.")

app = FastAPI(
    title="Medical Report Data Extractor",
    description="API to extract structured medical data from unstructured notes."
)

# 2. Define the /extract_medical_data POST endpoint
@app.post("/extract_medical_data", response_model=MedicalReport)
async def extract_medical_data(medical_note: str):
    """Extracts structured medical data from an unstructured medical note."""
    try:
        # Use instructor with OpenAI client to generate MedicalReport
        report = client.chat.completions.create(
            model="gpt-4o", # You can choose a suitable OpenAI model
            response_model=MedicalReport,
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly accurate medical data extraction assistant. Extract all relevant medical information from the provided note into the specified JSON format.",
                },
                {
                    "role": "user",
                    "content": f"Extract the diagnoses, medications, allergies, and treatment plans from the following medical note:\n\n{medical_note}"
                },
            ],
            temperature=0,
        )
        return report
    except Exception as e:
        # In a real application, you'd want more robust error handling and logging
        raise HTTPException(status_code=500, detail=str(e))

# To run this application, save it as main.py and use uvicorn:
# uvicorn main:app --reload

# Example usage with curl:
# curl -X POST "http://127.0.0.1:8000/extract_medical_data" -H "Content-Type: application/json" -d '"Patient presented with severe headache and fever. Diagnosed with viral meningitis. Prescribed Ibuprofen 400mg every 6 hours and instructed to rest. No known allergies. Follow-up in one week."'