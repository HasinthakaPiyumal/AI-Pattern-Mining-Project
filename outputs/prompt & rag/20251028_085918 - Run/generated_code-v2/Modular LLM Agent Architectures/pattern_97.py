from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Simulated Medical Knowledge Base
medical_kb = [
    {
        "title": "Diabetes Mellitus Type 2 - Management Guidelines",
        "content": "Type 2 Diabetes Mellitus management typically involves lifestyle modifications (diet, exercise), metformin as first-line therapy, followed by other oral agents or insulin if glycemic targets are not met. Regular monitoring of blood glucose, HbA1c, and renal function is crucial. Complications include neuropathy, retinopathy, and nephropathy."
    },
    {
        "title": "Hypertension - Treatment Options",
        "content": "First-line treatments for hypertension include ACE inhibitors, ARBs, thiazide diuretics, and calcium channel blockers. Lifestyle changes such as reduced sodium intake, regular physical activity, and weight management are also critical. Regular blood pressure monitoring is essential."
    },
    {
        "title": "Common Cold - Symptomatic Relief",
        "content": "The common cold is a viral infection with no specific cure. Treatment focuses on symptomatic relief, including rest, hydration, pain relievers (acetaminophen, ibuprofen), decongestants, and cough suppressants. Antibiotics are ineffective against viral colds."
    },
    {
        "title": "Aspirin - Uses and Side Effects",
        "content": "Aspirin is used as an analgesic, anti-inflammatory, antipyretic, and antiplatelet agent. Common side effects include gastrointestinal upset, heartburn, and increased bleeding risk. It should be used with caution in children due to Reye's syndrome risk."
    }
]

class AugmentRequest(BaseModel):
    query: str

class AugmentResponse(BaseModel):
    augmented_context: str

@app.post("/augment", response_model=AugmentResponse)
async def augment_medical_query(request: AugmentRequest):
    relevant_info = []
    query_lower = request.query.lower()

    for entry in medical_kb:
        if query_lower in entry["title"].lower() or query_lower in entry["content"].lower():
            relevant_info.append(entry["content"])
    
    if not relevant_info:
        augmented_context = "No specific relevant medical information found in the knowledge base."
    else:
        augmented_context = "\n\n".join(relevant_info)
    
    return AugmentResponse(augmented_context=augmented_context)