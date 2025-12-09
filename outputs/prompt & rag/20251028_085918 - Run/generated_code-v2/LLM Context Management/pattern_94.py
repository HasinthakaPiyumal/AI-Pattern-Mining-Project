import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import datetime
import uuid
import os

class LLMIntegration:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name

    def get_response(self, prompt: str) -> str:
        if "medication" in prompt.lower():
            return "Based on your records, your current medication is XYZ. Please ensure you take it as prescribed. Do you have any specific questions about it?"
        elif "symptoms" in prompt.lower():
            return "I see you've logged recent symptoms. Can you tell me more about their severity or frequency?"
        elif "health insights" in prompt.lower():
            return "Your recent health trends show consistent blood pressure readings within the normal range, which is good. Keep up with your healthy habits!"
        else:
            return f"Hello! How can I assist you with your chronic disease management today based on: {prompt[:100]}..."

patient_profiles_db: Dict[str, Dict[str, Any]] = {}
medical_records_db: Dict[str, List[Dict[str, Any]]] = {}
symptom_logs_db: Dict[str, List[Dict[str, Any]]] = {}
conversation_history_db: Dict[str, List[Dict[str, Any]]] = {}

chroma_client = chromadb.Client()
vector_collection = chroma_client.get_or_create_collection(name="patient_medical_data")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class LongContextManager:
    def __init__(self):
        self.embedding_model = embedding_model
        self.vector_collection = vector_collection

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()

    def add_medical_record(self, patient_id: str, record_type: str, content: str, timestamp: str) -> str:
        record_id = str(uuid.uuid4())
        record_data = {
            "id": record_id,
            "patient_id": patient_id,
            "type": record_type,
            "content": content,
            "timestamp": timestamp
        }
        
        if patient_id not in medical_records_db:
            medical_records_db[patient_id] = []
        medical_records_db[patient_id].append(record_data)

        self.vector_collection.add(
            documents=[content],
            metadatas=[{"patient_id": patient_id, "type": record_type, "timestamp": timestamp}],
            ids=[record_id]
        )
        return record_id

    def log_symptom_data(self, patient_id: str, symptom: str, severity: int, notes: str, timestamp: str) -> str:
        log_id = str(uuid.uuid4())
        log_data = {
            "id": log_id,
            "patient_id": patient_id,
            "symptom": symptom,
            "severity": severity,
            "notes": notes,
            "timestamp": timestamp
        }
        
        if patient_id not in symptom_logs_db:
            symptom_logs_db[patient_id] = []
        symptom_logs_db[patient_id].append(log_data)

        content = f"Patient reported symptom: {symptom}, severity: {severity}, notes: {notes}"
        self.vector_collection.add(
            documents=[content],
            metadatas=[{"patient_id": patient_id, "type": "symptom_log", "timestamp": timestamp}],
            ids=[log_id]
        )
        return log_id

    def get_patient_structured_data(self, patient_id: str) -> Dict[str, Any]:
        profile = patient_profiles_db.get(patient_id, {})
        records = medical_records_db.get(patient_id, [])
        symptoms = symptom_logs_db.get(patient_id, [])
        
        return {
            "profile": profile,
            "medical_records": records,
            "symptom_logs": symptoms
        }

    def retrieve_relevant_chunks(self, patient_id: str, query: str, top_k: int = 5) -> List[str]:
        query_embedding = self._get_embedding(query)
        
        results = self.vector_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"patient_id": patient_id}
        )
        
        documents = results['documents'][0] if results['documents'] else []
        return documents

    def _summarize_text(self, text: str, max_length: int = 100) -> str:
        if len(text.split()) > max_length:
            return " ".join(text.split()[:max_length]) + "..."
        return text

    def prioritize_and_format_context(self, patient_id: str, user_query: str, retrieved_chunks: List[str]) -> str:
        context_parts = []

        structured_data = self.get_patient_structured_data(patient_id)
        if structured_data["profile"]:
            context_parts.append(f"Patient Profile: {self._summarize_text(str(structured_data['profile']), 50)}")

        recent_records = sorted(
            structured_data["medical_records"],
            key=lambda x: datetime.datetime.strptime(x["timestamp"], "%Y-%m-%dT%H:%M:%S.%f") if isinstance(x["timestamp"], str) else x["timestamp"],
            reverse=True
        )[:2]
        for record in recent_records:
            context_parts.append(f"Recent Medical Record ({record['type']} on {record['timestamp']}): {self._summarize_text(record['content'])}")
        
        recent_symptoms = sorted(
            structured_data["symptom_logs"],
            key=lambda x: datetime.datetime.strptime(x["timestamp"], "%Y-%m-%dT%H:%M:%S.%f") if isinstance(x["timestamp"], str) else x["timestamp"],
            reverse=True
        )[:2]
        for log in recent_symptoms:
            context_parts.append(f"Recent Symptom Log ({log['symptom']} on {log['timestamp']}): {self._summarize_text(log['notes'] or '')}")

        for chunk in retrieved_chunks:
            context_parts.append(f"Relevant Information: {self._summarize_text(chunk)}")

        patient_conversation = conversation_history_db.get(patient_id, [])
        if patient_conversation:
            last_turns = " ".join([t['content'] for t in patient_conversation[-2:]])
            if last_turns:
                context_parts.append(f"Previous Conversation Snippet: {self._summarize_text(last_turns, 80)}")

        formatted_context = "\n".join(context_parts)
        
        max_context_length_chars = 1500
        if len(formatted_context) > max_context_length_chars:
            formatted_context = self._summarize_text(formatted_context, max_length=int(max_context_length_chars / 5))

        return formatted_context

class RAGSystem:
    def __init__(self, llm_integration: LLMIntegration, context_manager: LongContextManager):
        self.llm = llm_integration
        self.context_manager = context_manager

    def generate_response(self, patient_id: str, user_query: str) -> str:
        retrieved_chunks = self.context_manager.retrieve_relevant_chunks(patient_id, user_query)

        formatted_context = self.context_manager.prioritize_and_format_context(patient_id, user_query, retrieved_chunks)

        system_prompt = (
            "You are a personalized healthcare assistant for chronic disease management. "
            "Provide empathetic, accurate, and personalized advice based on the patient's medical history. "
            "Always prioritize patient safety and suggest consulting a doctor for critical decisions. "
            "Use the provided context to answer the user's query."
        )
        
        full_prompt = f"{system_prompt}\n\nContext:\n{formatted_context}\n\nPatient's Query: {user_query}\n\nResponse:"

        llm_response = self.llm.get_response(full_prompt)

        if patient_id not in conversation_history_db:
            conversation_history_db[patient_id] = []
        conversation_history_db[patient_id].append({"role": "user", "content": user_query, "timestamp": str(datetime.datetime.now())})
        conversation_history_db[patient_id].append({"role": "assistant", "content": llm_response, "timestamp": str(datetime.datetime.now())})

        return llm_response

app = FastAPI(title="Personalized Healthcare Assistant Backend")

llm_integration = LLMIntegration()
context_manager = LongContextManager()
rag_system = RAGSystem(llm_integration, context_manager)

class PatientProfile(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    conditions: List[str]
    allergies: List[str] = []
    medications: List[str] = []
    
class MedicalRecordInput(BaseModel):
    patient_id: str
    record_type: str
    content: str

class SymptomLogInput(BaseModel):
    patient_id: str
    symptom: str
    severity: int
    notes: Optional[str] = None

class ChatRequest(BaseModel):
    patient_id: str
    query: str

class PatientProfileResponse(BaseModel):
    message: str
    patient_data: Dict[str, Any]

class OperationResponse(BaseModel):
    message: str
    id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

@app.post("/patient_profile", response_model=OperationResponse)
async def create_or_update_patient_profile(profile: PatientProfile):
    patient_profiles_db[profile.patient_id] = profile.dict()
    return OperationResponse(message=f"Patient profile for {profile.name} created/updated successfully.")

@app.post("/add_medical_record", response_model=OperationResponse)
async def add_medical_record(record_input: MedicalRecordInput):
    if record_input.patient_id not in patient_profiles_db:
        raise HTTPException(status_code=404, detail="Patient not found.")
    
    timestamp = str(datetime.datetime.now())
    record_id = context_manager.add_medical_record(
        record_input.patient_id,
        record_input.record_type,
        record_input.content,
        timestamp
    )
    return OperationResponse(message="Medical record added successfully.", id=record_id)

@app.post("/log_symptom", response_model=OperationResponse)
async def log_symptom(symptom_input: SymptomLogInput):
    if symptom_input.patient_id not in patient_profiles_db:
        raise HTTPException(status_code=404, detail="Patient not found.")
    
    timestamp = str(datetime.datetime.now())
    log_id = context_manager.log_symptom_data(
        symptom_input.patient_id,
        symptom_input.symptom,
        symptom_input.severity,
        symptom_input.notes,
        timestamp
    )
    return OperationResponse(message="Symptom logged successfully.", id=log_id)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(chat_request: ChatRequest):
    if chat_request.patient_id not in patient_profiles_db:
        raise HTTPException(status_code=404, detail="Patient not found. Please create a patient profile first.")
    
    response = rag_system.generate_response(chat_request.patient_id, chat_request.query)
    return ChatResponse(response=response)

@app.get("/patient_data/{patient_id}", response_model=PatientProfileResponse)
async def get_patient_data(patient_id: str):
    if patient_id not in patient_profiles_db:
        raise HTTPException(status_code=404, detail="Patient not found.")
    
    data = context_manager.get_patient_structured_data(patient_id)
    data["conversation_history"] = conversation_history_db.get(patient_id, [])
    
    return PatientProfileResponse(message="Patient data retrieved successfully.", patient_data=data)

if __name__ == "__main__":
    print("Starting FastAPI server. Access at http://127.0.0.1:8000")
    print("Run `uvicorn healthcare_assistant:app --reload` in your terminal to enable hot-reloading.")
    uvicorn.run(app, host="0.0.0.0", port=8000)