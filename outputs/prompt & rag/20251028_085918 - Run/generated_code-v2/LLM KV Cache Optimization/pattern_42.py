import torch
import numpy as np
from loguru import logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import chromadb
import os

logger.add("file.log", rotation="10 MB")

client = chromadb.Client()
collection = client.get_or_create_collection(name="medical_knowledge_replication")

if collection.count() == 0:
    medical_docs = [
        "Common symptoms of influenza include fever, cough, sore throat, and body aches. It is a viral infection.",
        "Treatment for mild influenza often involves rest, fluids, and over-the-counter medications like acetaminophen. Antivirals may be prescribed in severe cases.",
        "Diabetes is a chronic metabolic condition characterized by high blood sugar levels over a prolonged period. Type 1 is autoimmune, Type 2 is insulin resistance.",
        "Managing type 2 diabetes involves lifestyle changes (diet, exercise) and often medication like metformin or insulin therapy.",
        "Hypertension, or high blood pressure, significantly increases the risk of heart disease, stroke, and kidney disease. A reading above 130/80 mmHg is generally considered hypertensive.",
        "Lifestyle changes such as a healthy diet (low sodium), regular exercise, weight management, and avoiding smoking are crucial for managing hypertension. Medications like ACE inhibitors or beta-blockers may also be prescribed."
    ]
    
    temp_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = temp_embedder.encode(medical_docs).tolist()
    collection.add(
        embeddings=doc_embeddings,
        documents=medical_docs,
        ids=[f"doc{i}" for i in range(len(medical_docs))]
    )
    logger.info("ChromaDB initialized with sample medical knowledge.")

class KVCacheManager:
    def __init__(self):
        self.gpu_cache = {}
        self.host_cache = {}
        self.gpu_failed = False
        logger.info("KV Cache Manager initialized.")

    def store_critical_kv(self, key: str, value: str):
        if not self.gpu_failed:
            self.gpu_cache[key] = value
            logger.info(f"Stored '{key}' in simulated GPU cache.")
        else:
            logger.warning(f"GPU failed. Not storing '{key}' in GPU cache, only host.")

        self.host_cache[key] = value
        logger.info(f"Replicated '{key}' to host memory.")

    def retrieve_critical_kv(self, key: str):
        if not self.gpu_failed and key in self.gpu_cache:
            logger.info(f"Retrieved '{key}' from simulated GPU cache.")
            return self.gpu_cache[key]
        elif key in self.host_cache:
            logger.warning(f"Simulated GPU cache lookup failed or GPU failed. Retrieving '{key}' from host cache.")
            return self.host_cache[key]
        else:
            logger.warning(f"'{key}' not found in either simulated GPU or host cache.")
            return None

    def simulate_gpu_failure(self):
        self.gpu_failed = True
        self.gpu_cache = {}
        logger.error("Simulating GPU failure. Simulated GPU cache cleared.")

    def recover_gpu_cache(self):
        if not self.gpu_failed:
            logger.info("Simulated GPU is not in a failed state. No recovery needed.")
            return

        for key, data in self.host_cache.items():
            self.gpu_cache[key] = data
        self.gpu_failed = False
        logger.info("Recovered simulated GPU cache from host memory.")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

logger.info("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
logger.info("Embedding model loaded.")

logger.info("Loading LLM model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)
logger.info("LLM model loaded.")

app = FastAPI()
kv_cache_manager = KVCacheManager()

class DiagnosisRequest(BaseModel):
    symptoms: str
    patient_context: str

@app.post("/diagnose")
async def diagnose(request: DiagnosisRequest):
    try:
        query_text = f"Symptoms: {request.symptoms}. Patient Context: {request.patient_context}"
        query_embedding = embedder.encode(query_text).tolist()

        retrieved_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=['documents']
        )
        retrieved_documents = retrieved_results['documents'][0] if retrieved_results and retrieved_results['documents'] else []
        logger.info(f"Retrieved documents: {retrieved_documents}")

        kv_cache_manager.store_critical_kv("patient_context", request.patient_context)
        
        combined_retrieved_docs_text = " ".join(retrieved_documents)
        if combined_retrieved_docs_text:
            kv_cache_manager.store_critical_kv("retrieved_medical_docs", combined_retrieved_docs_text)

        cached_patient_context = kv_cache_manager.retrieve_critical_kv("patient_context")
        cached_medical_docs = kv_cache_manager.retrieve_critical_kv("retrieved_medical_docs")

        prompt_parts = []
        prompt_parts.append("As a Medical Diagnosis Assistant, provide insights based on the following information.")
        
        if cached_patient_context:
            prompt_parts.append(f"Cached Patient History: {cached_patient_context}")
        else:
            prompt_parts.append(f"Patient History (from request): {request.patient_context}")

        if cached_medical_docs:
            prompt_parts.append(f"Cached Medical Information: {cached_medical_docs}")
        elif retrieved_documents:
            prompt_parts.append(f"Retrieved Medical Information: {combined_retrieved_docs_text}")
        
        prompt_parts.append(f"Current Symptoms: {request.symptoms}")
        prompt_parts.append("Provide a concise diagnosis or relevant medical insights.")

        final_prompt = "\n\n".join(prompt_parts)
        
        logger.info(f"Final Prompt:\n{final_prompt}")

        inputs = tokenizer(final_prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=5,
            early_stopping=True,
            temperature=0.7,
            top_p=0.9
        )
        diagnosis_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"diagnosis": diagnosis_text, "retrieved_documents": retrieved_documents, "using_cached_patient_context": bool(cached_patient_context), "using_cached_medical_docs": bool(cached_medical_docs)}

    except Exception as e:
        logger.error(f"Error during diagnosis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate_gpu_failure")
async def simulate_failure():
    kv_cache_manager.simulate_gpu_failure()
    return {"message": "GPU failure simulated. Critical KV cache on GPU cleared."}

@app.post("/recover_cache")
async def recover_cache():
    kv_cache_manager.recover_gpu_cache()
    return {"message": "Critical KV cache recovered from host memory."}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu_status": "failed" if kv_cache_manager.gpu_failed else "active"}