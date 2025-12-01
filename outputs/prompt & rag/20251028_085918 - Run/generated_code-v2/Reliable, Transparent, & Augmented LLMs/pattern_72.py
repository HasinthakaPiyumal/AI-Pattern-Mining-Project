import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Intelligent Bias-Aware Recruitment Assistant API")

# --- 1. Data Ingestion & Preprocessing Layer ---

class DocumentParser:
    def parse_text(self, document_content: str) -> str:
        # In a real application, this would parse PDF/DOCX. Here, we assume clean text input.
        return document_content

class DataCleanerAnonymizer:
    def anonymize_text(self, text: str) -> str:
        # Simple anonymization: replace common names, gendered pronouns with neutral placeholders
        text = re.sub(r'\b(he|she|him|her|his|hers)\b', 'they', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(Mr\.?|Ms\.?|Mrs\.?|Dr\.?)\s[A-Z][a-z]+\b', 'Candidate', text)
        # Further anonymization could involve more sophisticated NER and replacement
        return text

class TextEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

# --- 2. Core AI Matching & Ranking Layer ---

class SemanticMatcher:
    def calculate_similarity(self, job_embedding: np.ndarray, resume_embeddings: np.ndarray) -> np.ndarray:
        # Reshape job_embedding for cosine_similarity function
        job_embedding = job_embedding.reshape(1, -1)
        similarities = cosine_similarity(job_embedding, resume_embeddings)
        return similarities.flatten()

class CandidateRankingEngine:
    def rank_candidates(self, similarities: np.ndarray, candidate_ids: List[str]) -> List[Dict[str, Any]]:
        ranked_list = []
        for i, sim in enumerate(similarities):
            ranked_list.append({"candidate_id": candidate_ids[i], "similarity_score": float(sim)})
        ranked_list.sort(key=lambda x: x['similarity_score'], reverse=True)
        return ranked_list

# --- 3. Bias Detection & Mitigation Layer ---

class BiasDetector:
    def detect_bias(self, ranked_candidates: List[Dict[str, Any]], anonymized_resumes: Dict[str, str]) -> List[Dict[str, Any]]:
        # This is a highly simplified bias detection. In a real system, this would involve
        # more complex statistical analysis and potentially inferring protected attributes
        # from anonymized text, or using synthetic data for fairness evaluation.
        logging.info("Performing simplified bias detection...")

        # Example: Check for potential gendered language if not perfectly anonymized
        biased_flags = []
        for candidate in ranked_candidates:
            resume_text = anonymized_resumes.get(candidate['candidate_id'], '').lower()
            if "woman in tech" in resume_text or "man in engineering" in resume_text:
                biased_flags.append({"candidate_id": candidate['candidate_id'], "flag": "Potential gendered language detected"})
        
        return biased_flags


class DebiasingModule:
    def apply_debiasing(self, ranked_candidates: List[Dict[str, Any]], bias_flags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Placeholder for debiasing logic. This could involve re-ranking, adjusting scores,
        # or removing candidates based on severe bias flags.
        logging.info("Applying simplified debiasing...")
        # For this example, we just add a 'bias_warning' to candidates that were flagged
        debiased_candidates = []
        flagged_ids = {flag['candidate_id'] for flag in bias_flags}
        for candidate in ranked_candidates:
            if candidate['candidate_id'] in flagged_ids:
                candidate['bias_warning'] = "Detected potential bias. Review manually."
            debiased_candidates.append(candidate)
        return debiased_candidates

class ExplainabilityModule:
    def explain_match(self, job_description: str, resume_content: str) -> Dict[str, Any]:
        # Simplified XAI: find common keywords or phrases
        job_tokens = set(re.findall(r'\b\w+\b', job_description.lower()))
        resume_tokens = set(re.findall(r'\b\w+\b', resume_content.lower()))
        common_keywords = list(job_tokens.intersection(resume_tokens))
        return {"explanation": "Keywords in common", "keywords": common_keywords[:5]}

# --- In-memory Data Store (for demonstration) ---
job_descriptions_db: Dict[str, Dict[str, Any]] = {}
resumes_db: Dict[str, Dict[str, Any]] = {}

# --- API Endpoints --- 

parser = DocumentParser()
cleaner = DataCleanerAnonymizer()
embedder = TextEmbedder()
matcher = SemanticMatcher()
ranker = CandidateRankingEngine()
bias_detector = BiasDetector()
debiasing_module = DebiasingModule()
xai_module = ExplainabilityModule()

class JobDescriptionInput(BaseModel):
    job_id: str
    content: str

class ResumeInput(BaseModel):
    resume_id: str
    content: str

@app.post("/job_description")
async def upload_job_description(job: JobDescriptionInput):
    parsed_content = parser.parse_text(job.content)
    anonymized_content = cleaner.anonymize_text(parsed_content)
    embedding = embedder.embed_text([anonymized_content])[0]

    job_descriptions_db[job.job_id] = {
        "original_content": job.content,
        "anonymized_content": anonymized_content,
        "embedding": embedding.tolist()
    }
    logging.info(f"Job description '{job.job_id}' uploaded and processed.")
    return {"message": f"Job description '{job.job_id}' processed successfully."}

@app.post("/resume")
async def upload_resume(resume: ResumeInput):
    parsed_content = parser.parse_text(resume.content)
    anonymized_content = cleaner.anonymize_text(parsed_content)
    embedding = embedder.embed_text([anonymized_content])[0]

    resumes_db[resume.resume_id] = {
        "original_content": resume.content,
        "anonymized_content": anonymized_content,
        "embedding": embedding.tolist()
    }
    logging.info(f"Resume '{resume.resume_id}' uploaded and processed.")
    return {"message": f"Resume '{resume.resume_id}' processed successfully."}

@app.get("/match/{job_id}")
async def get_matches_for_job(job_id: str):
    if job_id not in job_descriptions_db:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")

    job_embedding = np.array(job_descriptions_db[job_id]["embedding"])

    if not resumes_db:
        return {"message": "No resumes uploaded yet.", "ranked_candidates": []}

    resume_ids = list(resumes_db.keys())
    resume_embeddings_list = [resumes_db[rid]["embedding"] for rid in resume_ids]
    resume_embeddings_array = np.array(resume_embeddings_list)

    similarities = matcher.calculate_similarity(job_embedding, resume_embeddings_array)
    ranked_candidates = ranker.rank_candidates(similarities, resume_ids)
    
    # Prepare anonymized resumes dict for bias detection
    anonymized_resumes_for_bias_check = {rid: resumes_db[rid]["anonymized_content"] for rid in resume_ids}

    bias_flags = bias_detector.detect_bias(ranked_candidates, anonymized_resumes_for_bias_check)
    final_ranked_candidates = debiasing_module.apply_debiasing(ranked_candidates, bias_flags)

    # Add explanations for top N candidates (e.g., top 3)
    for candidate in final_ranked_candidates[:3]:
        resume_content = resumes_db[candidate['candidate_id']]['anonymized_content']
        job_content = job_descriptions_db[job_id]['anonymized_content']
        candidate['explanation'] = xai_module.explain_match(job_content, resume_content)

    logging.info(f"Matches retrieved for job '{job_id}'.")
    return {"job_id": job_id, "ranked_candidates": final_ranked_candidates}

@app.get("/resume_details/{resume_id}")
async def get_resume_details(resume_id: str):
    if resume_id not in resumes_db:
        raise HTTPException(status_code=404, detail=f"Resume ID '{resume_id}' not found.")
    return resumes_db[resume_id]

@app.get("/job_details/{job_id}")
async def get_job_details(job_id: str):
    if job_id not in job_descriptions_db:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
    return job_descriptions_db[job_id]