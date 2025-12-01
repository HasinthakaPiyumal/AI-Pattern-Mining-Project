import nltk
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

# Download necessary NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

# 1. Feature Extraction & Embedding Layer
# Load a pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 1. Data Ingestion & Preprocessing Layer
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens)

def get_embeddings(texts):
    return model.encode(texts)

# 2. Bias Detection (Conceptual) & Mitigation Layer
def detect_bias(candidate_text):
    # A very simplistic placeholder for bias detection.
    # In a real system, this would involve sophisticated NLP and fairness metrics.
    # Example: detecting gendered pronouns, age-related terms, or proxies for protected attributes.
    bias_keywords = ["male", "female", "young", "old", "recent graduate", "ivy league"]
    bias_score = 0.0
    processed_text = candidate_text.lower()
    for keyword in bias_keywords:
        if keyword in processed_text:
            bias_score += 0.2  # Arbitrary bias increase
    return min(bias_score, 1.0)  # Cap at 1.0

def debias_score(similarity_score, bias_factor):
    # Simple debiasing: reduce similarity if a bias is detected.
    # The reduction factor (0.5 here) can be tuned.
    return similarity_score * (1 - bias_factor * 0.5)

# 3. Fairness-Aware Ranking Layer
def rank_candidates(job_description, candidate_resumes):
    processed_job_desc = preprocess_text(job_description)
    processed_candidates = [preprocess_text(res) for res in candidate_resumes]

    job_embedding = get_embeddings([processed_job_desc])[0]
    candidate_embeddings = get_embeddings(processed_candidates)

    similarities = cosine_similarity([job_embedding], candidate_embeddings)[0]

    ranked_candidates = []
    for i, (sim, candidate_text) in enumerate(zip(similarities, candidate_resumes)):
        bias_score = detect_bias(candidate_text)
        debiased_sim = debias_score(sim, bias_score)
        ranked_candidates.append({
            "candidate_index": i + 1,
            "original_similarity": float(sim),
            "bias_detected_score": float(bias_score),
            "debiased_similarity": float(debiased_sim),
            "candidate_text": candidate_text
        })

    # Sort by debiased similarity in descending order
    ranked_candidates.sort(key=lambda x: x["debiased_similarity"], reverse=True)
    return ranked_candidates

# 4. API Layer (FastAPI)
app = FastAPI()

class CandidateRankingRequest(BaseModel):
    job_description: str
    candidate_resumes: list[str]

@app.post("/rank_candidates")
async def rank_candidates_api(request: CandidateRankingRequest):
    ranked = rank_candidates(request.job_description, request.candidate_resumes)
    return {"ranked_candidates": ranked}

# 5. User Interface Layer (Gradio)
def gradio_interface_fn(job_desc, candidate_resumes_str):
    candidate_resumes = [res.strip() for res in candidate_resumes_str.split("\
---\
") if res.strip()]
    if not candidate_resumes:
        return "Please provide at least one candidate resume, separated by '---'."
    
    ranked_results = rank_candidates(job_desc, candidate_resumes)
    
    output_str = "## Ranked Candidates (Debiased Similarity)\n\n"
    for candidate in ranked_results:
        output_str += (
            f"**Candidate {candidate['candidate_index']}**\\n"
            f"Original Similarity: {candidate['original_similarity']:.4f}\\n"
            f"Bias Detected Score: {candidate['bias_detected_score']:.4f}\\n"
            f"Debiased Similarity: {candidate['debiased_similarity']:.4f}\\n"
            f"Candidate Snippet: {candidate['candidate_text'][:150]}...\\n\\n"
        )
    return output_str

iface = gr.Interface(
    fn=gradio_interface_fn,
    inputs=[
        gr.Textbox(lines=5, label="Job Description", placeholder="Enter the job description here..."),
        gr.Textbox(lines=10, label="Candidate Resumes (Separate with --- on a new line)", placeholder="Enter candidate resumes here. Use '---' on a new line to separate multiple resumes.")
    ],
    outputs="markdown",
    title="Inclusive Job Candidate Screening Platform",
    description="This platform ranks job candidates based on their resumes and a job description, attempting to mitigate biases. The bias detection is a simplistic simulation for demonstration purposes."
)

if __name__ == "__main__":
    # To run the Gradio interface:
    # Simply execute this Python script: python main.py
    # It will launch a local server with the UI.
    iface.launch()

    # To run the FastAPI server:
    # You would typically run this in a separate terminal using uvicorn:
    # uvicorn main:app --reload
    # The FastAPI server would then be accessible at http://127.0.0.1:8000/docs
