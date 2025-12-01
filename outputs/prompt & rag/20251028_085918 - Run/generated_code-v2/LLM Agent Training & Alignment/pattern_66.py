import uuid
import random
from typing import List, Dict
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import PPOTrainer
from datasets import Dataset
import os

DATABASE_NAME = "feedback.db"

def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            source TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            query_id TEXT NOT NULL,
            preferred_response_id TEXT NOT NULL,
            rejected_response_id TEXT NOT NULL,
            FOREIGN KEY (query_id) REFERENCES responses(id),
            FOREIGN KEY (preferred_response_id) REFERENCES responses(id),
            FOREIGN KEY (rejected_response_id) REFERENCES responses(id)
        )
    """)
    conn.commit()
    conn.close()

def generate_llm_response(query: str) -> str:
    if "shipping" in query.lower():
        return "Our standard shipping takes 3-5 business days. Expedited options are available at checkout."
    elif "return" in query.lower():
        return "You can return any unused item within 30 days for a full refund. Please see our returns policy for details."
    elif "product availability" in query.lower():
        return "Stock levels are updated real-time on each product page. If an item is out of stock, you can sign up for email notifications."
    else:
        return f"Hello! For your query about \'{query}\', I am an AI assistant designed to help with common e-commerce questions. How else can I assist you today?"

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_embedding_dim = 768
        self.fc1 = nn.Linear(self.dummy_embedding_dim * 2, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, query_embedding, response_embedding):
        combined_embedding = torch.cat((query_embedding, response_embedding), dim=1)
        x = self.fc1(combined_embedding)
        x = self.relu(x)
        return self.fc2(x)

def get_text_embedding(text: str) -> torch.Tensor:
    return torch.randn(1, 768)

reward_model = RewardModel()
optimizer_rm = torch.optim.Adam(reward_model.parameters(), lr=1e-5)

def train_reward_model(feedback_data: List[Dict]):
    if not feedback_data:
        return
    for feedback_entry in feedback_data:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT query, response FROM responses WHERE id = ?", (feedback_entry["preferred_response_id"],))
        pref_data = cursor.fetchone()
        cursor.execute("SELECT query, response FROM responses WHERE id = ?", (feedback_entry["rejected_response_id"],))
        rej_data = cursor.fetchone()
        conn.close()

        if not pref_data or not rej_data:
            continue

        query_text = pref_data[0]
        pref_response_text = pref_data[1]
        rej_response_text = rej_data[1]

        query_emb = get_text_embedding(query_text)
        pref_response_emb = get_text_embedding(pref_response_text)
        rej_response_emb = get_text_embedding(rej_response_text)

        pref_score = reward_model(query_emb, pref_response_emb)
        rej_score = reward_model(query_emb, rej_response_emb)

        loss = torch.max(torch.tensor(0.0), rej_score - pref_score + torch.tensor(0.1))

        optimizer_rm.zero_grad()
        loss.backward()
        optimizer_rm.step()

llm_model_for_rlhf = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer_for_rlhf = AutoTokenizer.from_pretrained("gpt2")
tokenizer_for_rlhf.pad_token = tokenizer_for_rlhf.eos_token

def fine_tune_llm_with_rlhf():
    pass

app = FastAPI()

class ChatbotQuery(BaseModel):
    query: str

class FeedbackSubmission(BaseModel):
    query_id: str
    preferred_response_id: str
    rejected_response_id: str

@app.on_event("startup")
async def startup_event():
    init_db()

@app.post("/generate_response")
async def generate_response_api(item: ChatbotQuery):
    llm_output = generate_llm_response(item.query)
    response_id = str(uuid.uuid4())

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO responses (id, query, response, source) VALUES (?, ?, ?, ?)",
        (response_id, item.query, llm_output, "llm")
    )
    conn.commit()
    conn.close()

    return {"query": item.query, "response": llm_output, "response_id": response_id}

@app.post("/submit_feedback")
async def submit_feedback_api(feedback: FeedbackSubmission):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM responses WHERE id = ?", (feedback.preferred_response_id,))
    if cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Preferred response ID not found")
    cursor.execute("SELECT id FROM responses WHERE id = ?", (feedback.rejected_response_id,))
    if cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Rejected response ID not found")
    cursor.execute("SELECT id FROM responses WHERE id = ?", (feedback.query_id,))
    if cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Query ID not found in responses, ensure it's a valid response entry ID.")


    feedback_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO feedback (id, query_id, preferred_response_id, rejected_response_id) VALUES (?, ?, ?, ?)",
        (feedback_id, feedback.query_id, feedback.preferred_response_id, feedback.rejected_response_id)
    )
    conn.commit()
    conn.close()
    return {"message": "Feedback submitted successfully", "feedback_id": feedback_id}

@app.get("/get_feedback")
async def get_feedback():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM feedback")
    feedback_entries = [{
        "id": row[0],
        "query_id": row[1],
        "preferred_response_id": row[2],
        "rejected_response_id": row[3]
    } for row in cursor.fetchall()]
    conn.close()
    return feedback_entries

@app.post("/train_models")
async def trigger_model_training():
    feedback_data = await get_feedback()
    train_reward_model(feedback_data)
    fine_tune_llm_with_rlhf()
    return {"message": "Training routines triggered (Reward Model trained, LLM fine-tuning simulated)."}
