from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ChatRequest(BaseModel):
    query: str

class LLMService:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm_pipeline = pipeline("text-generation", model="distilgpt2") # Using a small local LLM for demonstration
        self.exemplars = [
            {"input": "My order hasn't arrived yet.", "output": "Could you please provide your order number so I can check its status for you?"},
            {"input": "How do I return an item?", "output": "To return an item, please visit our 'Returns' page and follow the instructions. You'll typically need your order number and the reason for the return."},
            {"input": "Can I change my shipping address?", "output": "If your order has not yet shipped, we might be able to update the shipping address. Please provide your order number immediately."},
            {"input": "My product is damaged.", "output": "I'm sorry to hear that. Please provide your order number and a description or photo of the damage, and we'll assist you with a replacement or refund."},
            {"input": "What is your refund policy?", "output": "Our refund policy allows returns within 30 days of purchase for a full refund, provided the item is in its original condition. Please see our 'Refunds' page for full details."}
        ]

    def reorder_exemplars(self, user_query: str, exemplars: list) -> list:
        user_query_embedding = self.embedding_model.encode([user_query])
        exemplar_inputs = [ex["input"] for ex in exemplars]
        exemplar_embeddings = self.embedding_model.encode(exemplar_inputs)

        similarities = cosine_similarity(user_query_embedding, exemplar_embeddings)[0]
        
        exemplar_with_similarity = sorted(zip(similarities, exemplars), key=lambda x: x[0], reverse=True)
        
        return [ex for sim, ex in exemplar_with_similarity]

    def generate_prompt(self, user_query: str, ordered_exemplars: list) -> str:
        prompt_parts = []
        for ex in ordered_exemplars:
            prompt_parts.append(f"Customer: {ex['input']}\nAgent: {ex['output']}")
        
        prompt_parts.append(f"Customer: {user_query}\nAgent:")
        return "\n\n".join(prompt_parts)

    def get_llm_response(self, prompt: str) -> str:
        response = self.llm_pipeline(prompt, max_new_tokens=50, num_return_sequences=1, truncation=True)[0]['generated_text']
        agent_response_start = response.rfind("Agent:")
        if agent_response_start != -1:
            return response[agent_response_start + len("Agent:"):].strip()
        return response.strip()

app = FastAPI()
llm_service = LLMService()

@app.post("/chat")
async def chat(request: ChatRequest):
    ordered_exemplars = llm_service.reorder_exemplars(request.query, llm_service.exemplars)
    prompt = llm_service.generate_prompt(request.query, ordered_exemplars)
    response = llm_service.get_llm_response(prompt)
    return {"response": response}