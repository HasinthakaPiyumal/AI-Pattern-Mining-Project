import uuid
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)

class VectorDB:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.embedding_model = EmbeddingModel()

    def add_document(self, doc_id: str, content: str):
        embedding = self.embedding_model.encode([content])[0]
        self.documents.append({"id": doc_id, "content": content})
        self.embeddings.append(embedding)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.embedding_model.encode([query])[0]
        if not self.embeddings:
            return []
        similarities = cosine_similarity([query_embedding], np.array(self.embeddings))[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.documents[i]["content"] for i in top_indices]

class RAGSystem:
    def __init__(self):
        self.vector_db = VectorDB()
        self._populate_knowledge_base()

    def _populate_knowledge_base(self):
        self.vector_db.add_document("doc1", "The return policy allows for full refunds within 30 days of purchase with a valid receipt.")
        self.vector_db.add_document("doc2", "Our shipping options include standard (5-7 business days) and express (1-2 business days).")
        self.vector_db.add_document("doc3", "To reset your password, visit the login page and click 'Forgot Password'. A link will be sent to your registered email.")
        self.vector_db.add_document("doc4", "Customer support is available 24/7 via chat, email, and phone.")

    def retrieve(self, query: str) -> str:
        retrieved_docs = self.vector_db.search(query)
        return "\n".join(retrieved_docs) if retrieved_docs else "No relevant information found."

class KVCacheManager:
    def __init__(self):
        self.cache: Dict[tuple, Dict[str, Any]] = {}

    def get_prefix_kv(self, full_input_ids: torch.Tensor) -> (tuple, Dict[str, Any]) or (None, None):
        if not self.cache:
            return None, None

        longest_prefix_ids = None
        longest_prefix_data = None
        longest_prefix_len = 0

        for cached_prefix_ids_tuple, cached_data in self.cache.items():
            cached_prefix_ids_tensor = torch.tensor(cached_prefix_ids_tuple)
            if len(cached_prefix_ids_tensor) <= len(full_input_ids) and torch.equal(cached_prefix_ids_tensor, full_input_ids[:len(cached_prefix_ids_tensor)]):
                if len(cached_prefix_ids_tensor) > longest_prefix_len:
                    longest_prefix_len = len(cached_prefix_ids_tensor)
                    longest_prefix_ids = cached_prefix_ids_tuple
                    longest_prefix_data = cached_data
        return longest_prefix_ids, longest_prefix_data

    def update_kv_cache(self, input_ids: torch.Tensor, past_key_values: Any, attention_mask_length: int):
        # Ensure input_ids is a 1D tensor for the key tuple conversion
        self.cache[tuple(input_ids.tolist())] = {
            "past_key_values": past_key_values,
            "attention_mask_length": attention_mask_length
        }

class LLMService:
    def __init__(self):
        self.model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.kv_cache_manager = KVCacheManager()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def generate_response(self, full_prompt: str) -> str:
        full_input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")
        
        cached_prefix_ids_tuple, cached_data = self.kv_cache_manager.get_prefix_kv(full_input_ids[0])

        input_for_generate = full_input_ids
        past_key_values_for_generate = None
        attention_mask_for_generate = torch.ones(full_input_ids.shape)

        if cached_prefix_ids_tuple:
            prefix_len = len(cached_prefix_ids_tuple)
            input_for_generate = full_input_ids[:, prefix_len:]
            past_key_values_for_generate = cached_data["past_key_values"]
            
            cached_attn_len = cached_data["attention_mask_length"]
            attention_mask_for_generate = torch.cat([
                torch.ones(1, cached_attn_len),
                torch.ones(1, input_for_generate.shape[1])
            ], dim=-1)

            print(f"DEBUG: KV Cache HIT. Reusing cache for prefix length: {prefix_len}")
        else:
            print("DEBUG: KV Cache MISS. Processing full prompt.")
            
        with torch.no_grad():
            output_sequences = self.model.generate(
                input_for_generate,
                attention_mask=attention_mask_for_generate,
                past_key_values=past_key_values_for_generate,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )
        
        if not cached_prefix_ids_tuple:
            with torch.no_grad():
                outputs = self.model(full_input_ids, use_cache=True)
                self.kv_cache_manager.update_kv_cache(
                    full_input_ids[0],
                    outputs.past_key_values,
                    full_input_ids.shape[1]
                )
        
        input_token_len_for_decode = input_for_generate.shape[1]
        
        decoded_generated_part = self.tokenizer.decode(output_sequences[0, input_token_len_for_decode:], skip_special_tokens=True)
        
        return decoded_generated_part.strip()

conversation_history: Dict[str, List[Dict[str, str]]] = {}

app = FastAPI(title="Customer Support Co-pilot with KV Cache")
llm_service = LLMService()
rag_system = RAGSystem()

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    full_conversation: List[Dict[str, str]]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message

    if session_id not in conversation_history:
        conversation_history[session_id] = []

    conversation_history[session_id].append({"role": "user", "content": user_message})

    full_conversation_prompt = "The following is a conversation with a customer support agent. The agent is helpful, polite, and provides clear information.\n\n"
    for turn in conversation_history[session_id]:
        full_conversation_prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"
    
    rag_context = ""
    if any(keyword in user_message.lower() for keyword in ["policy", "shipping", "password", "support"]):
        retrieved_info = rag_system.retrieve(user_message)
        if retrieved_info != "No relevant information found.":
            rag_context = f"Retrieved Information: {retrieved_info}\n\n"
            print(f"DEBUG: RAG triggered. Retrieved: {retrieved_info[:50]}...")
    
    llm_input_prompt = f"{full_conversation_prompt}{rag_context}Agent:"

    agent_response = llm_service.generate_response(llm_input_prompt)
    
    conversation_history[session_id].append({"role": "agent", "content": agent_response})

    return ChatResponse(
        session_id=session_id,
        response=agent_response,
        full_conversation=conversation_history[session_id]
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)