
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer
import chromadb
import json
from typing import List, Dict, Any

# Ensure consistent language detection results
DetectorFactory.seed = 0

# --- Configuration --- #
# Hugging Face Model (choose a suitable multilingual model, e.g., for translation)
# For a full generative LLM, you might need a more powerful model and potentially quantization/vLLM
# We'll use a translation model to demonstrate cross-lingual capabilities within the prompt.
# For actual conversational generation, a truly multilingual generative LLM would be ideal.
LLM_MODEL_NAME = "Helsinki-NLP/opus-mt-en-es" # Example for English to Spanish translation
FALLBACK_LLM_MODEL_NAME = "Helsinki-NLP/opus-mt-es-en" # Example for Spanish to English translation

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Pydantic Models --- #
class ChatRequest(BaseModel):
    user_query: str
    user_id: str = "anonymous"

class ChatResponse(BaseModel):
    response: str
    language: str
    debug_info: Dict[str, Any] = {}

# --- Knowledge Base Simulation (ChromaDB) --- #
class KnowledgeBase:
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="customer_faqs")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self._populate_knowledge_base()

    def _populate_knowledge_base(self):
        # In-context examples for InCLT Crosslingual Transfer Prompting
        # Each example includes source (English) and target (Spanish) language versions
        # for demonstration. In a real scenario, you'd have more languages and diverse topics.
        faqs = [
            {
                "id": "faq1",
                "en_query": "How do I track my order?",
                "en_response": "You can track your order by logging into your account and visiting the 'My Orders' section. You will find the tracking number there.",
                "es_query": "¿Cómo rastreo mi pedido?",
                "es_response": "Puede rastrear su pedido iniciando sesión en su cuenta y visitando la sección 'Mis Pedidos'. Allí encontrará el número de seguimiento."
            },
            {
                "id": "faq2",
                "en_query": "What is your return policy?",
                "en_response": "Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging.",
                "es_query": "¿Cuál es su política de devoluciones?",
                "es_response": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra, siempre que el artículo no haya sido utilizado y esté en su embalaje original."
            },
            {
                "id": "faq3",
                "en_query": "Do you ship internationally?",
                "en_response": "Yes, we offer international shipping to most countries. Shipping costs and delivery times vary by destination.",
                "es_query": "¿Realizan envíos internacionales?",
                "es_response": "Sí, ofrecemos envíos internacionales a la mayoría de los países. Los costos de envío y los tiempos de entrega varían según el destino."
            },
            {
                "id": "faq4",
                "en_query": "Can I change my shipping address?",
                "en_response": "You can change your shipping address before the order is dispatched. Please contact customer support immediately.",
                "es_query": "¿Puedo cambiar mi dirección de envío?",
                "es_response": "Puede cambiar su dirección de envío antes de que se despache el pedido. Por favor, póngase en contacto con el servicio de atención al cliente de inmediato."
            }
        ]

        documents = []
        metadatas = []
        ids = []
        embeddings = []

        for faq in faqs:
            # Add both English and Spanish queries for retrieval, linking them to the full FAQ data
            documents.append(faq["en_query"])
            metadatas.append({"lang": "en", "original_faq": json.dumps(faq)})
            ids.append(f"{faq['id']}-en")

            documents.append(faq["es_query"])
            metadatas.append({"lang": "es", "original_faq": json.dumps(faq)})
            ids.append(f"{faq['id']}-es")
        
        # Generate embeddings for all documents
        embeddings = self.embedding_model.encode(documents).tolist()

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Knowledge base populated with {len(faqs)*2} entries.")

    def retrieve_examples(self, query: str, source_lang: str, top_k: int = 2) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={"lang": source_lang} # Try to retrieve examples in the source language first
        )

        retrieved_faqs = []
        if results and results['metadatas']:
            for metadata_item in results['metadatas'][0]:
                if metadata_item and 'original_faq' in metadata_item:
                    retrieved_faqs.append(json.loads(metadata_item['original_faq']))
        return retrieved_faqs

# --- Prompt Engineering Module --- #
class PromptEngineer:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "en" # Default to English if detection fails

    def construct_prompt(self, user_query: str, detected_lang: str, relevant_examples: List[Dict[str, Any]]) -> str:
        prompt_parts = []
        prompt_parts.append("You are a helpful customer support chatbot. Answer the user's question based on the provided context. If you cannot find the answer, politely state that you don't have enough information.")
        prompt_parts.append("\n--- In-Context Examples ---")

        for i, example in enumerate(relevant_examples):
            # The core InCLT pattern: include examples in both source and target languages
            # For this example, we assume target language is English for internal reasoning
            # and the user's detected language for direct relevance.
            
            # Example in the detected language
            if f"{detected_lang}_query" in example and f"{detected_lang}_response" in example:
                prompt_parts.append(f"Example {i+1} ({detected_lang.upper()}):")
                prompt_parts.append(f"Query: {example[f'{detected_lang}_query']}")
                prompt_parts.append(f"Answer: {example[f'{detected_lang}_response']}\n")
            
            # Example in a common pivot language (e.g., English), for cross-lingual transfer
            # This helps the LLM understand the intent even if the source language examples are limited.
            if "en_query" in example and "en_response" in example and detected_lang != "en":
                prompt_parts.append(f"Example {i+1} (EN - for transfer):")
                prompt_parts.append(f"Query: {example['en_query']}")
                prompt_parts.append(f"Answer: {example['en_response']}\n")

        prompt_parts.append("-------------------------\n")
        prompt_parts.append(f"User Query ({detected_lang.upper()}): {user_query}")
        prompt_parts.append("Chatbot Answer:")

        return "\n".join(prompt_parts)

# --- Multilingual LLM Integration --- #
class LLMService:
    def __init__(self, model_name: str, fallback_model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Load a fallback tokenizer and model for target language translation if primary is src->target_pivot
        # In a real system, you'd use one robust multilingual LLM or a pipeline of models.
        self.fallback_tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
        self.fallback_model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model_name)

    def generate_response(self, prompt: str, target_lang: str) -> str:
        # For this example, `Helsinki-NLP/opus-mt-en-es` translates EN->ES. 
        # If the target_lang is 'es', we can use it directly with the prompt (assuming prompt is in EN).
        # If target_lang is 'en', we need to use a different model or pipeline.
        # This simplification highlights the need for a truly multilingual generative LLM.

        # Let's assume the LLM (like a T5 or M2M100) can handle the prompt and generate response in target_lang
        # For Helsinki-NLP, it's a direct translation model.
        # So, we'll demonstrate a simplified response generation based on the prompt structure.
        
        # In a real scenario with a generative LLM:
        # inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        # outputs = self.model.generate(**inputs, max_new_tokens=150)
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # For demonstration with translation models, we will extract the expected answer or simulate it.
        # A more complex generative LLM would directly use the in-context examples.

        # Simple heuristic to simulate response based on prompt for this demo
        if "Answer:" in prompt:
            # Try to find the most relevant example's answer in the target language
            for line in reversed(prompt.split('\n')):
                if line.startswith(f"Answer:"):
                    # This is very simplistic and assumes the *last* example's answer is relevant
                    # A real LLM would use the entire context to generate a new answer.
                    simulated_answer = line.replace("Answer: ", "").strip()
                    
                    # If the LLM is a translation model and the target_lang is different from the 
                    # language of the simulated_answer, we would translate it.
                    # For now, let's assume the simulated answer is already in the detected_lang 
                    # or a suitable one if the LLM was truly generative.
                    return simulated_answer
        
        # Fallback if no specific answer found in examples (e.g., query is outside KB)
        # For a translation model, we can try to translate the user query to a default response.
        # This part needs a proper generative LLM for real 