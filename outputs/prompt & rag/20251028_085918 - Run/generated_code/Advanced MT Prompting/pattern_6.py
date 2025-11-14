
import os
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import langdetect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# --- Configuration and Mock Data --- #

# Placeholder for language models. In a real scenario, these would be loaded once or from a service.
# For demonstration, we'll use simplified pipelines or placeholders.
# NLLB (facebook/nllb-200-distilled-600M) is large, so we'll use a smaller general-purpose model for demonstration
# or a simple pass-through with langdetect.

# Using a small English-French model for demonstration of translation. 
# For full multi-lingual, a larger model like NLLB would be used.
translator_en_fr = None # Will be initialized if needed
translator_fr_en = None # Will be initialized if needed

# Sentence Transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Mock Knowledge Base (FAQs, product info)
knowledge_base_articles = [
    {"id": 1, "text": "Our shipping policy states that orders are delivered within 5-7 business days for standard shipping.", "language": "en"},
    {"id": 2, "text": "To initiate a return, please visit your order history and click 'Return Item'.", "language": "en"},
    {"id": 3, "text": "Le prix de cet article est de 49,99 euros. Nous offrons une garantie d'un an.", "language": "fr"},
    {"id": 4, "text": "How can I track my order? You can track your order by logging into your account and visiting the 'My Orders' section.", "language": "en"},
    {"id": 5, "text": "What are your customer service hours? Our customer service is available Monday to Friday, 9 AM to 5 PM local time.", "language": "en"},
]

# Simulate vector embeddings for the knowledge base
knowledge_base_embeddings = embedding_model.encode([article["text"] for article in knowledge_base_articles])

# Mock Customer Database
customer_db = {
    "user123": {"name": "Alice Smith", "orders": ["ORD789", "ORD456"], "preferred_language": "en"},
    "user456": {"name": "Bob Johnson", "orders": ["ORD101"], "preferred_language": "fr"},
}

# Mock CRM for human handoff
crm_tickets = []

# --- FastAPI App Setup --- #

app = FastAPI(
    title="Multi-Lingual Customer Support Chatbot",
    description="A chatbot leveraging contextual understanding and iterative refinement for global e-commerce."
)

# --- Pydantic Models --- #

class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: str = "default_session"

class ChatResponse(BaseModel):
    response: str
    language: str
    intent: str = "unknown"
    entities: Dict[str, Any] = {}
    clarification_needed: bool = False
    human_handoff_suggested: bool = False
    debug_info: Dict[str, Any] = {}

class FeedbackRequest(BaseModel):
    session_id: str
    user_id: str
    message: str
    bot_response: str
    is_helpful: bool
    comments: str = None

# --- Core Chatbot Modules (Simplified/Mocked) --- #

class LanguageDetector:
    def detect(self, text: str) -> str:
        try:
            # langdetect can raise an error for very short or ambiguous texts
            return langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            return "en" # Default to English if detection fails

class MachineTranslator:
    def __init__(self, target_lang: str = "en"):
        self.target_lang = target_lang
        # Initialize a conceptual translator. In a real app, you'd load a specific NLLB model
        # or use a cloud translation API.
        # For this example, we'll use a placeholder and mock the translation.
        try:
            # Using a smaller model for demonstration, like 'Helsinki-NLP/opus-mt-en-fr'
            # For true multi-lingual, a larger NLLB model would be required.
            self.tokenizer_en_fr = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            self.model_en_fr = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            self.translator_en_fr = pipeline("translation", model=self.model_en_fr, tokenizer=self.tokenizer_en_fr)

            self.tokenizer_fr_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            self.model_fr_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            self.translator_fr_en = pipeline("translation", model=self.model_fr_en, tokenizer=self.tokenizer_fr_en)

        except Exception as e:
            print(f"Warning: Could not load translation models. Falling back to mock translation. Error: {e}")
            self.translator_en_fr = None
            self.translator_fr_en = None

    def translate_to_processing_lang(self, text: str, source_lang: str) -> str:
        if source_lang == self.target_lang:
            return text
        
        if source_lang == "fr" and self.translator_fr_en:
            return self.translator_fr_en(text)[0]["translation_text"]
        
        # Mock translation if models aren't loaded or language pair not supported by demo model
        print(f"Mock translating '{text}' from {source_lang} to {self.target_lang}")
        return f"[Translated from {source_lang}]: {text}" 

    def translate_from_processing_lang(self, text: str, target_lang: str) -> str:
        if target_lang == self.target_lang:
            return text

        if target_lang == "fr" and self.translator_en_fr:
            return self.translator_en_fr(text)[0]["translation_text"]
        
        # Mock translation
        print(f"Mock translating '{text}' from {self.target_lang} to {target_lang}")
        return f"[Translated to {target_lang}]: {text}"

class NLUProcessor:
    def __init__(self):
        # For demonstration, we use a simple text classification pipeline for intent.
        # In a real system, this would be a fine-tuned model or a more sophisticated LLM prompt.
        try:
            self.intent_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            print(f"Warning: Could not load NLU intent classifier model. Falling back to mock NLU. Error: {e}")
            self.intent_classifier = None

    def process(self, text: str) -> Dict[str, Any]:
        intent = "unknown"
        entities = {}

        if self.intent_classifier:
            # Simplified intent mapping for demonstration
            result = self.intent_classifier(text)[0]
            if result["label"] == "POSITIVE" and result["score"] > 0.9:
                intent = "gratitude_or_positive_feedback"
            elif "track my order" in text.lower() or "where is my order" in text.lower():
                intent = "order_tracking"
            elif "return" in text.lower() or "refund" in text.lower():
                intent = "return_inquiry"
            elif "shipping" in text.lower() or "delivery" in text.lower():
                intent = "shipping_inquiry"
            elif "price" in text.lower() or "cost" in text.lower():
                intent = "product_price_inquiry"
            else:
                intent = "general_inquiry"
        else:
             # Mock NLU based on keywords if model not loaded
            if "track my order" in text.lower():
                intent = "order_tracking"
            elif "return" in text.lower() or "refund" in text.lower():
                intent = "return_inquiry"
            elif "shipping" in text.lower() or "delivery" in text.lower():
                intent = "shipping_inquiry"
            elif "price" in text.lower() or "cost" in text.lower():
                intent = "product_price_inquiry"
            else:
                intent = "general_inquiry"

        # Simple rule-based entity extraction for demonstration
        if "order" in text.lower():
            # Regex to find common order ID patterns (e.g., ORD123, #12345)
            import re
            order_ids = re.findall(r"(?:ORD|#)([A-Z0-9]{3,})", text.upper())
            if order_ids: entities["order_id"] = order_ids[0]

        return {"intent": intent, "entities": entities}

class KnowledgeRetriever:
    def __init__(self, articles: List[Dict[str, Any]], embeddings, model):
        self.articles = articles
        self.embeddings = embeddings
        self.model = model

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(query)
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top_k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        retrieved_info = []
        for i in top_indices:
            # Ensure the similarity is above a certain threshold to be relevant
            if similarities[i] > 0.4: # Adjustable threshold
                retrieved_info.append({"text": self.articles[i]["text"], "similarity": similarities[i]})
        
        return retrieved_info

    def get_customer_info(self, user_id: str) -> Dict[str, Any]:
        return customer_db.get(user_id, {})

class DialogueManager:
    def __init__(self):
        # This would typically integrate with a LangChain agent or a custom LLM orchestration.
        # For demonstration, we'll use a rule-based approach with LLM-like placeholders.
        pass

    def generate_response(self, user_id: str, intent: str, entities: Dict[str, Any], context: List[Dict[str, Any]]) -> str:
        customer_info = customer_db.get(user_id, {})
        customer_name = customer_info.get("name", "customer")

        base_response = f"Hello {customer_name}! How can I assist you today?"

        if intent == "order_tracking":
            order_id = entities.get("order_id")
            if order_id and order_id in customer_info.get("orders", []):
                base_response = f"I can help you with order {order_id}. Looking up its status... (This is a mock response) It is currently in transit."
            elif order_id:
                base_response = f"I couldn't find order {order_id} in your records. Please ensure the order ID is correct."
            else:
                base_response = "Could you please provide your order ID so I can track it for you?"
        elif intent == "return_inquiry":
            base_response = "To initiate a return, please visit your order history and click 'Return Item'. Do you need help finding your order history?"
        elif intent == "shipping_inquiry":
            base_response = "Our standard shipping takes 5-7 business days. Would you like to know more about expedited shipping options?"
        elif intent == "product_price_inquiry":
            base_response = "I can help you with product pricing. Which product are you interested in?"
        elif intent == "gratitude_or_positive_feedback":
            base_response = "You're very welcome! I'm glad I could help."
        
        # Augment with retrieved knowledge
        if context:
            context_text = " ".join([item["text"] for item in context])
            if len(context_text) > 50:
                # Simple LLM-like integration for context
                base_response += f"\nBased on our knowledge base: {context_text[:100]}..."
            else:
                base_response += f"\nBased on our knowledge base: {context_text}"

        return base_response

class IterativeRefinement:
    def __init__(self):
        pass

    def check_for_clarification(self, intent_confidence: float) -> bool:
        # If NLU confidence is low, suggest clarification
        return intent_confidence < 0.6

    def suggest_human_handoff(self, conversation_history: List[Dict[str, Any]], clarification_attempts: int = 0) -> bool:
        # Mock logic: If too many clarification attempts or specific keywords, suggest handoff
        if clarification_attempts >= 2 or any("speak to a human" in msg["message"].lower() for msg in conversation_history):
            return True
        return False

    def log_feedback(self, feedback: FeedbackRequest):
        print(f"Feedback received: Session {feedback.session_id}, Helpful: {feedback.is_helpful}, Comments: {feedback.comments}")
        # In a real system, this would store feedback in a database for model retraining.

# --- Instantiate Modules --- #

lang_detector = LanguageDetector()
machine_translator = MachineTranslator()
nlu_processor = NLUProcessor()
knowledge_retriever = KnowledgeRetriever(knowledge_base_articles, knowledge_base_embeddings, embedding_model)
dialogue_manager = DialogueManager()
iterative_refinement = IterativeRefinement()

# --- Session Management (In-memory for demo) --- #
# In a real application, this would be a persistent store (Redis, database).
active_sessions: Dict[str, List[Dict[str, Any]]] = {}

# --- API Endpoints --- #

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message
    session_id = request.session_id

    debug_info = {
        "original_message": user_message,
        "detected_language": None,
        "translated_message": None,
        "nlu_results": None,
        "retrieved_knowledge": [],
        "final_response_language": None
    }

    # 1. Retrieve session history
    conversation_history = active_sessions.get(session_id, [])

    # 2. Language Detection
    detected_lang = lang_detector.detect(user_message)
    debug_info["detected_language"] = detected_lang
    
    # Determine target response language (user's detected language or preferred from DB)
    customer_pref_lang = customer_db.get(user_id, {}).get("preferred_language", detected_lang)
    response_target_lang = customer_pref_lang if customer_pref_lang != "en" else detected_lang # Prioritize user pref, but fallback to detected

    # 3. Machine Translation (Input) - to processing language (English)
    processing_message = machine_translator.translate_to_processing_lang(user_message, detected_lang)
    debug_info["translated_message"] = processing_message

    # 4. NLU (Intent & Entity Extraction)
    nlu_results = nlu_processor.process(processing_message)
    intent = nlu_results["intent"]
    entities = nlu_results["entities"]
    debug_info["nlu_results"] = nlu_results

    # Mock intent confidence for iterative refinement
    # In a real NLU, this would come directly from the model.
    intent_confidence = 0.9 if intent != "unknown" else 0.4

    # 5. Knowledge Retrieval
    retrieved_knowledge = knowledge_retriever.retrieve(processing_message)
    debug_info["retrieved_knowledge"] = retrieved_knowledge
    customer_specific_info = knowledge_retriever.get_customer_info(user_id)
    if customer_specific_info: # Add customer info to context if available
        retrieved_knowledge.append({"text": f"Customer Info: {customer_specific_info}", "similarity": 1.0})

    # 6. Dialogue Management & Response Generation
    bot_response_processing_lang = dialogue_manager.generate_response(
        user_id, intent, entities, retrieved_knowledge
    )

    # 7. Iterative Refinement
    clarification_needed = iterative_refinement.check_for_clarification(intent_confidence)
    human_handoff_suggested = False
    if clarification_needed:
        bot_response_processing_lang = "I'm not entirely sure I understand. Could you please rephrase or provide more details?"
        # Increment clarification attempts (would be stored in session for a real app)
        # For this demo, let's just assume one attempt before considering handoff
        conversation_history.append({"role": "user", "message": user_message})
        conversation_history.append({"role": "bot", "message": bot_response_processing_lang})
        # Check if handoff is needed after clarification attempt (simplified)
        if iterative_refinement.suggest_human_handoff(conversation_history, clarification_attempts=1):
            human_handoff_suggested = True
            bot_response_processing_lang += " Would you like to speak to a human agent?"
            
    if human_handoff_suggested:
        # Simulate creating a CRM ticket
        crm_tickets.append({"user_id": user_id, "session_id": session_id, "issue": user_message, "history": conversation_history})
        print(f"Human handoff suggested for user {user_id}. Ticket created.")

    # 8. Machine Translation (Output) - back to user's language
    final_bot_response = machine_translator.translate_from_processing_lang(bot_response_processing_lang, response_target_lang)
    debug_info["final_response_language"] = response_target_lang

    # Update session history
    conversation_history.append({"role": "user", "message": user_message, "lang": detected_lang})
    conversation_history.append({"role": "bot", "message": final_bot_response, "lang": response_target_lang})
    active_sessions[session_id] = conversation_history

    return ChatResponse(
        response=final_bot_response,
        language=response_target_lang,
        intent=intent,
        entities=entities,
        clarification_needed=clarification_needed,
        human_handoff_suggested=human_handoff_suggested,
        debug_info=debug_info
    )

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    iterative_refinement.log_feedback(request)
    return {"status": "success", "message": "Feedback received. Thank you!"}

@app.post("/handover")
async def request_handoff(request: ChatRequest):
    user_id = request.user_id
    session_id = request.session_id
    conversation_history = active_sessions.get(session_id, [])

    # Simulate creating a CRM ticket explicitly
    crm_tickets.append({"user_id": user_id, "session_id": session_id, "issue": request.message, "history": conversation_history})
    print(f"Explicit human handoff requested by user {user_id}. Ticket created.")

    return {"status": "success", "message": "Connecting you with a human agent. Please wait, an agent will be with you shortly."}


# --- Example of running the app (for local development) ---
# To run this file:
# 1. pip install "fastapi[all]" langdetect transformers sentence-transformers scikit-learn
# 2. uvicorn chatbot_app:app --reload --port 8000
# 3. Access Swagger UI at http://127.0.0.1:8000/docs

# Example usage with curl:
# curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "user123", "message": "Where is my order?"}'
# curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "user456", "message": "Quel est le prix de cet article ?"}'
