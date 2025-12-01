
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import torch
from chromadb import Client, Settings
from chromadb.utils import embedding_functions

# --- 1. NLU (Intent Understanding) Module ---
class NLUModule:
    def __init__(self, intents: List[str]):
        self.intents = intents
        # In a real scenario, load a fine-tuned model and tokenizer
        # For demonstration, we'll simulate intent classification
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(intents))
        # Dummy model weights for classification (replace with actual fine-tuned weights)
        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, len(intents))
        torch.nn.init.xavier_uniform_(self.model.classifier.weight)
        self.model.classifier.bias.data.fill_(0.0)
        
        self.intent_map = {intent: i for i, intent in enumerate(intents)}
        self.reverse_intent_map = {i: intent for i, intent in enumerate(intents)}

    def predict_intent(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        
        # Simulate confidence and entity extraction
        highest_prob, predicted_idx = torch.max(probabilities, dim=-1)
        predicted_intent = self.reverse_intent_map[predicted_idx.item()]
        confidence = highest_prob.item()

        entities = self._extract_entities(text)
        
        return {
            "intent": predicted_intent,
            "confidence": confidence,
            "entities": entities,
            "all_intents": [{
                "name": self.reverse_intent_map[i],
                "confidence": probabilities[i].item()
            } for i in range(len(self.intents))]
        }

    def _extract_entities(self, text: str) -> Dict[str, str]:
        extracted = {}
        # Simplified NER: check for keywords
        if "order" in text.lower():
            import re
            order_match = re.search(r'(?:order number|order id|#)\s*(\w+)', text, re.IGNORECASE)
            if order_match: extracted["order_id"] = order_match.group(1)
        if "product" in text.lower() or "item" in text.lower():
            product_keywords = ["phone", "laptop", "headset"]
            for kw in product_keywords:
                if kw in text.lower():
                    extracted["product_name"] = kw
                    break
        return extracted

# --- 3. Knowledge Retrieval Module ---
class KnowledgeRetrievalModule:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = Client(Settings(allow_reset=True))
        try:
            self.chroma_client.reset()
        except: # Handle cases where reset might fail (e.g., first run)
            pass
        
        # Custom embedding function for ChromaDB
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.chroma_client.get_or_create_collection(name="ecommerce_kb", embedding_function=self.ef)

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None):
        self.collection.add(documents=[content], metadatas=[metadata or {}], ids=[doc_id])

    def retrieve_answer(self, query: str, num_results: int = 1) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=num_results,
            include=['documents', 'metadatas', 'distances']
        )
        retrieved_info = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                retrieved_info.append({
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
        return retrieved_info

# --- 4. Response Generation Module ---
class ResponseGenerator:
    def __init__(self):
        # In a real scenario, this could use a generative model (e.g., T5, GPT-2)
        pass

    def generate_response(self, intent: str, entities: Dict[str, str], retrieved_info: List[Dict[str, Any]]) -> str:
        if intent == "Order Status":
            order_id = entities.get("order_id")
            if order_id:
                # Simulate looking up order status
                return f"Checking status for order {order_id}. It is currently being processed and is expected to ship within 2 business days."
            else:
                return "Please provide your order ID so I can check its status."
        elif intent == "Return Policy":
            if retrieved_info:
                return f"Here's our return policy: {retrieved_info[0]['document']}. Do you have any specific questions about it?"
            return "Our general return policy allows returns within 30 days of purchase. Items must be unused and in original packaging."
        elif intent == "Product Inquiry":
            product_name = entities.get("product_name")
            if product_name and retrieved_info:
                return f"About the {product_name}: {retrieved_info[0]['document']}. Is there anything else you'd like to know?"
            elif retrieved_info:
                 return f"Here is some information: {retrieved_info[0]['document']}."
            return "What product are you interested in? I can provide details if you tell me the product name."
        elif intent == "Technical Support":
             if retrieved_info:
                return f"I found this information for technical support: {retrieved_info[0]['document']}. Can I assist you further?"
             return "For technical support, please describe your issue. You can also visit our support page or contact us directly."
        return "I'm not sure how to help with that. Can you please rephrase your request or choose from options like 'Order Status', 'Return Policy', 'Product Inquiry', or 'Technical Support'?"

    def generate_clarifying_question(self, ambiguity_type: str, context: Dict[str, Any]) -> str:
        if ambiguity_type == "multiple_intents":
            top_intents = context.get("top_intents", [])
            intent_names = [i['name'] for i in top_intents if i['name'] != 'unknown'] # Exclude 'unknown' if present
            if len(intent_names) >= 2:
                return f"I detected a few possible intents: '{intent_names[0]}' or '{intent_names[1]}'. Which one are you interested in?"
            return "I'm not entirely sure what you mean. Could you please clarify your request?"
        elif ambiguity_type == "missing_entity":
            if context.get("required_entity") == "order_id":
                return "To help you with your order, please provide your order ID."
            elif context.get("required_entity") == "product_name":
                return "What product are you asking about? Please tell me the product name."
        return "I need a bit more information to help you. Can you elaborate?"

# --- 2. Dialogue Management Module ---
class DialogueManager:
    def __init__(self, nlu_module: NLUModule, kb_module: KnowledgeRetrievalModule, response_generator: ResponseGenerator):
        self.nlu = nlu_module
        self.kb = kb_module
        self.generator = response_generator
        self.conversation_history: List[Dict[str, Any]] = [] # Simple history for current session

    def manage_conversation(self, user_query: str) -> str:
        nlu_result = self.nlu.predict_intent(user_query)
        intent = nlu_result["intent"]
        confidence = nlu_result["confidence"]
        entities = nlu_result["entities"]
        all_intents = nlu_result["all_intents"]

        self.conversation_history.append({"user": user_query, "nlu": nlu_result})

        # Ambiguity Detection Logic
        ambiguity_detected = False
        clarifying_question = None

        # Scenario 1: Low confidence in top intent
        if confidence < 0.7: # Threshold for low confidence
            ambiguity_detected = True
            # Check for multiple intents with similar high confidence
            sorted_intents = sorted(all_intents, key=lambda x: x['confidence'], reverse=True)
            if len(sorted_intents) >= 2 and (sorted_intents[0]['confidence'] - sorted_intents[1]['confidence'] < 0.2): # Small difference
                 clarifying_question = self.generator.generate_clarifying_question(
                     "multiple_intents", {"top_intents": sorted_intents[:2]}
                 )
            else:
                clarifying_question = self.generator.generate_clarifying_question("general_ambiguity", {})
        
        # Scenario 2: Missing critical entities for a clear intent
        required_entities = {
            "Order Status": "order_id",
            "Product Inquiry": "product_name"
        }
        if intent in required_entities and required_entities[intent] not in entities:
            ambiguity_detected = True
            clarifying_question = self.generator.generate_clarifying_question(
                "missing_entity", {"required_entity": required_entities[intent]}
            )
        
        if ambiguity_detected and clarifying_question: 
            return clarifying_question

        # If not ambiguous, proceed to knowledge retrieval and response generation
        retrieved_info = []
        if intent in ["Return Policy", "Product Inquiry", "Technical Support"]:
            # Use a slightly more refined query for retrieval
            retrieval_query = user_query 
            if entities: # Add entities to retrieval query for better context
                retrieval_query += " " + " ".join(entities.values())
            retrieved_info = self.kb.retrieve_answer(retrieval_query)
        
        response = self.generator.generate_response(intent, entities, retrieved_info)
        return response

# --- 5. System Integration (FastAPI App) ---
app = FastAPI()

# Initialize Modules
INTENTS = ["Order Status", "Return Policy", "Product Inquiry", "Technical Support", "Greeting", "Goodbye", "unknown"]
nlu_module = NLUModule(intents=INTENTS)
kb_module = KnowledgeRetrievalModule()
response_generator = ResponseGenerator()
dialogue_manager = DialogueManager(nlu_module, kb_module, response_generator)

# Populate Knowledge Base with Dummy Data
kb_module.add_document("doc1", "Our standard return policy allows returns within 30 days of purchase for most items, provided they are in their original condition and packaging. Some exclusions apply, such as digital goods and personalized items.", metadata={"category": "Return Policy"})
kb_module.add_document("doc2", "For technical support regarding your smartphone, please try restarting your device, checking for software updates, or visiting our troubleshooting guide at example.com/support. If the issue persists, contact our support line at 1-800-TECH.", metadata={"category": "Technical Support", "product": "smartphone"})
kb_module.add_document("doc3", "The XYZ Laptop features an Intel i7 processor, 16GB RAM, and a 512GB SSD. It boasts a 15-inch Full HD display and a 10-hour battery life, making it ideal for productivity and entertainment.", metadata={"category": "Product Details", "product_name": "XYZ Laptop"})
kb_module.add_document("doc4", "The ABC Headset offers noise-cancellation and Bluetooth 5.0 connectivity. It has a 20-hour battery life and comes with a 1-year warranty. Perfect for immersive audio experiences.", metadata={"category": "Product Details", "product_name": "ABC Headset"})

class ChatInput(BaseModel):
    query: str

class ChatOutput(BaseModel):
    response: str

@app.post("/chat", response_model=ChatOutput)
async def chat_with_bot(chat_input: ChatInput):
    user_query = chat_input.query
    bot_response = dialogue_manager.manage_conversation(user_query)
    return ChatOutput(response=bot_response)

if __name__ == "__main__":
    import uvicorn
    # Run with: uvicorn chatbot_service:app --reload
    print("Chatbot service starting. Access at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
