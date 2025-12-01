from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import numpy as np
from loguru import logger

class CustomerProfileManager:
    def __init__(self):
        self.customer_profiles = defaultdict(lambda: {"history": [], "embeddings": [], "faiss_index": None})
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def update_profile(self, customer_id, query, response):
        self.customer_profiles[customer_id]["history"].append({"query": query, "response": response})
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
        self.customer_profiles[customer_id]["embeddings"].append(query_embedding)
        logger.info(f"Updated profile for customer {customer_id}")

    def get_personalized_context(self, customer_id, current_query, top_k=2):
        if not self.customer_profiles[customer_id]["history"]:
            return ""
        
        current_query_embedding = self.embedding_model.encode(current_query, convert_to_tensor=False)
        past_embeddings = np.array(self.customer_profiles[customer_id]["embeddings"])
        
        if past_embeddings.shape[0] == 0:
            return ""

        cosine_scores = util.cos_sim(current_query_embedding, past_embeddings)[0].cpu().numpy()
        top_indices = cosine_scores.argsort()[-top_k:][::-1]

        context = []
        for idx in top_indices:
            if cosine_scores[idx] > 0.6:  # Threshold for relevance
                hist_item = self.customer_profiles[customer_id]["history"][idx]
                context.append(f"Past interaction: Q: {hist_item["query"]}. A: {hist_item["response"]}.")
        return " ".join(context)


class IntentRecognizer:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.intents = [
            "Order Status", "Product Information", "Return/Refund", 
            "Technical Support", "General Inquiry", "Shipping Inquiry"
        ]

    def predict_intent(self, query):
        result = self.classifier(query, self.intents, multi_label=False)
        return result["labels"][0], result["scores"][0]


class DialogueManager:
    def __init__(self):
        pass

    def clarify_query(self, query, intent, confidence):
        if confidence < 0.75:
            logger.warning(f"Low confidence for intent '{intent}'. Initiating clarification.")
            return f"I'm not entirely sure if I understood your request about '{intent}'. Could you please rephrase or provide more details?"
        return None


class ToolOrchestrator:
    def __init__(self):
        pass

    def _get_order_status(self, details):
        logger.info(f"Calling Order Status Tool with details: {details}")
        return f"Your order #12345 is currently being processed and is expected to ship by tomorrow."

    def _get_product_info(self, details):
        logger.info(f"Calling Product Info Tool with details: {details}")
        return f"The 'Quantum Widget' is a high-performance device designed for advanced computational tasks. It costs $99.99."

    def _initiate_return(self, details):
        logger.info(f"Calling Return Initiation Tool with details: {details}")
        return f"To initiate a return for item '{details}', please visit our returns portal or provide your order number."

    def orchestrate_action(self, intent, query, context):
        if intent == "Order Status":
            return self._get_order_status(query)
        elif intent == "Product Information":
            return self._get_product_info(query)
        elif intent == "Return/Refund":
            return self._initiate_return(query)
        elif intent == "Technical Support":
            return "Please describe your technical issue in more detail so I can connect you with a specialist."
        elif intent == "Shipping Inquiry":
            return "I can help with shipping inquiries. What specifically would you like to know about your shipment?"
        else: # General Inquiry or unhandled intents
            return f"I can assist with various queries. Could you tell me more about what you need? For now, I understand you're interested in {intent}."


class HumanHandoff:
    def __init__(self):
        pass

    def handoff(self, customer_id, query, intent, context):
        logger.info(f"Escalating to human agent for customer {customer_id}.")
        return (
            f"I'm unable to fully resolve your request at this moment. "
            f"Connecting you to a human agent who will have the following context: "
            f"Customer ID: {customer_id}, Query: '{query}', Identified Intent: '{intent}'. "
            f"Relevant past interactions: {context}"
        )


class SmartCustomerSupportAssistant:
    def __init__(self):
        self.profile_manager = CustomerProfileManager()
        self.intent_recognizer = IntentRecognizer()
        self.dialogue_manager = DialogueManager()
        self.tool_orchestrator = ToolOrchestrator()
        self.human_handoff = HumanHandoff()
        logger.add("assistant.log", rotation="10 MB")

    def process_query(self, customer_id, query):
        logger.info(f"Processing query from customer {customer_id}: '{query}'")

        # 1. Intent Recognition
        intent, confidence = self.intent_recognizer.predict_intent(query)
        logger.info(f"Identified intent: '{intent}' with confidence: {confidence:.2f}")

        # 2. Dialogue Management for clarification
        clarification_needed = self.dialogue_manager.clarify_query(query, intent, confidence)
        if clarification_needed:
            return clarification_needed

        # 3. Customer Profile & Personalization
        context = self.profile_manager.get_personalized_context(customer_id, query)
        logger.info(f"Personalized context for customer {customer_id}: {context if context else 'None'}")

        # Decide if human handoff is immediately needed due to low confidence on primary intent after clarification attempt (if any)
        if confidence < 0.6: # Lower threshold for handoff after initial clarification attempts
            return self.human_handoff.handoff(customer_id, query, intent, context)

        # 4. Tool Orchestration
        response = self.tool_orchestrator.orchestrate_action(intent, query, context)
        
        # 5. Update profile with current interaction
        self.profile_manager.update_profile(customer_id, query, response)

        return response

if __name__ == "__main__":
    assistant = SmartCustomerSupportAssistant()

    print("--- Scenario 1: Order Status Query ---")
    customer1_id = "user123"
    response = assistant.process_query(customer1_id, "Where is my stuff?")
    print(f"Assistant: {response}")
    print("\n")

    print("--- Scenario 2: Product Information Query ---")
    response = assistant.process_query(customer1_id, "Tell me about the Quantum Widget.")
    print(f"Assistant: {response}")
    print("\n")

    print("--- Scenario 3: Ambiguous Query leading to Clarification ---")
    customer2_id = "user456"
    response = assistant.process_query(customer2_id, "I have an issue with my recent purchase.") # Low confidence expected
    print(f"Assistant: {response}")
    print("\n")

    print("--- Scenario 4: Follow-up on Ambiguous Query (simulating more detail) ---")
    response = assistant.process_query(customer2_id, "I want to return the jacket I bought last week.") # Clearer intent
    print(f"Assistant: {response}")
    print("\n")

    print("--- Scenario 5: Personalized follow-up after a while ---")
    response = assistant.process_query(customer1_id, "What about shipping costs?")
    print(f"Assistant: {response}")
    print("\n")

    print("--- Scenario 6: Handoff due to very low confidence ---")
    customer3_id = "user789"
    response = assistant.process_query(customer3_id, "Do you like cats?")
    print(f"Assistant: {response}")
    print("\n")