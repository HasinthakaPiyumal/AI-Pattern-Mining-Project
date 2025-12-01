from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

class IntentRecognizer:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='liblinear'))
        self.intent_clarification_map = {
            "technical_support": "Are you experiencing a complete outage, slow speeds, or issues connecting a specific device?",
            "billing_inquiry": "Is your inquiry about a recent charge, a bill statement, or payment methods?",
            "product_info": "Are you looking for details on features, pricing, or compatibility of a product?"
        }
        self.supported_intents = list(self.intent_clarification_map.keys()) + ["greeting", "thanks", "goodbye"]
        self.trained = False

    def train_classifier(self, training_data, labels):
        embeddings = self.model.encode(training_data)
        self.classifier.fit(embeddings, labels)
        self.trained = True

    def predict_intents(self, query):
        if not self.trained:
            return []
        query_embedding = self.model.encode([query])
        probabilities = self.classifier.predict_proba(query_embedding)[0]
        intent_probabilities = []
        for i, intent in enumerate(self.classifier.classes_):
            intent_probabilities.append((intent, probabilities[i]))
        intent_probabilities.sort(key=lambda x: x[1], reverse=True)
        return intent_probabilities

    def get_clarifying_question(self, intents):
        for intent, _ in intents:
            if intent in self.intent_clarification_map:
                return self.intent_clarification_map[intent]
        return None

class ChatbotCore:
    def __init__(self, intent_recognizer):
        self.intent_recognizer = intent_recognizer
        self.conversation_state = {
            "current_intent": None,
            "clarifying_mode": False,
            "asked_questions": [],
            "last_response": None
        }
        self.CONFIDENCE_THRESHOLD = 0.6
        self.AMBIGUITY_DIFF_THRESHOLD = 0.1

    def process_query(self, user_query):
        if self.conversation_state["clarifying_mode"]:
            response = self._process_clarification(user_query)
            return response

        predicted_intents = self.intent_recognizer.predict_intents(user_query)

        if not predicted_intents:
            return "I am sorry, I am not trained yet. Please train me first."

        top_intent, top_confidence = predicted_intents[0]

        if top_confidence < self.CONFIDENCE_THRESHOLD or \
           (len(predicted_intents) > 1 and (predicted_intents[0][1] - predicted_intents[1][1]) < self.AMBIGUITY_DIFF_THRESHOLD):
            
            clarifying_question = self.intent_recognizer.get_clarifying_question(predicted_intents)
            if clarifying_question and clarifying_question not in self.conversation_state["asked_questions"]:
                self.conversation_state["clarifying_mode"] = True
                self.conversation_state["asked_questions"].append(clarifying_question)
                self.conversation_state["current_intent"] = [intent for intent, _ in predicted_intents[:2]] 
                return clarifying_question
            else:
                return "I am having trouble understanding your request. Could you please rephrase or provide more details?"
        else:
            self.conversation_state["current_intent"] = top_intent
            self.conversation_state["clarifying_mode"] = False
            self.conversation_state["asked_questions"] = []
            action_result = self._map_intent_to_action(top_intent)
            return self._generate_response(action_result)

    def _process_clarification(self, user_response):
        # Simple keyword-based clarification for demonstration
        if "outage" in user_response.lower() or "down" in user_response.lower() or "not working" in user_response.lower():
            confirmed_intent = "technical_support"
        elif "slow" in user_response.lower() or "speed" in user_response.lower():
            confirmed_intent = "technical_support"
        elif "bill" in user_response.lower() or "charge" in user_response.lower() or "payment" in user_response.lower():
            confirmed_intent = "billing_inquiry"
        elif "features" in user_response.lower() or "price" in user_response.lower():
            confirmed_intent = "product_info"
        else:
            self.conversation_state["clarifying_mode"] = False
            self.conversation_state["asked_questions"] = []
            return "Thank you for the clarification. I will try to process your request based on this. If you are still facing issues, please provide more details."

        self.conversation_state["current_intent"] = confirmed_intent
        self.conversation_state["clarifying_mode"] = False
        self.conversation_state["asked_questions"] = []
        action_result = self._map_intent_to_action(confirmed_intent)
        return self._generate_response(action_result)

    def _map_intent_to_action(self, intent):
        if intent == "technical_support":
            return "Initiating network diagnostics and connecting you to a technical agent if needed."
        elif intent == "billing_inquiry":
            return "Accessing your billing history and preparing to provide details or connect you to a billing specialist."
        elif intent == "product_info":
            return "Retrieving information about our products and services to assist you."
        elif intent == "greeting":
            return "Hello! How can I assist you today?"
        elif intent == "thanks":
            return "You're welcome! Is there anything else I can help you with?"
        elif intent == "goodbye":
            return "Goodbye! Have a great day."
        return "I am not sure how to handle this specific request yet."

    def _generate_response(self, action_result):
        return action_result

if __name__ == "__main__":
    # Mock Data for training
    training_queries = [
        "My internet is not working",
        "I have no internet connection",
        "My wifi is down",
        "How much do I owe?",
        "What is my current bill?",
        "Can I get details on my last payment?",
        "Tell me about your new smartphone",
        "What are the features of product X?",
        "Pricing for your premium plan",
        "Hi there",
        "Hello",
        "Thank you",
        "Thanks a lot",
        "Bye",
        "See you later"
    ]
    training_labels = [
        "technical_support",
        "technical_support",
        "technical_support",
        "billing_inquiry",
        "billing_inquiry",
        "billing_inquiry",
        "product_info",
        "product_info",
        "product_info",
        "greeting",
        "greeting",
        "thanks",
        "thanks",
        "goodbye",
        "goodbye"
    ]

    intent_recognizer = IntentRecognizer()
    intent_recognizer.train_classifier(training_queries, training_labels)
    chatbot = ChatbotCore(intent_recognizer)

    print("Chatbot initialized. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = chatbot.process_query(user_input)
        print(f"Bot: {response}")