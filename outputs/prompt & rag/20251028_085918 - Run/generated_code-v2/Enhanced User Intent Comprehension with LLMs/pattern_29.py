import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

class NLUModule:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        return self.model.encode(text)

class IntentClassifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.intents = []

    def train(self, X_embeddings, y_intents):
        self.intents = list(set(y_intents))
        self.model.fit(X_embeddings, y_intents)

    def predict_intent(self, embedding):
        return self.model.predict(embedding.reshape(1, -1))[0]

class DialogueManager:
    def __init__(self):
        self.responses = {
            "greeting": "Hello! How can I help you today?",
            "order_status": "Please provide your order number so I can check its status.",
            "return_policy": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
            "product_inquiry": "What product are you interested in? I can provide more details.",
            "shipping_info": "Standard shipping takes 3-5 business days. Expedited options are available.",
            "speak_to_agent": "Connecting you to a human agent. Please wait a moment.",
            "thank_you": "You're welcome! Is there anything else I can assist you with?",
            "goodbye": "Goodbye! Have a great day.",
            "fallback": "I'm sorry, I don't understand. Could you please rephrase your query?"
        }

    def get_response(self, intent):
        return self.responses.get(intent, self.responses["fallback"])

class ToolActionRouter:
    def handle_action(self, intent):
        if intent == "order_status":
            return "Initiating order status lookup tool... (placeholder)"
        elif intent == "speak_to_agent":
            return "Routing to human agent... (placeholder)"
        else:
            return None

if __name__ == "__main__":
    # 1. Data Management (Training Data Simulation)
    training_data = [
        ("hi there", "greeting"),
        ("hello", "greeting"),
        ("what's my order status", "order_status"),
        ("where is my package", "order_status"),
        ("how can I return an item", "return_policy"),
        ("what's your return policy", "return_policy"),
        ("tell me about this product", "product_inquiry"),
        ("do you have details on product X", "product_inquiry"),
        ("how long does shipping take", "shipping_info"),
        ("what are the shipping options", "shipping_info"),
        ("I need to talk to someone", "speak_to_agent"),
        ("can I speak to a representative", "speak_to_agent"),
        ("thanks", "thank_you"),
        ("thank you very much", "thank_you"),
        ("bye", "goodbye"),
        ("see you later", "goodbye"),
        ("I don't understand", "fallback"),
        ("what did you say", "fallback")
    ]

    queries = [item[0] for item in training_data]
    intents = [item[1] for item in training_data]

    # Initialize NLU Module
    nlu_module = NLUModule()

    # Generate embeddings for training data
    X_train_embeddings = np.array([nlu_module.get_embedding(q) for q in queries])

    # Initialize and train Intent Classifier
    intent_classifier = IntentClassifier()
    intent_classifier.train(X_train_embeddings, intents)

    # Initialize Dialogue Manager and Tool Action Router
    dialogue_manager = DialogueManager()
    tool_router = ToolActionRouter()

    print("E-commerce Chatbot initialized. Type 'quit' to exit.")

    # User Interface (Basic Command-Line Interface)
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'quit':
            break

        # NLU: Get embedding for user query
        user_embedding = nlu_module.get_embedding(user_query)

        # Intent Classification: Predict intent
        predicted_intent = intent_classifier.predict_intent(user_embedding)
        print(f"[DEBUG] Detected Intent: {predicted_intent}")

        # Tool/Action Routing (if applicable)
        tool_action_response = tool_router.handle_action(predicted_intent)
        if tool_action_response:
            print(f"Bot: {tool_action_response}")
        else:
            # Dialogue Management: Get response
            bot_response = dialogue_manager.get_response(predicted_intent)
            print(f"Bot: {bot_response}")

    print("Chatbot session ended.")