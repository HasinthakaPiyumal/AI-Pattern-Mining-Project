from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np

class SimulatedFoundationModel:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embedding(self, text):
        return self.model.encode(text)

class IntentClassifier:
    def __init__(self, intents, model_embedding):
        self.intents = intents
        self.model_embedding = model_embedding
        self.classifier = LogisticRegression(random_state=42)
        self.X_train = []
        self.y_train = []

    def add_training_data(self, queries, intent):
        for query in queries:
            self.X_train.append(self.model_embedding.get_embedding(query))
            self.y_train.append(intent)

    def train(self):
        if not self.X_train or not self.y_train:
            raise ValueError("No training data provided. Please add data using add_training_data.")
        self.classifier.fit(np.array(self.X_train), np.array(self.y_train))

    def predict_intent(self, query):
        embedding = self.model_embedding.get_embedding(query)
        probabilities = self.classifier.predict_proba([embedding])[0]
        max_prob_idx = np.argmax(probabilities)
        predicted_intent = self.classifier.classes_[max_prob_idx]
        confidence = probabilities[max_prob_idx]
        return predicted_intent, confidence

class BackendServices:
    def track_order(self, order_id):
        if order_id and order_id.isdigit():
            return f"Your order {order_id} is currently in transit and expected to arrive within 2-3 business days."
        return "Please provide a valid order ID to track your order."

    def initiate_return(self, product_name):
        if product_name:
            return f"Initiating return process for {product_name}. A return label has been sent to your email."
        return "Please specify the product you wish to return."

    def product_inquiry(self, product_name):
        if product_name:
            return f"Details for {product_name}: It's a high-quality item with excellent reviews. Would you like to know more?"
        return "Please tell me which product you are interested in."

    def generic_response(self):
        return "I'm sorry, I couldn't understand your request fully. Could you please rephrase or provide more details?"

class ContextTracker:
    def __init__(self):
        self.context = {}

    def set_context(self, key, value):
        self.context[key] = value

    def get_context(self, key):
        return self.context.get(key)

    def clear_context(self):
        self.context = {}

class Chatbot:
    def __init__(self):
        self.fm = SimulatedFoundationModel()
        self.intents = ['Track Order', 'Initiate Return', 'Product Inquiry', 'Generic']
        self.intent_classifier = IntentClassifier(self.intents, self.fm)
        self.backend = BackendServices()
        self.context_tracker = ContextTracker()
        self._setup_training_data()
        self.intent_classifier.train()
        self.confidence_threshold = 0.7

    def _setup_training_data(self):
        # Sample training data for instruction tuning
        self.intent_classifier.add_training_data(
            ["Where is my order?", "Track my package", "Status of my delivery", "When will my item arrive?"],
            "Track Order"
        )
        self.intent_classifier.add_training_data(
            ["I want to send this back", "How do I return something?", "Return an item", "Process a refund"],
            "Initiate Return"
        )
        self.intent_classifier.add_training_data(
            ["Tell me about this product", "Product information", "Details on item X", "Is this available?"],
            "Product Inquiry"
        )
        self.intent_classifier.add_training_data(
            ["Hello", "Hi", "How are you?", "Thanks"],
            "Generic"
        )

    def _resolve_ambiguity(self, predicted_intent, query):
        # Simple ambiguity resolution based on predicted intent
        if predicted_intent == "Track Order" and "order ID" not in query.lower() and not self.context_tracker.get_context("order_id"):
            return "To track your order, I'll need your order ID. Can you please provide it?", None
        elif predicted_intent == "Initiate Return" and "product" not in query.lower() and not self.context_tracker.get_context("product_name_return"):
            return "To initiate a return, please tell me which product you wish to return.", None
        elif predicted_intent == "Product Inquiry" and "product" not in query.lower() and not self.context_tracker.get_context("product_name_inquiry"):
            return "Which product are you interested in?", None
        return None, predicted_intent

    def process_message(self, user_query):
        predicted_intent, confidence = self.intent_classifier.predict_intent(user_query)
        print(f"DEBUG: Predicted Intent: {predicted_intent}, Confidence: {confidence:.2f}")

        if confidence < self.confidence_threshold:
            return self.backend.generic_response()

        clarifying_question, confirmed_intent = self._resolve_ambiguity(predicted_intent, user_query)

        if clarifying_question:
            self.context_tracker.set_context("awaiting_info_for", predicted_intent)
            return clarifying_question

        # If we were awaiting info and now received it
        if self.context_tracker.get_context("awaiting_info_for"):
            original_intent = self.context_tracker.get_context("awaiting_info_for")
            self.context_tracker.clear_context()
            if original_intent == "Track Order":
                order_id = ''.join(filter(str.isdigit, user_query))
                self.context_tracker.set_context("order_id", order_id)
                return self.backend.track_order(order_id)
            elif original_intent == "Initiate Return":
                product_name = user_query # Simple extraction, could be more sophisticated
                self.context_tracker.set_context("product_name_return", product_name)
                return self.backend.initiate_return(product_name)
            elif original_intent == "Product Inquiry":
                product_name = user_query
                self.context_tracker.set_context("product_name_inquiry", product_name)
                return self.backend.product_inquiry(product_name)


        if confirmed_intent == 'Track Order':
            order_id = self.context_tracker.get_context("order_id") or ''.join(filter(str.isdigit, user_query))
            self.context_tracker.set_context("order_id", order_id)
            return self.backend.track_order(order_id)
        elif confirmed_intent == 'Initiate Return':
            product_name = self.context_tracker.get_context("product_name_return") or user_query # Naive extraction
            self.context_tracker.set_context("product_name_return", product_name)
            return self.backend.initiate_return(product_name)
        elif confirmed_intent == 'Product Inquiry':
            product_name = self.context_tracker.get_context("product_name_inquiry") or user_query
            self.context_tracker.set_context("product_name_inquiry", product_name)
            return self.backend.product_inquiry(product_name)
        elif confirmed_intent == 'Generic':
            return "Hello! How can I assist you today with your e-commerce needs?"
        else:
            return self.backend.generic_response()

# --- UI (Simplified Text Interface) ---
if __name__ == "__main__":
    chatbot = Chatbot()
    print("Welcome to the E-commerce Support Chatbot! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        response = chatbot.process_message(user_input)
        print(f"Chatbot: {response}")
