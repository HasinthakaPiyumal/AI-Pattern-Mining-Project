import sqlite3
import json
import random

class UserDB:
    def __init__(self, db_name="user_data.db"):
        self.db_name = db_name
        self._create_table()

    def _create_table(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                history TEXT,
                preferences TEXT
            )
        """)
        conn.commit()
        conn.close()

    def get_user_data(self, user_id):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT history, preferences FROM users WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return json.loads(result[0]), json.loads(result[1])
        return [], {}

    def save_user_data(self, user_id, history, preferences):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        history_json = json.dumps(history)
        preferences_json = json.dumps(preferences)
        cursor.execute("INSERT OR REPLACE INTO users (user_id, history, preferences) VALUES (?, ?, ?)",
                       (user_id, history_json, preferences_json))
        conn.commit()
        conn.close()

class IntentRecognizer:
    def __init__(self):
        # In a real application, this would load a fine-tuned LLM from transformers.
        # For this prototype, we'll use simple keyword matching.
        self.intent_keywords = {
            "order_status": ["order", "status", "where is my", "track my"],
            "return_policy": ["return", "policy", "refund", "exchange"],
            "product_info": ["about", "details", "specifications", "product"],
            "contact_support": ["speak to", "human", "representative", "call me"],
            "greeting": ["hello", "hi", "hey"]
        }
        self.fallback_intent = "unknown"

    def recognize_intent(self, query):
        query_lower = query.lower()
        detected_intents = []

        for intent, keywords in self.intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        if len(detected_intents) == 1:
            return {"intent": detected_intents[0], "confidence": 0.9}
        elif len(detected_intents) > 1:
            # Simulate polysemous query / ambiguity
            return {"intent": "ambiguous", "potential_intents": detected_intents, "confidence": 0.5}
        else:
            return {"intent": self.fallback_intent, "confidence": 0.3}

class ToolRetriever:
    def __init__(self):
        # Simulate an e-commerce backend and a knowledge base
        self.order_db = {"12345": {"status": "shipped", "eta": "2 days"}, "67890": {"status": "processing"}}
        self.faq_kb = {
            "return_policy": "Our return policy allows returns within 30 days of purchase for a full refund. Items must be in original condition.",
            "shipping_info": "Standard shipping takes 5-7 business days. Expedited options are available at checkout."
        }

    def get_order_status(self, order_id):
        return self.order_db.get(order_id, {"status": "Order not found"})

    def get_faq(self, topic):
        return self.faq_kb.get(topic, "I'm sorry, I don't have information on that specific topic yet.")

class DialogueManager:
    def __init__(self, user_db, intent_recognizer, tool_retriever):
        self.user_db = user_db
        self.intent_recognizer = intent_recognizer
        self.tool_retriever = tool_retriever
        self.current_user_id = "guest_user"
        self.user_history = []
        self.user_preferences = {}
        self.clarification_needed = False
        self.potential_intents = []

    def set_user(self, user_id):
        self.current_user_id = user_id
        self.user_history, self.user_preferences = self.user_db.get_user_data(user_id)

    def _generate_response(self, intent, data=None):
        if intent == "greeting":
            return "Hello! How can I assist you today?"
        elif intent == "order_status":
            if data and data.get("status") == "Order not found":
                return "I couldn't find an order with that ID. Could you please double-check?"
            elif data:
                return f"Your order (ID: {data['order_id']}) is currently {data['status']}. ETA: {data.get('eta', 'not available')}."
            else:
                return "Please provide your order ID so I can check its status."
        elif intent == "return_policy":
            return self.tool_retriever.get_faq("return_policy")
        elif intent == "product_info":
            return "What specific product are you interested in? I can help with details if you provide a product name or ID."
        elif intent == "contact_support":
            return "I can connect you with a human representative. Please wait while I transfer you."
        elif intent == "ambiguous":
            return f"It seems like you might be asking about a few things: {', '.join([i.replace('_', ' ') for i in self.potential_intents])}. Could you please clarify?"
        elif intent == "unknown":
            return "I'm sorry, I didn't quite understand that. Could you rephrase or ask about something else?"
        return "Something went wrong. Please try again."

    def handle_query(self, query):
        self.user_history.append({"role": "user", "text": query})
        
        if self.clarification_needed and query.lower() in [i.replace('_', ' ') for i in self.potential_intents]:
            resolved_intent = query.lower().replace(' ', '_')
            self.clarification_needed = False
            self.potential_intents = []
            return self._process_resolved_intent(resolved_intent, query)
        
        intent_result = self.intent_recognizer.recognize_intent(query)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        if intent == "ambiguous":
            self.clarification_needed = True
            self.potential_intents = intent_result["potential_intents"]
            response = self._generate_response(intent)
        elif intent == "unknown" and confidence < 0.4:
            response = self._generate_response(intent)
        else:
            response = self._process_resolved_intent(intent, query)

        self.user_history.append({"role": "bot", "text": response})
        self.user_db.save_user_data(self.current_user_id, self.user_history, self.user_preferences)
        return response

    def _process_resolved_intent(self, intent, query):
        data = None
        if intent == "order_status":
            order_id = ''.join(filter(str.isdigit, query))
            if order_id:
                data = self.tool_retriever.get_order_status(order_id)
                data["order_id"] = order_id
            else:
                data = None # Signal that order ID is missing
        elif intent == "return_policy":
            data = {"info": self.tool_retriever.get_faq("return_policy")}

        return self._generate_response(intent, data)


def main():
    user_db = UserDB()
    intent_recognizer = IntentRecognizer()
    tool_retriever = ToolRetriever()
    dialogue_manager = DialogueManager(user_db, intent_recognizer, tool_retriever)

    print("Welcome to the E-commerce Customer Support Chatbot!")
    user_id = input("Please enter your user ID (or press Enter for guest): ")
    if not user_id:
        user_id = "guest_" + str(random.randint(1000, 9999))
    dialogue_manager.set_user(user_id)
    print(f"Hello, {user_id}! How can I assist you today?")

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye!")
            break

        bot_response = dialogue_manager.handle_query(user_query)
        print(f"Bot: {bot_response}")

if __name__ == "__main__":
    main()