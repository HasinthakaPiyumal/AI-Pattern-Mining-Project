from transformers import pipeline
import enum

class Intent(enum.Enum):
    ORDER_STATUS = "order_status"
    RETURN_ITEM = "return_item"
    PRODUCT_INFORMATION = "product_information"
    CONNECT_AGENT = "connect_agent"
    GREETING = "greeting"
    FAREWELL = "farewell"
    UNKNOWN = "unknown"

class EcommerceChatbot:
    def __init__(self):
        self.intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.candidate_labels = [intent.value for intent in Intent if intent != Intent.UNKNOWN]
        self.user_sessions = {}

    def _get_user_session(self, user_id):
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "history": [],
                "name": "Customer"
            }
        return self.user_sessions[user_id]

    def recognize_intent(self, query: str) -> Intent:
        if not query.strip():
            return Intent.UNKNOWN

        # Use a simple keyword-based approach for common greetings/farewells first
        query_lower = query.lower()
        if any(greet in query_lower for greet in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return Intent.GREETING
        if any(bye in query_lower for bye in ["bye", "goodbye", "see you", "thanks bye"]):
            return Intent.FAREWELL

        # Use zero-shot classification for other intents
        results = self.intent_classifier(query, self.candidate_labels, multi_label=False)
        if results["scores"][0] > 0.7:  # Confidence threshold
            return Intent(results["labels"][0])
        else:
            return Intent.UNKNOWN

    def handle_intent(self, user_id: str, intent: Intent, query: str) -> str:
        session = self._get_user_session(user_id)
        session["history"].append({"query": query, "intent": intent.value})

        response = "I'm sorry, I couldn't understand your request. Can you please rephrase it or provide more details?"

        if intent == Intent.GREETING:
            response = f"Hello {session['name']}! How can I assist you today?"
        elif intent == Intent.FAREWELL:
            response = f"Goodbye {session['name']}! Have a great day!"
        elif intent == Intent.ORDER_STATUS:
            response = "To check your order status, please provide your order number. (This would typically trigger an API call to an order system)"
        elif intent == Intent.RETURN_ITEM:
            response = "I can help you with returns. Please provide your order number and the item you wish to return. (This would initiate a return process flow)"
        elif intent == Intent.PRODUCT_INFORMATION:
            response = f"What product are you interested in, {session['name']}? Please specify the product name or category. (This would typically search a product catalog)"
        elif intent == Intent.CONNECT_AGENT:
            response = "Please wait while I connect you to a customer service agent. Your estimated wait time is 2 minutes. (This would transfer to a human agent)"
        elif intent == Intent.UNKNOWN:
            response = "I'm not sure how to help with that. Could you try asking in a different way or choose from common topics like 'order status', 'return an item', or 'product information'?"
        
        return response

    def chat(self, user_id: str, query: str) -> str:
        intent = self.recognize_intent(query)
        response = self.handle_intent(user_id, intent, query)
        return response

if __name__ == "__main__":
    chatbot = EcommerceChatbot()
    print("Welcome to the E-commerce Customer Support Chatbot! Type 'exit' to end the conversation.")
    user_id = "test_user_123" # In a real app, this would come from user login/session

    while True:
        user_input = input(f"You ({user_id}): ")
        if user_input.lower() == 'exit':
            print("Thank you for chatting with us. Goodbye!")
            break
        
        # Simulate user personalization (e.g., setting a name)
        if "my name is" in user_input.lower() and chatbot.recognize_intent(user_input) == Intent.UNKNOWN:
            name_start_idx = user_input.lower().find("my name is") + len("my name is")
            name = user_input[name_start_idx:].strip().split()[0].capitalize()
            chatbot._get_user_session(user_id)["name"] = name
            print(f"Chatbot: Hello {name}! Nice to meet you.")
            continue

        response = chatbot.chat(user_id, user_input)
        print(f"Chatbot: {response}")
