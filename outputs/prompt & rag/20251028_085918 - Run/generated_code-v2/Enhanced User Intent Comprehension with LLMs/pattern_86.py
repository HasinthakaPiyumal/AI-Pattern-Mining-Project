from transformers import pipeline
import random

class IntentUnderstandingModule:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.candidate_labels = [
            "track_order", "initiate_return", "update_account", 
            "product_inquiry", "billing_issue", "technical_support", 
            "general_feedback", "speak_to_agent"
        ]

    def get_intent(self, query):
        result = self.classifier(query, self.candidate_labels)
        intent = result["labels"][0]
        confidence = result["scores"][0]
        return intent, confidence

class MockToolAPIs:
    def track_order(self, order_id):
        return f"Tracking order {order_id}: It is currently in transit and expected by {{random.choice(['tomorrow', 'end of week'])}}."

    def initiate_return(self, order_id, item):
        return f"Return initiated for item '{item}' from order {order_id}. You will receive an email with instructions."

    def update_account(self, field, new_value):
        return f"Account updated: Your {field} has been changed to '{new_value}'."

    def provide_faq_link(self):
        return "Please visit our FAQ page for common questions: https://example.com/faq"
    
    def route_to_human_agent(self):
        return "Connecting you with a human agent. Please wait a moment."

class DialogueManager:
    def __init__(self, confidence_threshold=0.75):
        self.confidence_threshold = confidence_threshold
        self.clarification_needed = False
        self.last_query = ""

    def evaluate_intent(self, intent, confidence, query):
        if confidence < self.confidence_threshold and not self.clarification_needed:
            self.clarification_needed = True
            self.last_query = query
            return "clarify", "I'm not entirely sure I understood. Could you please rephrase or provide more details?"
        elif self.clarification_needed:
            self.clarification_needed = False
            return "re-evaluate", "Thanks for the additional information! Let me re-evaluate."
        else:
            return "proceed", None

class CustomerSupportAgent:
    def __init__(self):
        self.intent_module = IntentUnderstandingModule()
        self.dialogue_manager = DialogueManager()
        self.tool_apis = MockToolAPIs()
        self.user_history = {}

    def handle_query(self, query):
        intent, confidence = self.intent_module.get_intent(query)

        dialogue_action, clarification_message = self.dialogue_manager.evaluate_intent(intent, confidence, query)

        if dialogue_action == "clarify":
            return clarification_message
        elif dialogue_action == "re-evaluate":
            intent, confidence = self.intent_module.get_intent(self.dialogue_manager.last_query + " " + query)
            return self._execute_action(intent, confidence)
        else:
            return self._execute_action(intent, confidence)

    def _execute_action(self, intent, confidence):
        if intent == "track_order":
            order_id = input("Please provide your order ID: ") 
            return self.tool_apis.track_order(order_id)
        elif intent == "initiate_return":
            order_id = input("Please provide your order ID: ")
            item = input("Which item would you like to return?: ")
            return self.tool_apis.initiate_return(order_id, item)
        elif intent == "update_account":
            field = input("Which field would you like to update (e.g., 'email', 'address')?: ")
            new_value = input(f"What is the new value for your {field}?: ")
            return self.tool_apis.update_account(field, new_value)
        elif intent == "general_feedback" or intent == "product_inquiry":
            return self.tool_apis.provide_faq_link()
        elif intent == "billing_issue" or intent == "technical_support" or intent == "speak_to_agent":
            return self.tool_apis.route_to_human_agent()
        else:
            return f"I'm sorry, I couldn't understand your request ({intent}, confidence: {confidence:.2f}). Please try again or ask to speak to a human agent."

if __name__ == "__main__":
    agent = CustomerSupportAgent()
    print("Hello! How can I assist you today? (Type 'exit' to quit)")

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break
        
        response = agent.handle_query(user_query)
        print(f"Agent: {response}")

    print("Goodbye!")