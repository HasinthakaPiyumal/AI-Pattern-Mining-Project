import sys

try:
    from transformers import pipeline
except ImportError:
    print("Error: The 'transformers' library is not installed.", file=sys.stderr)
    print("Please install it using: pip install transformers", file=sys.stderr)
    sys.exit(1)


class IntentRecognizer:
    def __init__(self):
        # Using a zero-shot classification pipeline for intent recognition
        # For a production system, a fine-tuned model or a larger foundation model might be preferred
        # For demonstration, 'facebook/bart-large-mnli' offers a good balance.
        print("Initializing IntentRecognizer with 'facebook/bart-large-mnli' model. This may take a moment...")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        print("IntentRecognizer initialized.")
        self.candidate_labels = [
            "product inquiry",
            "order status",
            "return request",
            "technical support",
            "account management",
            "general greeting",
            "escalate to human"  # Added for direct escalation
        ]
        self.intent_threshold = 0.7  # Confidence threshold for clear intent

    def recognize_intent(self, text):
        # Perform zero-shot classification
        result = self.classifier(text, self.candidate_labels)
        # Sort by score in descending order
        sorted_results = sorted(zip(result['labels'], result['scores']), key=lambda x: x[1], reverse=True)

        # Get the top intent and its confidence
        top_intent_label = sorted_results[0][0]
        top_intent_score = sorted_results[0][1]

        if top_intent_score >= self.intent_threshold:
            return {"intent": top_intent_label, "confidence": top_intent_score, "is_ambiguous": False}
        else:
            return {"intent": None, "confidence": top_intent_score, "is_ambiguous": True, "alternative_intents": sorted_results[0:3]}  # Provide top 3 alternatives for clarification


class ActionDispatcher:
    def __init__(self):
        self.actions = {
            "product inquiry": self._handle_product_inquiry,
            "order status": self._handle_order_status,
            "return request": self._handle_return_request,
            "technical support": self._handle_technical_support,
            "account management": self._handle_account_management,
            "general greeting": self._handle_greeting,
            "escalate to human": self._handle_escalation,
            "default": self._handle_default
        }

    def _handle_product_inquiry(self, query):
        return f"Sure, I can help with product inquiries. What product are you interested in or what specific information do you need about a product?"

    def _handle_order_status(self, query):
        return f"To check your order status, please provide your order number. (Simulating action for: '{query}')"

    def _handle_return_request(self, query):
        return f"I can assist with return requests. Could you please provide your order number and the reason for the return?"

    def _handle_technical_support(self, query):
        return f"For technical support, please describe your issue in detail. (Simulating action for: '{query}')"

    def _handle_account_management(self, query):
        return f"I can help with account management. Are you looking to update your profile, change password, or something else?"

    def _handle_greeting(self, query):
        return f"Hello! How can I assist you today?"

    def _handle_escalation(self, query):
        return f"I understand you need further assistance. I'm connecting you to a human agent now. Please hold. (Your initial query: '{query}')"

    def _handle_default(self, query):
        return f"I'm not sure how to directly answer your request: '{query}'. Can you rephrase or provide more details?"

    def dispatch(self, intent, query):
        action_func = self.actions.get(intent, self.actions["default"])
        return action_func(query)


class PersonalizationManager:
    def __init__(self):
        # In a real system, this would be backed by a persistent database per user.
        self.user_profiles = {}  # Stores user_id -> {'interaction_history': [], 'common_intents': []}

    def update_user_profile(self, user_id, intent, query):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {'interaction_history': [], 'common_intents': []}

        # Simple personalization: store interaction history and common intents
        self.user_profiles[user_id]['interaction_history'].append({"query": query, "intent": intent})
        if intent and intent not in self.user_profiles[user_id]['common_intents']:
            self.user_profiles[user_id]['common_intents'].append(intent)

    def get_user_context(self, user_id):
        # In a more advanced system, this would feed context back into the intent recognition.
        return self.user_profiles.get(user_id, {})


class SmartChatbot:
    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.action_dispatcher = ActionDispatcher()
        self.personalization_manager = PersonalizationManager()
        self.current_user_id = "guest_user_123"  # A placeholder user ID for demonstration

    def run(self):
        print("Welcome to the Smart Customer Support Chatbot! Type 'exit' to quit.")

        while True:
            user_input = input("\nYou > ")
            if user_input.lower() == 'exit':
                break

            # Always try to recognize intent first
            intent_result = self.intent_recognizer.recognize_intent(user_input)
            intent = intent_result["intent"]
            is_ambiguous = intent_result["is_ambiguous"]

            if is_ambiguous:
                print("\nChatbot > I'm not entirely sure I understand. Did you mean:")
                for label, score in intent_result["alternative_intents"]:
                    print(f"          - {label} (confidence: {score:.2f})")
                print("          Please clarify your request, choose an option, or type 'escalate' to connect with a human.")

                clarification_input = input("Your clarification > ")

                if clarification_input.lower() == 'escalate':
                    print(f"Chatbot > {self.action_dispatcher.dispatch('escalate to human', user_input)}")
                    self.personalization_manager.update_user_profile(self.current_user_id, "escalate to human", user_input)
                elif clarification_input.lower() in self.intent_recognizer.candidate_labels:
                    # User clarified with a valid intent
                    clarified_intent = clarification_input.lower()
                    print(f"Chatbot > Okay, processing your request as '{clarified_intent}'.")
                    response = self.action_dispatcher.dispatch(clarified_intent, user_input)
                    print(f"Chatbot > {response}")
                    self.personalization_manager.update_user_profile(self.current_user_id, clarified_intent, user_input)
                else:
                    # Clarification is still unclear, fallback to default
                    print(f"Chatbot > {self.action_dispatcher.dispatch('default', user_input)}")
                    self.personalization_manager.update_user_profile(self.current_user_id, None, user_input)  # Log None for unresolved intent
            else:
                # Intent is clear, dispatch action directly
                print(f"Chatbot > Detected intent: {intent}")
                response = self.action_dispatcher.dispatch(intent, user_input)
                print(f"Chatbot > {response}")
                self.personalization_manager.update_user_profile(self.current_user_id, intent, user_input)


if __name__ == "__main__":
    chatbot = SmartChatbot()
    chatbot.run()