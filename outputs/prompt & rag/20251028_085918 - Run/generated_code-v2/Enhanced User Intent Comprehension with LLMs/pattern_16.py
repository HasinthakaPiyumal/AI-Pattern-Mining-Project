class NLUModule:
    def __init__(self):
        self.intents = {
            "faq": ["how to", "what is", "tell me about", "faq", "information about"],
            "refund": ["refund", "return", "money back", "cancel order"],
            "agent": ["speak to human", "talk to representative", "customer service", "connect me"],
            "status": ["order status", "where is my order", "delivery status", "track order"]
        }
        self.entities = {
            "order_id": [r"#?(\d{6,})", r"order number is (\d+)"],
            "product_name": [r"product (\w+)", r"item (\w+)"]
        }

    def process_query(self, query):
        query_lower = query.lower()
        detected_intent = "unknown"
        confidence = 0.5
        extracted_entities = {}

        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected_intent = intent
                    confidence = 0.8
                    break
            if detected_intent != "unknown":
                break
        
        if "ambiguous" in query_lower: # Simple ambiguity simulation
            confidence = 0.3
            detected_intent = "ambiguous"

        import re
        for entity_type, patterns in self.entities.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    extracted_entities[entity_type] = match.group(1)

        return {"intent": detected_intent, "entities": extracted_entities, "confidence_score": confidence}

class KnowledgeBase:
    def __init__(self):
        self.faqs = {
            "shipping": "Standard shipping takes 5-7 business days. Expedited options are available.",
            "returns": "You can return most items within 30 days of purchase for a full refund.",
            "account": "To reset your password, go to the login page and click 'Forgot Password'."
        }

    def get_faq(self, topic):
        return self.faqs.get(topic.lower(), "Sorry, I can't find information on that specific topic in our FAQs.")

class ActionExecutionModule:
    def __init__(self, kb):
        self.kb = kb

    def execute_action(self, intent, entities):
        if intent == "faq":
            topic = entities.get("topic", "general") # Assuming topic entity for FAQ
            return self.kb.get_faq(topic)
        elif intent == "refund":
            order_id = entities.get("order_id")
            if order_id:
                return f"Initiating refund for order {order_id}. This may take 3-5 business days to process."
            else:
                return "Please provide an order ID to initiate a refund."
        elif intent == "agent":
            return "Connecting you to a human agent. Please hold."
        elif intent == "status":
            order_id = entities.get("order_id")
            if order_id:
                return f"Checking status for order {order_id}. It is currently in transit."
            else:
                return "Please provide an order ID to check the status."
        else:
            return "I'm sorry, I don't know how to handle that request."

class DialogueManagementModule:
    def __init__(self, nlu, ae):
        self.nlu = nlu
        self.ae = ae
        self.conversation_history = []
        self.user_context = {}
        self.personalization_data = {}

    def _clarify_ambiguity(self):
        return "I'm a bit unsure about your request. Could you please rephrase or provide more details?"

    def _generate_personalized_response(self, response):
        user_name = self.personalization_data.get("name")
        if user_name:
            return f"Hello {user_name}! {response}"
        return response

    def process_user_query(self, query):
        self.conversation_history.append({"role": "user", "text": query})
        
        nlu_result = self.nlu.process_query(query)
        intent = nlu_result["intent"]
        entities = nlu_result["entities"]
        confidence = nlu_result["confidence_score"]

        response = ""

        if confidence < 0.6 or intent == "ambiguous":
            response = self._clarify_ambiguity()
        elif intent == "unknown":
            response = "I'm sorry, I couldn't understand your request. Could you please try again?"
        else:
            # Update user context with extracted entities if any
            self.user_context.update(entities)
            # Merge context with current entities for action execution
            action_entities = {**self.user_context, **entities}
            response = self.ae.execute_action(intent, action_entities)
        
        final_response = self._generate_personalized_response(response)
        self.conversation_history.append({"role": "assistant", "text": final_response})
        return final_response
    
    def set_personalization_data(self, key, value):
        self.personalization_data[key] = value

if __name__ == "__main__":
    kb = KnowledgeBase()
    nlu = NLUModule()
    ae = ActionExecutionModule(kb)
    dm = DialogueManagementModule(nlu, ae)

    print("Welcome to IntelliSupport! How can I help you today? (Type 'exit' to quit)")
    
    # Simulate user setting personalization data
    dm.set_personalization_data("name", "Alice")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("IntelliSupport: Goodbye!")
            break
        
        assistant_response = dm.process_user_query(user_input)
        print(f"IntelliSupport: {assistant_response}")