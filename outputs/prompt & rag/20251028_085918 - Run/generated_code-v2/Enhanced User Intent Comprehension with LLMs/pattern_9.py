import re

class NLUModule:
    def __init__(self):
        self.intent_keywords = {
            "reset_password": ["reset password", "forgot password", "can't log in"],
            "check_order_status": ["order status", "where is my order", "track my delivery"],
            "product_inquiry": ["about product", "product details", "specifications"],
            "technical_support": ["technical issue", "troubleshoot", "error message"],
            "billing_issue": ["bill", "invoice", "charge", "payment"],
            "general_greeting": ["hello", "hi", "hey"],
            "goodbye": ["bye", "goodbye", "see you"],
        }
        self.entity_patterns = {
            "order_number": r"#?(\d{6,})",
            "product_name": r"(?:about|for|regarding)\s+(.*?)(?:\s+details|\s+specifications|\s+price|\s+issue|$)",
            "issue_type": r"(?:technical issue with|problem with|error in)\s+(.*)"
        }

    def predict_intent_and_entities(self, text):
        text_lower = text.lower()
        predicted_intent = "unknown_intent"
        extracted_entities = {}

        # Simple intent classification based on keywords
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    predicted_intent = intent
                    break
            if predicted_intent != "unknown_intent":
                break

        # Simple entity extraction based on regex patterns
        for entity_type, pattern in self.entity_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                if entity_type == "product_name" and len(match.groups()) > 0:
                    # Heuristic to get a reasonable product name
                    product_name = match.group(1).strip()
                    if not product_name.startswith(("the ", "a ", "an ")) and len(product_name.split()) <= 4:
                        extracted_entities[entity_type] = product_name
                    elif len(match.groups()) > 1 and match.group(2):
                        extracted_entities[entity_type] = match.group(2).strip()
                    elif len(match.groups()) > 0:
                         extracted_entities[entity_type] = match.group(1).strip()

                else:
                    extracted_entities[entity_type] = match.group(1) if match.group(1) else match.group(0)


        return {"intent": predicted_intent, "entities": extracted_entities}

class ActionModule:
    def execute_action(self, intent, entities):
        if intent == "reset_password":
            return "I can help you reset your password. Please visit our 'Forgot Password' page: [Link to Password Reset]"
        elif intent == "check_order_status":
            order_number = entities.get("order_number")
            if order_number:
                # Simulate API call to order system
                return f"Checking status for order {order_number}... Your order is currently 'In Transit'."
            else:
                return "Please provide your order number so I can check its status."
        elif intent == "product_inquiry":
            product_name = entities.get("product_name")
            if product_name:
                # Simulate product database lookup
                return f"Could you tell me more about what you'd like to know about the {product_name}?"
            else:
                return "I can help with product inquiries. Which product are you interested in?"
        elif intent == "technical_support":
            issue = entities.get("issue_type", "your technical issue")
            return f"I understand you're facing a {issue}. Please describe your problem in more detail, and I can connect you to a technical expert if needed."
        elif intent == "billing_issue":
            return "For billing concerns, I can direct you to our billing department. Would you like me to open a support ticket?"
        elif intent == "general_greeting":
            return "Hello! How can I assist you today?"
        elif intent == "goodbye":
            return "Goodbye! Have a great day."
        else:
            return "I'm not sure how to help with that. Could you please rephrase or provide more details?"

class DialogueManager:
    def __init__(self):
        self.nlu = NLUModule()
        self.action_module = ActionModule()
        self.conversation_history = []
        self.current_context = {}

    def respond(self, user_query):
        self.conversation_history.append({"user": user_query})

        nlu_result = self.nlu.predict_intent_and_entities(user_query)
        intent = nlu_result["intent"]
        entities = nlu_result["entities"]

        # Update context with new entities
        self.current_context.update(entities)

        response = ""

        # Ambiguity Resolution / Clarification
        if intent == "product_inquiry" and not self.current_context.get("product_name"):
            response = "Which product are you interested in? Please provide the product name."
            self.current_context["awaiting_product_name"] = True
        elif self.current_context.get("awaiting_product_name") and entities.get("product_name"):
            self.current_context["product_name"] = entities["product_name"]
            del self.current_context["awaiting_product_name"]
            response = self.action_module.execute_action("product_inquiry", self.current_context)
        elif intent == "check_order_status" and not self.current_context.get("order_number"):
            response = "Could you please provide your order number?"
            self.current_context["awaiting_order_number"] = True
        elif self.current_context.get("awaiting_order_number") and entities.get("order_number"):
            self.current_context["order_number"] = entities["order_number"]
            del self.current_context["awaiting_order_number"]
            response = self.action_module.execute_action("check_order_status", self.current_context)
        else:
            # Execute action based on current intent and context
            response = self.action_module.execute_action(intent, self.current_context)

        self.conversation_history.append({"bot": response})
        return response

# Main Chatbot Loop
if __name__ == "__main__":
    chatbot = DialogueManager()
    print("Hello! I am your Smart Customer Support Chatbot. How can I help you today? (Type 'exit' to end)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Bot: Goodbye!")
            break
        
        bot_response = chatbot.respond(user_input)
        print(f"Bot: {bot_response}")
