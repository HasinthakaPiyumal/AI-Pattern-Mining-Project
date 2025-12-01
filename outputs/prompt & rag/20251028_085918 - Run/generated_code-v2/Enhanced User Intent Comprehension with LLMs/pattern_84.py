import re

class SmartCustomerSupportAssistant:
    def __init__(self):
        self.conversation_context = {"history": []}

    def _recognize_intent(self, query):
        query = query.lower()

        if "track order" in query or "where is my order" in query:
            order_id_match = re.search(r"order (\w+)", query)
            order_id = order_id_match.group(1) if order_id_match else None
            if order_id:
                return {"intent": "order_tracking", "entities": {"order_id": order_id}, "confidence": 0.9}
            else:
                return {"intent": "order_tracking", "entities": {}, "confidence": 0.6} # Needs clarification
        elif "return" in query or "send back" in query:
            product_match = re.search(r"return (the|my)?\s*(.*?)(?: for| because|$)", query)
            product_name = product_match.group(2).strip() if product_match else None
            if product_name and product_name not in ["my", "the"]:
                return {"intent": "initiate_return", "entities": {"product_name": product_name}, "confidence": 0.9}
            else:
                return {"intent": "initiate_return", "entities": {}, "confidence": 0.6} # Needs clarification
        elif "account" in query or "my info" in query or "change details" in query:
            return {"intent": "account_query", "entities": {}, "confidence": 0.9}
        elif "product info" in query or "details about" in query:
            product_match = re.search(r"(?:about|for|details of) (the|a)?\s*(.*?)(?:\?|$)", query)
            product_name = product_match.group(2).strip() if product_match else None
            if product_name and product_name not in ["the", "a"]:
                return {"intent": "provide_product_info", "entities": {"product_name": product_name}, "confidence": 0.9}
            else:
                return {"intent": "provide_product_info", "entities": {}, "confidence": 0.6}
        elif "talk to human" in query or "speak to agent" in query or "complex issue" in query:
            return {"intent": "escalate_to_human", "entities": {}, "confidence": 1.0}
        elif "hello" in query or "hi" in query:
            return {"intent": "greet", "entities": {}, "confidence": 1.0}
        elif "thank you" in query or "thanks" in query:
            return {"intent": "thank", "entities": {}, "confidence": 1.0}
        elif "bye" in query or "goodbye" in query:
            return {"intent": "farewell", "entities": {}, "confidence": 1.0}

        return {"intent": "unknown", "entities": {}, "confidence": 0.5}

    def _track_order(self, order_id):
        if order_id == "12345":
            return f"Your order {order_id} is currently out for delivery and expected by 5 PM today."
        elif order_id:
            return f"I cannot find order {order_id}. Please double-check the order ID."
        else:
            return "I need an order ID to track your order. Can you please provide it?"

    def _initiate_return(self, product_name):
        if product_name == "shirt":
            return f"To initiate a return for the {product_name}, please visit our returns portal or provide your order ID."
        elif product_name:
            return f"We can help you with the return of {product_name}. Please provide your order ID."
        else:
            return "What product would you like to return? Please specify the item."

    def _update_account_info(self):
        return "Please visit the 'My Account' section on our website to update your personal information."

    def _provide_product_info(self, product_name):
        if product_name == "laptop":
            return f"The {product_name} is a high-performance model with 16GB RAM and a 512GB SSD. It's currently in stock."
        elif product_name:
            return f"I can provide information about the {product_name}. Can you be more specific about which model or brand you are interested in?"
        else:
            return "What product are you interested in?"

    def _escalate_to_human(self, query, conversation_history):
        return "I am connecting you to a human agent who can assist you further. Please hold."

    def _tool_dispatcher(self, intent, entities):
        if intent == "order_tracking":
            return self._track_order(entities.get("order_id"))
        elif intent == "initiate_return":
            return self._initiate_return(entities.get("product_name"))
        elif intent == "account_query":
            return self._update_account_info()
        elif intent == "provide_product_info":
            return self._provide_product_info(entities.get("product_name"))
        elif intent == "escalate_to_human":
            return self._escalate_to_human(self.conversation_context["history"][-1]["user"], self.conversation_context["history"])
        elif intent == "greet":
            return "Hello! How can I assist you today?"
        elif intent == "thank":
            return "You're welcome! Is there anything else I can help with?"
        elif intent == "farewell":
            return "Goodbye! Have a great day."
        elif intent == "unknown":
            return "I'm not sure how to help with that. Can you please rephrase or be more specific?"
        return "I'm sorry, I didn't understand that. Could you please try again?"

    def _generate_response(self, action_result):
        return action_result

    def chat(self):
        print("Welcome to the Smart Customer Support Assistant! Type 'exit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Assistant: Goodbye!")
                break

            self.conversation_context["history"].append({"user": user_input})
            
            intent_recognition_result = self._recognize_intent(user_input)
            intent = intent_recognition_result["intent"]
            entities = intent_recognition_result["entities"]
            confidence = intent_recognition_result["confidence"]

            response = ""

            if confidence < 0.7 or (intent in ["order_tracking", "initiate_return", "provide_product_info"] and not entities):
                if intent == "order_tracking" and not entities:
                    response = "I can help track your order, but I need the order ID. Can you please provide it?"
                elif intent == "initiate_return" and not entities:
                    response = "To initiate a return, please tell me which product you'd like to return."
                elif intent == "provide_product_info" and not entities:
                    response = "What product are you interested in?"
                else:
                    response = "I'm not entirely sure I understood. Could you clarify if you're trying to {} or something else?".format(intent.replace('_', ' '))
            else:
                action_result = self._tool_dispatcher(intent, entities)
                response = self._generate_response(action_result)

            self.conversation_context["history"].append({"assistant": response})
            print(f"Assistant: {response}")

if __name__ == "__main__":
    assistant = SmartCustomerSupportAssistant()
    assistant.chat()