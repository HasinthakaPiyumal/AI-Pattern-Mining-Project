
class ChatbotModel:
    """
    A simplified model simulating LLM-based intent understanding.
    In a real application, this would involve a fine-tuned Large Language Model
    to interpret user queries, classify intent, and generate responses.
    """
    def __init__(self):
        # For demonstration, we use a rule-based approach to simulate intent recognition.
        # In a real scenario, an LLM would be fine-tuned on e-commerce customer support data.
        self.intent_map = {
            "track order": ["track", "where is my order", "delivery status"],
            "return item": ["return", "send back", "damaged item"],
            "change shipping address": ["change address", "update shipping", "wrong address"],
            "product information": ["about product", "details of", "specifications"],
            "greeting": ["hello", "hi", "hey"],
            "goodbye": ["bye", "goodbye", "exit", "quit"]
        }

    def predict_response_and_intent(self, query: str, conversation_history: list) -> tuple:
        """
        Simulates intent prediction and response generation by an LLM.
        """
        query_lower = query.lower()
        predicted_intent = None

        # Simple intent matching based on keywords
        for intent, keywords in self.intent_map.items():
            for keyword in keywords:
                if keyword in query_lower:
                    predicted_intent = intent
                    break
            if predicted_intent: # Found an intent
                break
        
        response = "I'm sorry, I couldn't understand your request. Can you please rephrase or provide more details?"

        if predicted_intent == "greeting":
            response = "Hello! How can I assist you with your e-commerce needs today?"
        elif predicted_intent == "goodbye":
            response = "Goodbye! Have a great day."
        elif predicted_intent == "track order":
            response = "Sure, I can help you track your order. Please provide your order number."
        elif predicted_intent == "return item":
            response = "To initiate a return, please provide your order number and the reason for return."
        elif predicted_intent == "change shipping address":
            response = "I can help update your shipping address. Please provide your order number and the new address."
        elif predicted_intent == "product information":
            response = "I can provide product information. Please tell me the name or ID of the product you are interested in."
        
        # Simulate clarification for ambiguous queries (very basic)
        if not predicted_intent and len(query.split()) < 3 and "what" in query_lower:
             response = "Could you please elaborate on what you are looking for?"
             predicted_intent = "clarification_needed"

        # Basic personalization placeholder: acknowledge previous queries if recent
        if conversation_history and len(conversation_history) > 0:
            last_user_query = conversation_history[-1].get("user", "")
            if "thank you" in query_lower and "track order" in last_user_query.lower():
                response = "You're welcome! Let me know if you need anything else regarding your order."
        
        return response, predicted_intent


class BackendActions:
    """
    Simulates backend API calls or database operations for an e-commerce platform.
    """
    def __init__(self):
        # Dummy data for demonstration
        self.orders = {
            "12345": {"status": "Shipped", "address": "123 Main St", "items": ["Laptop"]},
            "67890": {"status": "Processing", "address": "456 Oak Ave", "items": ["Smartphone"]}
        }
        self.products = {
            "Laptop": {"price": "$1200", "specs": "Intel i7, 16GB RAM, 512GB SSD"},
            "Smartphone": {"price": "$800", "specs": "Snapdragon 8 Gen 2, 8GB RAM, 128GB Storage"}
        }

    def execute_action(self, intent: str, user_id: str, query: str) -> str:
        """
        Executes a simulated backend action based on the identified intent.
        """
        if intent == "track order":
            # Extract order number (simplified for demo)
            import re
            order_match = re.search(r'order number (\d+)', query, re.IGNORECASE)
            if order_match:
                order_id = order_match.group(1)
                order_info = self.orders.get(order_id)
                if order_info:
                    return f"Order {order_id} status: {order_info['status']}. Shipping to: {order_info['address']}."
                else:
                    return f"Order {order_id} not found. Please double-check your order number."
            else:
                return "Please provide your order number to track it."
        
        elif intent == "return item":
            # In a real scenario, this would initiate a return process
            return "Return request initiated. We will send you an email with return instructions shortly."
        
        elif intent == "change shipping address":
            # Simplified: just confirms the intent, not actual update
            return "Your request to change the shipping address has been noted. A representative will contact you to confirm the details."
        
        elif intent == "product information":
            import re
            product_match = re.search(r'about (\w+)', query, re.IGNORECASE)
            if product_match:
                product_name = product_match.group(1).capitalize()
                product_info = self.products.get(product_name)
                if product_info:
                    return f"Details for {product_name}: Price: {product_info['price']}, Specs: {product_info['specs']}."
                else:
                    return f"Sorry, I don't have information on '{product_name}'."
            else:
                return "Please specify the product you are interested in."

        return "Action could not be completed at this moment. Please try again later."


class CustomerSupportChatbot:
    """
    Orchestrates the intelligent customer support chatbot interaction.
    It uses a ChatbotModel for intent understanding and BackendActions for executing tasks.
    """
    def __init__(self):
        self.chatbot_model = ChatbotModel()
        self.backend_actions = BackendActions()
        self.user_sessions = {}

    def run(self):
        print("\n--- Welcome to the E-commerce Customer Support Chatbot! ---")
        print("Type 'exit' or 'quit' to end the conversation.")
        user_id = input("Please enter your user ID to start your session: ")
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {"history": []}
        
        print(f"Chatbot: Hello {user_id}! How can I assist you today?")

        while True:
            user_query = input(f"{user_id}: ")
            if user_query.lower() in ["exit", "quit", "bye"]:
                print("Chatbot: Goodbye! Feel free to reach out again if you need anything.")
                break
            
            # Predict intent and get initial response from the chatbot model
            response, intent = self.chatbot_model.predict_response_and_intent(user_query, self.user_sessions[user_id]["history"])
            
            final_chatbot_response = response
            action_result = ""

            if intent == "clarification_needed":
                print(f"Chatbot: {response}")
            elif intent and intent not in ["greeting", "goodbye", "clarification_needed"]:
                # If a specific action-oriented intent is identified, try to execute the backend action
                action_result = self.backend_actions.execute_action(intent, user_id, user_query)
                if action_result:
                    final_chatbot_response = f"{response}\n{action_result}"
                else:
                    final_chatbot_response = f"{response}\nI encountered an issue trying to complete this action."
            
            # Print the final response to the user
            print(f"Chatbot: {final_chatbot_response}")

            # Update conversation history for personalization/context
            self.user_sessions[user_id]["history"].append({"user": user_query, "chatbot": final_chatbot_response})

if __name__ == "__main__":
    chatbot = CustomerSupportChatbot()
    chatbot.run()
