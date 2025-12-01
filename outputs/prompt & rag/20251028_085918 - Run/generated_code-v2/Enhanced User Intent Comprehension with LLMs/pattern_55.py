import random
import json

class CustomerSupportAssistant:
    def __init__(self, intents_config_path="intents.json", user_data_path="user_data.json"):
        self.intents = self._load_intents(intents_config_path)
        self.user_data = self._load_user_data(user_data_path) # For personalization
        self.current_user_id = None # Simulating a logged-in user

    def _load_intents(self, config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Default intents if file not found
            print(f"Warning: {config_path} not found. Using default intents.")
            return {
                "check_order_status": {"keywords": ["where", "my order", "status", "track", "delivery", "package"], "response_template": "Let me check the status of your order {order_id}."},
                "initiate_return_exchange": {"keywords": ["return", "exchange", "refund", "faulty", "wrong item", "damaged"], "response_template": "I can help you initiate a return or exchange for your order {order_id}."},
                "update_shipping_info": {"keywords": ["change address", "update shipping", "delivery details", "new address"], "response_template": "To update your shipping information, please provide your order ID and new address."},
                "route_to_human_agent": {"keywords": ["talk to human", "agent", "can't find", "help", "speak to someone"], "response_template": "Connecting you to a human agent now. Please hold."}
            }

    def _load_user_data(self, data_path):
        try:
            with open(data_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {data_path} not found. Starting with empty user data.")
            return {}

    def set_user(self, user_id):
        self.current_user_id = user_id
        if user_id not in self.user_data:
            self.user_data[user_id] = {"past_interactions": [], "preferences": {}}

    def _recognize_intent(self, query):
        query = query.lower()
        intent_scores = {intent: 0 for intent in self.intents}

        for intent_name, intent_data in self.intents.items():
            for keyword in intent_data["keywords"]:
                if keyword in query:
                    intent_scores[intent_name] += 1

        # Filter out intents with zero scores
        possible_intents = {k: v for k, v in intent_scores.items() if v > 0}

        if not possible_intents:
            return "unknown", 0 # No intent recognized

        sorted_intents = sorted(possible_intents.items(), key=lambda item: item[1], reverse=True)

        top_intent, top_score = sorted_intents[0]

        # Check for ambiguity (e.g., if top two intents have very close scores)
        if len(sorted_intents) > 1:
            second_intent, second_score = sorted_intents[1]
            # If the top two scores are identical or very close, consider it ambiguous
            if top_score > 0 and (top_score - second_score) < 2:
                return "ambiguous", sorted_intents[:2] # Return top 2 ambiguous intents

        return top_intent, top_score

    def _ask_clarifying_question(self, ambiguous_intents):
        # For simplicity, we'll just pick one of the ambiguous options to ask about
        # In a real system, this would be more sophisticated.
        options = ", or ".join([intent_name.replace("_", " ") for intent_name, _ in ambiguous_intents])
        return f"I'm not sure if you want to {options}? Can you clarify?"

    def _execute_action(self, intent, query):
        # Placeholder for actual API calls or database lookups
        if intent == "check_order_status":
            order_id_match = next((word for word in query.split() if word.isdigit()), "N/A")
            return self.intents[intent]["response_template"].format(order_id=order_id_match)
        elif intent == "initiate_return_exchange":
            order_id_match = next((word for word in query.split() if word.isdigit()), "N/A")
            return self.intents[intent]["response_template"].format(order_id=order_id_match)
        elif intent == "update_shipping_info":
            return self.intents[intent]["response_template"]
        elif intent == "route_to_human_agent":
            return self.intents[intent]["response_template"]
        else:
            return "I apologize, I couldn't perform that action. Can I help with anything else?"

    def _personalize_response(self, response, user_id=None):
        if user_id and user_id in self.user_data:
            user_info = self.user_data[user_id]
            # Example personalization: if user has a preferred name
            if "name" in user_info["preferences"]:
                response = response.replace("you", user_info["preferences"]["name"])
            # More complex personalization based on past interactions would go here
        return response

    def chat(self, user_query):
        if not self.current_user_id:
            self.set_user("guest_user") # Assign a guest ID if not set

        print(f"User: {user_query}")
        self.user_data[self.current_user_id]["past_interactions"].append(user_query)

        intent, data = self._recognize_intent(user_query)
        response = ""

        if intent == "ambiguous":
            response = self._ask_clarifying_question(data)
        elif intent == "unknown":
            response = "I'm sorry, I didn't understand your request. Can you please rephrase it or ask for a human agent?"
        else:
            response = self._execute_action(intent, user_query)

        final_response = self._personalize_response(response, self.current_user_id)
        print(f"Assistant: {final_response}")
        return final_response

# Example Usage:
if __name__ == "__main__":
    # Create dummy intents.json and user_data.json for demonstration
    dummy_intents = {
        "check_order_status": {"keywords": ["where", "my order", "status", "track", "delivery", "package"], "response_template": "Let me check the status of your order {order_id}."},
        "initiate_return_exchange": {"keywords": ["return", "exchange", "refund", "faulty", "wrong item", "damaged"], "response_template": "I can help you initiate a return or exchange for your order {order_id}."},
        "update_shipping_info": {"keywords": ["change address", "update shipping", "delivery details", "new address"], "response_template": "To update your shipping information, please provide your order ID and new address."},
        "route_to_human_agent": {"keywords": ["talk to human", "agent", "can't find", "help", "speak to someone"], "response_template": "Connecting you to a human agent now. Please hold."}
    }
    with open("intents.json", "w") as f:
        json.dump(dummy_intents, f, indent=4)

    dummy_user_data = {
        "user123": {"past_interactions": [], "preferences": {"name": "Alex", "favorite_product": "Smartphone"}},
        "guest_user": {"past_interactions": [], "preferences": {}}
    }
    with open("user_data.json", "w") as f:
        json.dump(dummy_user_data, f, indent=4)

    assistant = CustomerSupportAssistant()
    assistant.set_user("user123")

    assistant.chat("Where is my package 12345?")
    assistant.chat("I want to return a faulty item.")
    assistant.chat("Can I change my delivery address for order 67890?")
    assistant.chat("I need help, connect me to an agent.")
    assistant.chat("What is my current order status?") # Ambiguous if no order ID provided, but here it hits order status.
    assistant.chat("I want to send it back") # Intent: return/exchange
    assistant.chat("I want to return an item and also track my order") # Ambiguous example
    assistant.chat("Just a general question.") # Unknown intent

    assistant.set_user("guest_user")
    assistant.chat("Track my order please.")

    # Clean up dummy files
    # import os
    # os.remove("intents.json")
    # os.remove("user_data.json")
