import re

# --- Simulated Data Storage ---
CUSTOMER_DATABASE = {
    "CUST001": {
        "name": "Alice Smith",
        "plan": "Premium Unlimited",
        "bill_due": "2024-03-15",
        "bill_amount": "$75.50",
        "past_issues": ["slow internet (resolved)", "billing discrepancy (resolved)"],
        "service_address": "123 Main St, Anytown"
    },
    "CUST002": {
        "name": "Bob Johnson",
        "plan": "Basic 100Mbps",
        "bill_due": "2024-03-20",
        "bill_amount": "$50.00",
        "past_issues": [],
        "service_address": "456 Oak Ave, Anytown"
    }
}

CUSTOMER_HISTORY = {
    "CUST001": {
        "interaction_count": 10,
        "preferred_channel": "chat",
        "common_intents": {"check_bill": 5, "technical_support": 3}
    },
    "CUST002": {
        "interaction_count": 3,
        "preferred_channel": "phone",
        "common_intents": {"change_plan": 1, "query_service_availability": 1}
    }
}

# --- NLU Module ---
class NLU:
    def preprocess_text(self, text):
        return text.lower().strip()

    def recognize_intent(self, text):
        # Simplified intent recognition using keyword matching
        if "bill" in text or "invoice" in text or "payment" in text:
            return {"intent": "check_bill", "confidence": 0.9}
        elif "problem" in text or "slow" in text or "no internet" in text or "issue" in text:
            return {"intent": "technical_support", "confidence": 0.85}
        elif "plan" in text or "upgrade" in text or "downgrade" in text or "change service" in text:
            return {"intent": "change_plan", "confidence": 0.9}
        elif "service" in text and ("availability" in text or "status" in text or "coverage" in text):
            return {"intent": "query_service_availability", "confidence": 0.8}
        elif "hello" in text or "hi" in text or "hey" in text:
            return {"intent": "greet", "confidence": 0.95}
        elif "bye" in text or "goodbye" in text:
            return {"intent": "goodbye", "confidence": 0.95}
        return {"intent": "unclear", "confidence": 0.4}

    def extract_entities(self, text, intent):
        entities = {}
        if intent == "check_bill":
            account_match = re.search(r"account number (\d+)", text)
            if account_match: entities["account_number"] = account_match.group(1)
            # Mocking specific bill details if asked
            if "amount" in text: entities["bill_detail"] = "amount"
            if "due" in text: entities["bill_detail"] = "due_date"
        elif intent == "technical_support":
            issue_match = re.search(r"my (.*?) is not working", text)
            if issue_match: entities["issue_type"] = issue_match.group(1)
            if "internet" in text: entities["issue_type"] = "internet"
            if "tv" in text: entities["issue_type"] = "tv"
            if "phone" in text: entities["issue_type"] = "phone"
        elif intent == "change_plan":
            plan_match = re.search(r"to (\w+ plan)", text)
            if plan_match: entities["new_plan"] = plan_match.group(1)
        elif intent == "query_service_availability":
            city_match = re.search(r"in (\w+)", text)
            if city_match: entities["city"] = city_match.group(1)
            else: entities["city"] = "Anytown" # Default for simulation
        return entities

# --- Knowledge Base / Action Execution Module ---
class ActionExecution:
    def get_bill_details(self, customer_id, detail_type=None):
        customer = CUSTOMER_DATABASE.get(customer_id)
        if not customer: return "I cannot find details for that customer ID."

        if detail_type == "amount":
            return f"Your current bill amount is {customer['bill_amount']}."
        elif detail_type == "due_date":
            return f"Your bill is due on {customer['bill_due']}."
        else:
            return f"Your bill amount is {customer['bill_amount']} and is due on {customer['bill_due']}."

    def report_technical_issue(self, customer_id, issue_type):
        customer = CUSTOMER_DATABASE.get(customer_id)
        if not customer: return "I cannot report an issue without a valid customer ID."

        customer["past_issues"].append(f"{issue_type} (reported)")
        return f"I have logged a technical issue regarding your {issue_type}. A technician will contact you shortly, {customer['name']}."

    def update_plan_details(self, customer_id, new_plan):
        customer = CUSTOMER_DATABASE.get(customer_id)
        if not customer: return "I cannot update the plan without a valid customer ID."

        customer["plan"] = new_plan
        return f"Your plan has been updated to {new_plan}, {customer['name']}. A confirmation email will be sent."

    def check_service_status(self, city):
        # Simplified: assume service is generally available in Anytown
        if city.lower() == "anytown":
            return f"Service is currently operational in {city}."
        return f"I need to check service status for {city}. Please hold on."

# --- Dialogue Management Module ---
class DialogueManager:
    def __init__(self):
        self.state = {"current_intent": None, "entities": {}, "awaiting_clarification": False}
        self.customer_id = None # In a real system, this would be authenticated
        self.nlu = NLU()
        self.actions = ActionExecution()

    def personalize_response(self, customer_id, response):
        history = CUSTOMER_HISTORY.get(customer_id)
        if history and history.get("interaction_count", 0) > 5:
            return f"Welcome back! {response}"
        return response

    def resolve_ambiguity(self, nlu_output):
        if nlu_output["confidence"] < 0.6 and not self.state["awaiting_clarification"]:
            self.state["awaiting_clarification"] = True
            return "I\'m not entirely sure what you mean. Could you please rephrase or provide more details?"
        return None

    def generate_response(self, intent, entities, action_result=None):
        response = "I understand. How can I help you with that?"
        if intent == "greet":
            response = "Hello! How can I assist you today?"
        elif intent == "check_bill":
            response = action_result or "I can help you with your bill. What specific information are you looking for?"
        elif intent == "technical_support":
            response = action_result or "Please tell me more about the technical issue you are experiencing."
        elif intent == "change_plan":
            response = action_result or "What plan would you like to change to?"
        elif intent == "query_service_availability":
            response = action_result or "Which city are you interested in checking service availability for?"
        elif intent == "unclear":
            response = "I\'m sorry, I didn\'t understand that. Could you please try again?"
        elif intent == "goodbye":
            response = "Goodbye! Have a great day."

        if self.customer_id:
            return self.personalize_response(self.customer_id, response)
        return response

    def handle_query(self, user_input):
        processed_input = self.nlu.preprocess_text(user_input)
        nlu_output = self.nlu.recognize_intent(processed_input)

        clarification_response = self.resolve_ambiguity(nlu_output)
        if clarification_response:
            return clarification_response

        self.state["current_intent"] = nlu_output["intent"]
        self.state["entities"] = self.nlu.extract_entities(processed_input, self.state["current_intent"])
        self.state["awaiting_clarification"] = False # Reset after successful intent recognition

        action_result = None
        if self.state["current_intent"] == "check_bill" and self.customer_id:
            detail_type = self.state["entities"].get("bill_detail")
            action_result = self.actions.get_bill_details(self.customer_id, detail_type)
        elif self.state["current_intent"] == "technical_support" and self.customer_id:
            issue_type = self.state["entities"].get("issue_type", "unspecified issue")
            action_result = self.actions.report_technical_issue(self.customer_id, issue_type)
        elif self.state["current_intent"] == "change_plan" and self.customer_id:
            new_plan = self.state["entities"].get("new_plan")
            if new_plan:
                action_result = self.actions.update_plan_details(self.customer_id, new_plan)
            else:
                action_result = "What is the new plan you would like to switch to?"
        elif self.state["current_intent"] == "query_service_availability":
            city = self.state["entities"].get("city", "Anytown")
            action_result = self.actions.check_service_status(city)

        return self.generate_response(self.state["current_intent"], self.state["entities"], action_result)

# --- Main Chatbot Application --- 
if __name__ == "__main__":
    chatbot = DialogueManager()
    print("Welcome to Telecom Support! Please enter your customer ID to start (e.g., CUST001, CUST002):")
    customer_id_input = input("You: ")
    if customer_id_input in CUSTOMER_DATABASE:
        chatbot.customer_id = customer_id_input
        print(f"Chatbot: Hello {CUSTOMER_DATABASE[chatbot.customer_id]['name']}! How can I assist you today?")
    else:
        print("Chatbot: Invalid customer ID. Proceeding as a guest. Some personalized features may not be available.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print(chatbot.handle_query("goodbye"))
            break

        response = chatbot.handle_query(user_input)
        print(f"Chatbot: {response}")
