import random
import time

class IntentUnderstandingModel:
    def __init__(self):
        self.intent_map = {
            "internet isn't working": {"intent": "internet_outage", "confidence": 0.9},
            "my internet is down": {"intent": "internet_outage", "confidence": 0.95},
            "can't connect to internet": {"intent": "internet_outage", "confidence": 0.85},
            "bill is too high": {"intent": "billing_issue", "confidence": 0.9},
            "question about my invoice": {"intent": "billing_issue", "confidence": 0.88},
            "router problems": {"intent": "router_troubleshooting", "confidence": 0.8},
            "help with my router": {"intent": "router_troubleshooting", "confidence": 0.85},
            "speak to a human": {"intent": "human_handover", "confidence": 1.0},
            "i need help": {"intent": "general_inquiry", "confidence": 0.6},
            "what's wrong": {"intent": "general_inquiry", "confidence": 0.55},
        }

    def infer_intent(self, query, user_history=None):
        query_lower = query.lower()
        for phrase, data in self.intent_map.items():
            if phrase in query_lower:
                confidence_boost = 0
                if user_history and data["intent"] in user_history.get("frequent_intents", []):
                    confidence_boost = 0.05
                return {"intent": data["intent"], "confidence": min(1.0, data["confidence"] + confidence_boost)}
        
        if "internet" in query_lower and "slow" in query_lower:
            return {"intent": "internet_speed_issue", "confidence": 0.7}
        if "payment" in query_lower and "failed" in query_lower:
            return {"intent": "billing_issue", "confidence": 0.8}

        return {"intent": "unknown", "confidence": 0.4}

class KnowledgeBase:
    def __init__(self):
        self.articles = {
            "internet_outage": "It seems you're experiencing an internet outage. Please check our service status page at [link] for known issues in your area. If no outage is reported, try restarting your modem and router.",
            "billing_issue": "For billing inquiries, you can view your latest statement and payment history in your account portal at [link]. If you have a specific question, please provide your account number.",
            "router_troubleshooting": "To troubleshoot your router, first ensure all cables are securely connected. Try power cycling your router by unplugging it for 30 seconds and then plugging it back in. Wait a few minutes for it to reconnect.",
            "internet_speed_issue": "If your internet speed is slow, try connecting directly to your modem with an ethernet cable to rule out Wi-Fi issues. Also, ensure no large downloads are running in the background. You can test your speed at [speedtest_link]."
        }

    def get_article(self, intent):
        return self.articles.get(intent, "I don't have specific information on that topic right now. Would you like to speak to a human agent?")

class ExternalAPIs:
    def check_system_status(self):
        time.sleep(1) 
        status = random.choice(["operational", "minor_outage", "major_outage"])
        if status == "operational":
            return "All services are currently operational."
        elif status == "minor_outage":
            return "We are experiencing minor service disruptions in some areas. Our team is working to resolve it."
        else:
            return "We are experiencing a major service outage affecting many customers. We apologize for the inconvenience and are working to restore services as quickly as possible."

    def billing_api_call(self, account_number):
        time.sleep(1) 
        if account_number == "12345":
            return "Your last bill was $75.00 due on October 26, 2023. No outstanding balance."
        else:
            return "Could not find billing information for the provided account number. Please verify your account number."

class ActionMapper:
    def __init__(self, kb, apis):
        self.kb = kb
        self.apis = apis

    def perform_action(self, intent, dialogue_state=None):
        if intent == "internet_outage":
            api_response = self.apis.check_system_status()
            return f"Checking system status... {api_response}\n\n{self.kb.get_article('internet_outage')}"
        elif intent == "billing_issue":
            if dialogue_state and dialogue_state.get("account_number"):
                account_num = dialogue_state["account_number"]
                api_response = self.apis.billing_api_call(account_num)
                return f"Retrieving billing information for account {account_num}... {api_response}"
            else:
                return "To help with your billing inquiry, please provide your account number."
        elif intent == "router_troubleshooting" or intent == "internet_speed_issue":
            return self.kb.get_article(intent)
        elif intent == "human_handover":
            return "Please hold while I connect you to a human agent. I will provide them with a summary of our conversation."
        elif intent == "general_inquiry":
            return "I can help with common issues like internet outages, billing questions, or router troubleshooting. What specifically can I assist you with?"
        else:
            return "I'm not sure how to handle that request. Can you rephrase or tell me more?"

class PersonalizationEngine:
    def __init__(self):
        self.user_profiles = {}

    def update_profile(self, user_id, intent_resolved):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {"frequent_intents": [], "history": []}
        
        if intent_resolved and intent_resolved not in self.user_profiles[user_id]["frequent_intents"]:
            self.user_profiles[user_id]["frequent_intents"].append(intent_resolved)

    def get_profile(self, user_id):
        return self.user_profiles.get(user_id, {})

class DialogueManager:
    def __init__(self):
        self.iue = IntentUnderstandingModel()
        self.kb = KnowledgeBase()
        self.apis = ExternalAPIs()
        self.am = ActionMapper(self.kb, self.apis)
        self.pe = PersonalizationEngine()
        self.dialogue_state = {"user_id": "test_user", "conversation_history": [], "current_intent": None, "account_number": None}

    def process_query(self, query):
        self.dialogue_state["conversation_history"].append(f"User: {query}")
        user_profile = self.pe.get_profile(self.dialogue_state["user_id"])
        
        intent_result = self.iue.infer_intent(query, user_profile)
        inferred_intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        response = ""
        if inferred_intent == "unknown" and confidence < 0.6 and not self.dialogue_state["current_intent"]:
            response = "I'm not quite sure what you mean. Can you provide more details or rephrase your question?"
            self.dialogue_state["current_intent"] = None
        elif inferred_intent == "billing_issue" and "account_number" in query.lower():
            try:
                account_num_str = ''.join(filter(str.isdigit, query))
                if len(account_num_str) >= 5: 
                    self.dialogue_state["account_number"] = account_num_str
                    response = self.am.perform_action(inferred_intent, self.dialogue_state)
                    self.pe.update_profile(self.dialogue_state["user_id"], inferred_intent)
                else:
                    response = "Please provide a valid account number."
            except ValueError:
                response = "I couldn't extract an account number. Please try again."
        elif confidence < 0.7 and inferred_intent != "human_handover":
            response = f"I think you might be asking about '{inferred_intent.replace('_', ' ')}', but I'm not entirely sure. Can you confirm or clarify?"
            self.dialogue_state["current_intent"] = inferred_intent # Tentatively set current intent for follow-up
        else:
            if self.dialogue_state["current_intent"] and inferred_intent == "general_inquiry":
                inferred_intent = self.dialogue_state["current_intent"]
            
            response = self.am.perform_action(inferred_intent, self.dialogue_state)
            if inferred_intent != "unknown" and inferred_intent != "general_inquiry":
                self.pe.update_profile(self.dialogue_state["user_id"], inferred_intent)
            self.dialogue_state["current_intent"] = None
        
        self.dialogue_state["conversation_history"].append(f"Agent: {response}")
        return response

    def get_handover_context(self):
        summary = f"User has been discussing: {', '.join(self.pe.get_profile(self.dialogue_state['user_id']).get('frequent_intents', ['no specific topics']))}.\nConversation History:\n" + "\n".join(self.dialogue_state["conversation_history"])
        return summary

if __name__ == "__main__":
    print("Welcome to the Smart Customer Support Agent. How can I help you today?")
    dm = DialogueManager()

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Agent: Goodbye! Have a great day.")
            break

        response = dm.process_query(user_query)
        print(f"Agent: {response}")

        if "human agent" in response.lower() or "connect you to a human" in response.lower():
            print("--- Handover to Human Agent ---")
            print(dm.get_handover_context())
            print("-------------------------------")
            break