import random
import re

class NLU:
    def __init__(self):
        self.intents = {
            "internet issue": {
                "keywords": ["internet", "wifi", "network", "connection", "slow", "down", "outage"],
                "clarification": [
                    "Are you experiencing a complete outage, slow speeds, or intermittent connection issues?",
                    "Is this for your home internet or mobile data?"
                ]
            },
            "account management": {
                "keywords": ["account", "bill", "billing", "plan", "password", "details", "login", "charges"],
                "clarification": [
                    "Are you looking to update billing information, change your password, or check your service plan?",
                    "Could you specify what kind of help you need with your account?"
                ]
            },
            "order problem": {
                "keywords": ["order", "package", "delivery", "track", "shipment", "missing", "wrong item"],
                "clarification": [
                    "Do you need to track an existing order or report a problem with a delivery?",
                    "Could you provide your order number?"
                ]
            },
            "live agent": {
                "keywords": ["agent", "human", "representative", "talk to someone", "speak to"],
                "clarification": [""]
            }
        }
        self.vague_terms = ["problem", "help", "issue", "something wrong", "trouble", "can't"]

    def classify_intent(self, query):
        query_lower = query.lower()

        if any(keyword in query_lower for keyword in self.intents["live agent"]["keywords"]):
            return {"intent": "live agent", "ambiguous": False}

        detected_scores = {intent: 0 for intent in self.intents if intent != "live agent"}
        

        for intent_name, intent_data in self.intents.items():
            if intent_name == "live agent":
                continue
            for keyword in intent_data["keywords"]:
                if keyword in query_lower:
                    detected_scores[intent_name] += 1
        
        candidate_intents = {intent: score for intent, score in detected_scores.items() if score > 0}

        if not candidate_intents:
            return {"intent": "unknown", "ambiguous": True, "clarification": "I'm not sure how to help with that. Could you please rephrase or tell me more specifically what you need?"}

        max_score = 0
        for score in candidate_intents.values():
            if score > max_score:
                max_score = score
        
        top_intents = [intent for intent, score in candidate_intents.items() if score == max_score]

        if len(top_intents) > 1:
            suggested_clarification_intent = random.choice(top_intents)
            return {
                "intent": "ambiguous",
                "ambiguous": True,
                "clarification": f"It seems you might be asking about a few things. Regarding your {suggested_clarification_intent}, {random.choice(self.intents[suggested_clarification_intent]['clarification'])} Or, could you focus on one specific issue at a time?"
            }
        
        resolved_intent = top_intents[0]

        is_vague = False
        for term in self.vague_terms:
            if term in query_lower:
                is_vague = True
                break
        
        if is_vague:
            return {
                "intent": resolved_intent,
                "ambiguous": True,
                "clarification": f"I understand you have an {resolved_intent}. Could you tell me more specifically about the problem? For example, {random.choice(self.intents[resolved_intent]['clarification'])}"
            }
        
        return {"intent": resolved_intent, "ambiguous": False}


class ToolExecutor:
    def execute_tool(self, intent, entities=None):
        if intent == "internet issue":
            if entities and "outage" in entities:
                return "Checking for known internet outages in your area... There are no reported outages. Please try restarting your router."
            if entities and "slow speeds" in entities:
                return "For slow speeds, please try running a speed test and ensure your router firmware is up to date."
            if entities and "intermittent" in entities:
                return "Intermittent connection issues can be tricky. Let's try resetting your network settings."
            return "Running diagnostics on your internet connection. Please try restarting your router, or for more specific help, tell me if it's an outage, slow speed, or intermittent connection."
        elif intent == "account management":
            if entities and "bill" in entities:
                return "You can view your current bill by logging into your account on our website. Would you like a direct link to the billing section?"
            if entities and "password" in entities:
                return "To reset your password, please visit our password reset page. I can send you the link if you'd like."
            if entities and "plan" in entities:
                return "You can review your current service plan details in your account dashboard. What specifically about your plan would you like to know?"
            return "I can help with account details. What specific information are you looking for (e.g., billing, password reset, plan details)?"
        elif intent == "order problem":
            if entities and "order_number" in entities:
                return f"Tracking order {entities['order_number']}. It is currently out for delivery and expected by 5 PM today."
            if entities and "missing" in entities:
                return "I understand your order is missing. Please provide your order number so I can investigate further."
            return "Please provide your order number so I can track your shipment or investigate the problem."
        elif intent == "live agent":
            return "Connecting you to a live support agent. Please hold while I transfer your chat."
        else:
            return "I'm sorry, I couldn't perform that action. Would you like to connect with a live agent?"

class PersonalizedLearning:
    def __init__(self):
        self.user_preferences = {}

    def update_preference(self, user_id, intent, preference_data):
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        self.user_preferences[user_id][intent] = preference_data

    def get_preference(self, user_id, intent):
        return self.user_preferences.get(user_id, {}).get(intent)

class DialogueManager:
    def __init__(self, nlu, tool_executor, personalized_learning):
        self.nlu = nlu
        self.tool_executor = tool_executor
        self.personalized_learning = personalized_learning
        self.conversation_history = {}
        self.current_context = {}

    def process_query(self, user_id, query):
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
            self.current_context[user_id] = {'last_nlu_result': None, 'awaiting_clarification': False, 'clarification_for_intent': None}

        self.conversation_history[user_id].append({"user": query})
        context = self.current_context[user_id]
        
        nlu_result = self.nlu.classify_intent(query)
        intent = nlu_result['intent']
        ambiguous = nlu_result['ambiguous']
        clarification_message = nlu_result.get('clarification', "")
        
        response = ""

        if intent == "live agent":
            response = self.tool_executor.execute_tool("live agent")
            context['awaiting_clarification'] = False
            context['clarification_for_intent'] = None
        elif context['awaiting_clarification']:
            if not ambiguous and intent == context['clarification_for_intent']:
                context['awaiting_clarification'] = False
                context['clarification_for_intent'] = None
                response = self.handle_resolved_intent(user_id, intent, query)
            else:
                response = f"I'm still trying to understand. {clarification_message if clarification_message else 'Could you please be more specific?'}"
                if context['clarification_for_intent'] and context['clarification_for_intent'] != "unknown" and not clarification_message:
                    response += f" Regarding your {context['clarification_for_intent']}, {random.choice(self.nlu.intents[context['clarification_for_intent']]['clarification'])}"
                
        elif ambiguous:
            context['awaiting_clarification'] = True
            context['clarification_for_intent'] = intent if intent != "unknown" else None
            response = clarification_message
        else:
            response = self.handle_resolved_intent(user_id, intent, query)
            
        self.current_context[user_id]['last_nlu_result'] = nlu_result
        self.conversation_history[user_id].append({"assistant": response})
        return response

    def handle_resolved_intent(self, user_id, intent, query):
        entities = self._extract_entities(query)
        
        preference = self.personalized_learning.get_preference(user_id, intent)
        if intent == "account management" and "bill" in query.lower() and preference == "view_current_bill":
            return "Based on your past interactions, you often look for your current bill. " + self.tool_executor.execute_tool(intent, {"bill": True})
        
        tool_response = self.tool_executor.execute_tool(intent, entities)

        if intent == "account management" and "bill" in query.lower():
            self.personalized_learning.update_preference(user_id, intent, "view_current_bill")

        return tool_response

    def _extract_entities(self, query):
        entities = {}
        query_lower = query.lower()

        order_number_match = re.search(r"(?:order|tracking)\s*number\s*(\w+)", query_lower)
        if order_number_match:
            entities["order_number"] = order_number_match.group(1)

        if "bill" in query_lower or "billing" in query_lower or "charges" in query_lower:
            entities["bill"] = True
        if "password" in query_lower:
            entities["password"] = True
        if "plan" in query_lower or "service plan" in query_lower:
            entities["plan"] = True

        if "outage" in query_lower:
            entities["outage"] = True
        if "slow speeds" in query_lower or "slow internet" in query_lower:
            entities["slow speeds"] = True
        if "intermittent connection" in query_lower:
            entities["intermittent"] = True
        
        if "missing" in query_lower or "not received" in query_lower:
            entities["missing"] = True

        return entities

def main():
    nlu_service = NLU()
    tool_executor_service = ToolExecutor()
    personalized_learning_service = PersonalizedLearning()
    dialogue_manager = DialogueManager(nlu_service, tool_executor_service, personalized_learning_service)

    print("Welcome to the AI Customer Support Assistant! How can I help you today?")
    user_id = "customer_123"

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit", "bye", "goodbye"]:
            print("Assistant: Goodbye! Have a great day.")
            break
        
        response = dialogue_manager.process_query(user_id, user_query)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()