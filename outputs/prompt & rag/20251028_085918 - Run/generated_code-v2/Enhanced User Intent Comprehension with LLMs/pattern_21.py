import random

class IntelliSupportAgent:
    def __init__(self):
        self.intents = {
            "technical_issue": {
                "keywords": ["internet", "wifi", "router", "connection", "slow", "not working", "device"],
                "questions": [
                    "Are you experiencing no internet connection at all, slow speeds, or an issue with a specific device?",
                    "Could you describe the technical issue in more detail?"
                ],
                "tools": {
                    "no_connection": self._network_diagnostic_tool,
                    "slow_speeds": self._speed_troubleshoot_tool,
                    "device_issue": self._device_support_tool
                }
            },
            "billing_inquiry": {
                "keywords": ["bill", "invoice", "charge", "payment", "price", "cost"],
                "questions": [
                    "Are you asking about a recent bill, a payment issue, or pricing for a service?",
                    "Could you provide your account number or the billing period in question?"
                ],
                "tools": {
                    "view_bill": self._billing_lookup_tool,
                    "payment_issue": self._payment_resolution_tool
                }
            },
            "product_information": {
                "keywords": ["product", "service", "feature", "details", "about"],
                "questions": [
                    "Which product or service are you interested in?",
                    "Are you looking for features, pricing, or compatibility information?"
                ],
                "tools": {
                    "product_details": self._product_info_tool
                }
            },
            "account_management": {
                "keywords": ["account", "login", "password", "update", "change", "cancel"],
                "questions": [
                    "Are you trying to update your account details, change your password, or something else?",
                    "What specifically would you like to manage with your account?"
                ],
                "tools": {
                    "password_reset": self._password_reset_tool,
                    "update_details": self._update_account_details_tool
                }
            },
            "general_query": {
                "keywords": [],
                "questions": [
                    "Can you rephrase your question?",
                    "Could you tell me more about what you need assistance with?"
                ],
                "tools": {}
            }
        }

        self.confidence_threshold = 0.7

    def _preprocess_query(self, query):
        return query.lower().strip()

    def _initial_intent_recognition(self, preprocessed_query):
        scores = {intent: 0 for intent in self.intents}
        for intent, data in self.intents.items():
            for keyword in data["keywords"]:
                if keyword in preprocessed_query:
                    scores[intent] += 1
        
        if not any(scores.values()):
            return "general_query", 0.5 # Default to general if no keywords match

        max_score_intent = max(scores, key=scores.get)
        confidence = scores[max_score_intent] / (len(preprocessed_query.split()) + 1) # Simple confidence calculation
        
        # Simulate LLM providing a slightly varied confidence
        confidence += random.uniform(-0.1, 0.1)
        confidence = max(0.1, min(0.95, confidence))
        
        return max_score_intent, confidence

    def _detect_ambiguity(self, intent, confidence, preprocessed_query):
        if confidence < self.confidence_threshold:
            return True
        
        # Example of keyword-based ambiguity for known intents
        if intent == "technical_issue" and any(k in preprocessed_query for k in ["internet", "slow", "not working"]):
            return True # These keywords often need clarification

        return False

    def _generate_clarifying_questions(self, intent):
        return self.intents.get(intent, {}).get("questions", ["Could you please provide more details?"])

    def _get_user_clarification(self, questions):
        print(f"IntelliSupport: {random.choice(questions)}")
        clarification = input("You: ")
        return clarification

    def _refine_intent(self, original_query, clarification):
        combined_query = original_query + " " + clarification
        return self._initial_intent_recognition(self._preprocess_query(combined_query))

    def _network_diagnostic_tool(self, parameters=None):
        print("IntelliSupport: Running network diagnostics... (Simulated result: No major issues found, please reboot your router.)")
        return "No major network issues detected, please try rebooting your router."

    def _speed_troubleshoot_tool(self, parameters=None):
        print("IntelliSupport: Initiating speed troubleshooting steps... (Simulated result: Check your Wi-Fi signal strength and close background applications.)")
        return "Please check your Wi-Fi signal strength and close any bandwidth-intensive background applications."

    def _device_support_tool(self, parameters=None):
        device = parameters.get("device", "your device") if parameters else "your device"
        print(f"IntelliSupport: Connecting you to a specialist for {device} issue. (Simulated result: Specialist will contact you within 5 minutes.)")
        return f"We are connecting you to a specialist to help with your {device} issue. They will contact you shortly."

    def _billing_lookup_tool(self, parameters=None):
        account_info = parameters.get("account_info", "your account") if parameters else "your account"
        print(f"IntelliSupport: Accessing billing information for {account_info}... (Simulated result: Your last bill was $59.99 on Oct 26th.)")
        return f"Your last bill for {account_info} was $59.99, paid on October 26th."

    def _payment_resolution_tool(self, parameters=None):
        print("IntelliSupport: Initiating payment issue resolution process... (Simulated result: Please check your payment methods on your online portal.)")
        return "For payment issues, please visit your online portal's billing section to review or update your payment methods."

    def _product_info_tool(self, parameters=None):
        product = parameters.get("product", "the requested product") if parameters else "the requested product"
        print(f"IntelliSupport: Retrieving information for {product}... (Simulated result: Details about {product} can be found on our website.)")
        return f"You can find detailed information about {product} on our official website's product page."

    def _password_reset_tool(self, parameters=None):
        print("IntelliSupport: Initiating password reset... (Simulated result: A password reset link has been sent to your registered email.)")
        return "A password reset link has been sent to your registered email address. Please check your inbox."

    def _update_account_details_tool(self, parameters=None):
        print("IntelliSupport: Directing you to account update page... (Simulated result: Please log in to update your details.)")
        return "Please log in to your account portal to update your personal details or contact information."

    def _extract_tool_parameters(self, intent, refined_query):
        parameters = {}
        if intent == "technical_issue":
            if "device" in refined_query:
                # A very simple keyword-based extraction for demonstration
                if "router" in refined_query: parameters["device"] = "router"
                elif "laptop" in refined_query: parameters["device"] = "laptop"
                elif "phone" in refined_query: parameters["device"] = "phone"
        elif intent == "billing_inquiry":
            if "account" in refined_query or "number" in refined_query: # Simplistic
                parameters["account_info"] = "your provided account"
        return parameters

    def _generate_personalized_response(self, intent, tool_output=None):
        if tool_output:
            return f"Based on your query and the tool execution: {tool_output}"
        
        responses = {
            "technical_issue": "I understand you're having a technical issue. Let me help you with that.",
            "billing_inquiry": "Regarding your billing inquiry, I can assist you.",
            "product_information": "I can provide you with details about our products or services.",
            "account_management": "For account management, please specify what you'd like to do.",
            "general_query": "I'm not entirely sure how to help. Could you please rephrase or provide more context?"
        }
        return responses.get(intent, "How can I help you today?")

    def _log_interaction(self, query, initial_intent, refined_intent, response, tool_executed=None):
        print(f"\n--- Interaction Log ---")
        print(f"User Query: {query}")
        print(f"Initial Intent: {initial_intent}")
        print(f"Refined Intent: {refined_intent}")
        print(f"Response: {response}")
        if tool_executed:
            print(f"Tool Executed: {tool_executed}")
        print(f"-----------------------\n")

    def process_query(self, user_query):
        preprocessed_query = self._preprocess_query(user_query)
        
        initial_intent, confidence = self._initial_intent_recognition(preprocessed_query)
        refined_intent = initial_intent # Start with initial, refine if needed
        
        print(f"IntelliSupport: Detected initial intent '{initial_intent}' with confidence {confidence:.2f}.")

        tool_output = None
        response_message = ""
        tool_executed = None

        if self._detect_ambiguity(initial_intent, confidence, preprocessed_query):
            print("IntelliSupport: Your query seems a bit vague. Let me ask some clarifying questions.")
            clarifying_questions = self._generate_clarifying_questions(initial_intent)
            user_clarification = self._get_user_clarification(clarifying_questions)
            
            refined_intent, _ = self._refine_intent(user_query, user_clarification)
            print(f"IntelliSupport: Based on your clarification, the refined intent is '{refined_intent}'.")
            
            # Update preprocessed query for tool parameter extraction if clarification added info
            preprocessed_query = self._preprocess_query(user_query + " " + user_clarification)

        # Tool Orchestration
        if refined_intent in self.intents and self.intents[refined_intent]["tools"]:
            # A simplified logic to pick a tool based on refined_intent and keywords in query
            selected_tool_func = None
            parameters = self._extract_tool_parameters(refined_intent, preprocessed_query)

            if refined_intent == "technical_issue":
                if "no connection" in preprocessed_query: selected_tool_func = self.intents["technical_issue"]["tools"]["no_connection"]
                elif "slow" in preprocessed_query: selected_tool_func = self.intents["technical_issue"]["tools"]["slow_speeds"]
                elif "device" in preprocessed_query: selected_tool_func = self.intents["technical_issue"]["tools"]["device_issue"]
                else: selected_tool_func = self.intents["technical_issue"]["tools"]["no_connection"] # Default for tech issues
            elif refined_intent == "billing_inquiry":
                if "bill" in preprocessed_query: selected_tool_func = self.intents["billing_inquiry"]["tools"]["view_bill"]
                elif "payment" in preprocessed_query: selected_tool_func = self.intents["billing_inquiry"]["tools"]["payment_issue"]
                else: selected_tool_func = self.intents["billing_inquiry"]["tools"]["view_bill"] # Default for billing
            elif refined_intent == "product_information":
                selected_tool_func = self.intents["product_information"]["tools"]["product_details"]
            elif refined_intent == "account_management":
                if "password" in preprocessed_query: selected_tool_func = self.intents["account_management"]["tools"]["password_reset"]
                else: selected_tool_func = self.intents["account_management"]["tools"]["update_details"]

            if selected_tool_func:
                tool_output = selected_tool_func(parameters)
                tool_executed = selected_tool_func.__name__
        
        response_message = self._generate_personalized_response(refined_intent, tool_output)
        print(f"IntelliSupport: {response_message}")

        self._log_interaction(user_query, initial_intent, refined_intent, response_message, tool_executed)

if __name__ == "__main__":
    agent = IntelliSupportAgent()
    
    print("Welcome to IntelliSupport! How can I help you today? (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        agent.process_query(user_input)
