class SmartCustomerSupportAssistant:
    def __init__(self):
        self.rephrase_map = {
            "internet isn't working": "The customer is experiencing a network connectivity issue. Investigate potential causes like no signal, slow speed, or specific device failure. Suggest initial troubleshooting like router restart or request more details.",
            "my bill is wrong": "The customer has a billing discrepancy. Investigate recent charges, plan changes, or promotional offers. Prepare to check account details.",
            "can't log in": "The customer is having trouble accessing their account. Investigate forgotten password, locked account, or incorrect credentials. Suggest password reset or account recovery steps.",
            "slow speed": "The customer is reporting slow internet speed. Investigate network congestion, device limitations, or service plan details. Suggest speed test or router reset."
        }

        self.response_map = {
            "network connectivity issue": "I understand your internet isn't working. To help me troubleshoot, could you please tell me: 1. Are you getting any error messages? 2. Which device are you trying to connect with (e.g., laptop, phone)? 3. Have you tried restarting your router?",
            "billing discrepancy": "I can help you with your bill. Could you please provide your account number or the email associated with your account so I can look into the charges?",
            "trouble accessing their account": "I apologize for the inconvenience. Are you having trouble remembering your password or is there another issue? We can try a password reset or account recovery.",
            "slow internet speed": "I see you're experiencing slow internet. Could you please perform a speed test at [link to speed test site] and tell me the results? Also, have you tried restarting your modem and router?"
        }

    def receive_query(self, query: str) -> str:
        return query.lower()

    def rephrase_and_expand_query(self, original_query: str) -> str:
        for phrase, expansion in self.rephrase_map.items():
            if phrase in original_query:
                return expansion
        return f"The customer is asking about: {original_query}. Further investigation is needed."

    def generate_response(self, expanded_query: str) -> str:
        for keyword, response in self.response_map.items():
            if keyword in expanded_query:
                return response
        return "I'm sorry, I need more information to assist you. Could you please elaborate on your issue?"

    def output_response(self, response: str) -> None:
        print(response)

    def run_assistant(self, customer_query: str) -> None:
        original_input = self.receive_query(customer_query)
        expanded_question = self.rephrase_and_expand_query(original_input)
        final_response = self.generate_response(expanded_question)
        self.output_response(final_response)

if __name__ == "__main__":
    assistant = SmartCustomerSupportAssistant()

    print("--- Scenario 1: Internet Issue ---")
    assistant.run_assistant("My internet isn't working.")
    print("\n--- Scenario 2: Billing Issue ---")
    assistant.run_assistant("My bill is wrong this month.")
    print("\n--- Scenario 3: Login Problem ---")
    assistant.run_assistant("I can't log into my account.")
    print("\n--- Scenario 4: Slow Speed ---")
    assistant.run_assistant("My internet is super slow.")
    print("\n--- Scenario 5: Unhandled Query ---")
    assistant.run_assistant("I need help with my new phone.")