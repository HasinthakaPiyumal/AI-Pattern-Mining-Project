from intent_classifier import IntentClassifier
from tool_manager import ToolManager

class CustomerSupportAssistant:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.tool_manager = ToolManager()
        self.user_context = {}

    def process_query(self, user_id: str, query: str) -> str:
        # Store user's last query for potential context in a real system
        self.user_context[user_id] = {"last_query": query}

        intent = self.intent_classifier.classify_intent(query)

        if intent == "ambiguous_intent":
            # In a real LLM, this would involve retrieving the top N potential intents with scores
            # For this simplified example, we'll simulate by just showing some common ambiguous ones.
            potential_intents = ["order_status", "refund_inquiry", "return_request"]
            return self.intent_classifier.clarify_ambiguity(query, potential_intents)
        elif intent == "unclear_intent":
            return "I'm sorry, I couldn't understand your request. Could you please rephrase it or provide more details?"
        elif intent == "order_status":
            return self.tool_manager.get_order_status(user_id)
        elif intent == "return_request":
            return self.tool_manager.initiate_return(user_id)
        elif intent == "refund_inquiry":
            return self.tool_manager.process_refund_inquiry(user_id)
        elif intent == "faq_general":
            # For FAQ, we'd ideally pass the query to the FAQ tool for a more specific answer
            return self.tool_manager.get_faq_answer(query)
        elif intent == "live_agent":
            return self.tool_manager.hand_over_to_agent(query)
        else:
            return "An unexpected error occurred. Please try again later."

# Example Usage:
if __name__ == "__main__":
    assistant = CustomerSupportAssistant()

    print("\n--- Scenario 1: Order Status ---")
    response = assistant.process_query("customer123", "Where is my package?")
    print(f"Customer: Where is my package?\nAssistant: {response}")

    print("\n--- Scenario 2: Return Request ---")
    response = assistant.process_query("customer123", "I need to return an item.")
    print(f"Customer: I need to return an item.\nAssistant: {response}")

    print("\n--- Scenario 3: Refund Inquiry ---")
    response = assistant.process_query("customer123", "How can I get my money back?")
    print(f"Customer: How can I get my money back?\nAssistant: {response}")

    print("\n--- Scenario 4: General FAQ ---")
    response = assistant.process_query("customer123", "Tell me about shipping.")
    print(f"Customer: Tell me about shipping.\nAssistant: {response}")

    print("\n--- Scenario 5: Live Agent Handover ---")
    response = assistant.process_query("customer123", "I want to talk to a human.")
    print(f"Customer: I want to talk to a human.\nAssistant: {response}")

    print("\n--- Scenario 6: Unclear Intent ---")
    response = assistant.process_query("customer123", "Blah blah blah.")
    print(f"Customer: Blah blah blah.\nAssistant: {response}")

    print("\n--- Scenario 7: Ambiguous Intent (simulated) ---")
    # In a real system, keywords like 'late' could trigger ambiguity between status/refund/complaint
    # Here we'll simulate by having an ambiguous intent returned from the classifier.
    # For this specific example, the classifier would likely lean towards 'order_status' based on the current simple keyword logic.
    # To truly test 'ambiguous_intent' from the classifier, the `IntentClassifier` class would need a more sophisticated keyword overlap.
    # Let's manually trigger it for demonstration.
    original_classify_intent = IntentClassifier.classify_intent
    def mock_classify_intent(self, query):
        if "late" in query.lower():
            return "ambiguous_intent"
        return original_classify_intent(self, query)
    IntentClassifier.classify_intent = mock_classify_intent

    response = assistant.process_query("customer123", "My order is late!")
    print(f"Customer: My order is late!\nAssistant: {response}")

    # Reset the mock for other tests if any
    IntentClassifier.classify_intent = original_classify_intent