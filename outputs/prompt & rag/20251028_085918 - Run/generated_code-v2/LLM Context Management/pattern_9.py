
class QueryComplexityClassifier:
    def classify(self, query: str) -> str:
        query_lower = query.lower()
        if "order status" in query_lower or "shipping info" in query_lower or "return policy" in query_lower and len(query.split()) < 10:
            return "straightforward"
        elif "troubleshoot" in query_lower or "product details" in query_lower or "compare" in query_lower and len(query.split()) < 20:
            return "moderate"
        else:
            return "complex"

class KnowledgeBase:
    def __init__(self):
        self.documents = {
            "order status": "To check your order status, please visit the 'My Orders' section on our website and enter your order ID.",
            "shipping info": "Standard shipping takes 3-5 business days. Expedited shipping options are available at checkout.",
            "return policy": "You can return most items within 30 days of purchase for a full refund. Please see our full return policy online.",
            "product details": "Our product XYZ features a 12MP camera, 6-inch display, and a 4000mAh battery.",
            "troubleshoot printer": "If your printer is not working, first check the power connection and ink levels. For further assistance, consult the user manual.",
            "account access": "If you are having trouble accessing your account, try resetting your password. Ensure you are using the correct email address.",
            "payment methods": "We accept Visa, MasterCard, American Express, PayPal, and Google Pay.",
            "damaged item": "If you received a damaged item, please contact our support team immediately with your order number and photos of the damage."
        }

    def retrieve(self, query_keywords: list) -> str:
        relevant_docs = []
        for keyword in query_keywords:
            for doc_key, doc_content in self.documents.items():
                if keyword.lower() in doc_key.lower() or keyword.lower() in doc_content.lower():
                    relevant_docs.append(doc_content)
        return " ".join(list(set(relevant_docs))) if relevant_docs else "No relevant information found in knowledge base."

class LargeLanguageModel:
    def generate_response(self, prompt: str) -> str:
        if "order status" in prompt.lower() and "order id" in prompt.lower():
            return f"Based on your query, please provide your order ID to check its status. (Simulated LLM response)"
        elif "shipping info" in prompt.lower():
            return f"The standard shipping time is 3-5 business days. Would you like to know more about expedited options? (Simulated LLM response)"
        elif "return policy" in prompt.lower():
            return f"Our return policy allows returns within 30 days of purchase. Is there a specific item you're looking to return? (Simulated LLM response)"
        elif "product details" in prompt.lower() and "XYZ" in prompt:
            return f"Regarding Product XYZ, it boasts a 12MP camera and a 6-inch display. What specific details are you interested in? (Simulated LLM response)"
        elif "troubleshoot" in prompt.lower():
            return f"I can help you troubleshoot. Please describe the problem in more detail. (Simulated LLM response)"
        elif "account access" in prompt.lower():
            return f"If you're having trouble with account access, you can try resetting your password. If that doesn't work, let me know. (Simulated LLM response)"
        elif "payment methods" in prompt.lower():
            return f"We accept various payment methods including Visa, MasterCard, American Express, PayPal, and Google Pay. (Simulated LLM response)"
        elif "damaged item" in prompt.lower():
            return f"I understand you received a damaged item. Please provide your order number and photos of the damage so we can assist you further. (Simulated LLM response)"
        elif "human agent" in prompt.lower() or "escalate" in prompt.lower():
            return "I am escalating your query to a human agent. Please wait while we connect you. (Simulated LLM response - Human Handoff)"
        elif "no relevant information" in prompt.lower():
             return "I couldn't find a direct answer in my knowledge base. Would you like me to try a different approach or connect you to a human agent? (Simulated LLM response)"
        else:
            return f"I'm processing your request. How else can I assist you? (Simulated LLM response - General)"

class SmartCustomerSupportAssistant:
    def __init__(self):
        self.qcc = QueryComplexityClassifier()
        self.kb = KnowledgeBase()
        self.llm = LargeLanguageModel()

    def handle_query(self, query: str) -> str:
        complexity = self.qcc.classify(query)

        if complexity == "straightforward":
            return self._handle_straightforward_query(query)
        elif complexity == "moderate":
            return self._handle_moderate_query(query)
        elif complexity == "complex":
            return self._handle_complex_query(query)
        else:
            return "An unexpected error occurred in query classification."

    def _handle_straightforward_query(self, query: str) -> str:
        query_lower = query.lower()
        if "order status" in query_lower:
            return "To check your order status, please visit the 'My Orders' section on our website and enter your order ID."
        elif "shipping info" in query_lower:
            return "Standard shipping takes 3-5 business days. Expedited shipping options are available at checkout."
        elif "return policy" in query_lower:
            return "You can return most items within 30 days of purchase for a full refund. Please see our full return policy online."
        else:
            return "I can directly provide information on common topics. Could you please specify your request?"

    def _handle_moderate_query(self, query: str) -> str:
        keywords = query.lower().split()
        retrieved_context = self.kb.retrieve(keywords)
        prompt = f"Based on the query: '{query}' and the following context: '{retrieved_context}', please generate a concise answer."
        llm_response = self.llm.generate_response(prompt)
        return f"Moderate Query Handler (Single-step RAG): {llm_response}"

    def _handle_complex_query(self, query: str) -> str:
        initial_keywords = query.lower().split()
        context_step1 = self.kb.retrieve(initial_keywords)
        
        if "no relevant information" in context_step1.lower() or len(query.split()) > 25:
            # Simulate escalation for very complex or unanswerable queries
            prompt_escalate = f"The query '{query}' is complex and requires further investigation or human intervention. Escalating to human agent."
            return self.llm.generate_response(prompt_escalate)
        
        # Simulate multi-step RAG
        refined_keywords = [word for word in initial_keywords if len(word) > 3] # simple refinement
        context_step2 = self.kb.retrieve(refined_keywords)
        
        combined_context = f"{context_step1} {context_step2}"
        prompt = f"Given the complex query: '{query}' and the detailed context: '{combined_context}', provide a comprehensive answer. If unable to fully resolve, suggest further steps or escalation."
        llm_response = self.llm.generate_response(prompt)
        
        if "escalate" in llm_response.lower() or "human agent" in llm_response.lower():
            return llm_response
        else:
            return f"Complex Query Handler (Multi-step RAG): {llm_response}"


if __name__ == "__main__":
    assistant = SmartCustomerSupportAssistant()

    print("--- Straightforward Queries ---")
    print(f"Query: What is my order status?\nResponse: {assistant.handle_query('What is my order status?')}\n")
    print(f"Query: Tell me about shipping info.\nResponse: {assistant.handle_query('Tell me about shipping info.')}\n")
    print(f"Query: I want to return an item. What is the policy?\nResponse: {assistant.handle_query('I want to return an item. What is the policy?')}\n")

    print("--- Moderate Queries ---")
    print(f"Query: Can you give me product details for XYZ?\nResponse: {assistant.handle_query('Can you give me product details for XYZ?')}\n")
    print(f"Query: My printer is not working, how to troubleshoot?\nResponse: {assistant.handle_query('My printer is not working, how to troubleshoot?')}\n")
    print(f"Query: I can't access my account. What should I do?\nResponse: {assistant.handle_query('I can\'t access my account. What should I do?')}\n")

    print("--- Complex Queries ---")
    print(f"Query: I need detailed information about the new privacy policy updates, how they affect my data, and the process for data rectification, specifically concerning third-party sharing agreements and compliance with GDPR regulations.\nResponse: {assistant.handle_query('I need detailed information about the new privacy policy updates, how they affect my data, and the process for data rectification, specifically concerning third-party sharing agreements and compliance with GDPR regulations.')}\n")
    print(f"Query: I received a damaged item, order number 12345, and I also have a question about setting up recurring payments for my subscription. Please help.\nResponse: {assistant.handle_query('I received a damaged item, order number 12345, and I also have a question about setting up recurring payments for my subscription. Please help.')}\n")
    print(f"Query: I have a very obscure question about a discontinued product from five years ago, its warranty details, and if any replacement parts are available internationally. This also involves a complex billing dispute from a previous order.\nResponse: {assistant.handle_query('I have a very obscure question about a discontinued product from five years ago, its warranty details, and if any replacement parts are available internationally. This also involves a complex billing dispute from a previous order.')}\n")
    print(f"Query: Where can I find information about the company's environmental sustainability initiatives and their impact on local communities, including detailed reports and future plans for renewable energy adoption?\nResponse: {assistant.handle_query('Where can I find information about the company\'s environmental sustainability initiatives and their impact on local communities, including detailed reports and future plans for renewable energy adoption?')}\n")
