class QueryClassifier:
    def classify_query(self, query: str) -> str:
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["return policy", "shipping cost", "delivery time", "order status", "payment methods"]):
            return "straightforward"
        elif any(keyword in query_lower for keyword in ["troubleshoot", "compare", "recommendation", "product features", "account settings"]):
            return "moderate"
        else:
            return "complex"

class KnowledgeBase:
    def __init__(self):
        self.kb = {
            "return policy": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
            "shipping cost": "Standard shipping within the country costs $5.99. Expedited options are available.",
            "delivery time": "Standard delivery usually takes 3-5 business days after dispatch.",
            "order status": "Please provide your order number for us to check your order status.",
            "payment methods": "We accept Visa, MasterCard, American Express, PayPal, and Google Pay.",
            "troubleshoot login": "If you're having trouble logging in, please try resetting your password or clearing your browser cache.",
            "compare products": "To compare products, please provide the names or IDs of the items you are interested in.",
            "product recommendations": "We can offer product recommendations if you tell us your preferences or needs.",
            "account settings": "You can update your account settings, including shipping address and payment methods, in your profile section."
        }

    def retrieve_information(self, query: str) -> str:
        query_lower = query.lower()
        for keyword, info in self.kb.items():
            if keyword in query_lower:
                return info
        return "I couldn't find specific information for that in our knowledge base. Can you rephrase or provide more details?"

class ChatbotCore:
    def __init__(self):
        self.classifier = QueryClassifier()
        self.knowledge_base = KnowledgeBase()

    def process_customer_query(self, query: str) -> str:
        complexity = self.classifier.classify_query(query)

        if complexity == "straightforward":
            response = self.knowledge_base.retrieve_information(query)
            if "I couldn't find specific information" in response:
                return f"Got it. {response} For straightforward questions, I can usually help directly."
            return f"For your straightforward question: {response}"
        elif complexity == "moderate":
            retrieved_info = self.knowledge_base.retrieve_information(query)
            return f"This seems like a moderate query. Here's what I found: {retrieved_info} Is there anything else I can help with रिगार्डिंग this?"
        elif complexity == "complex":
            return "Your query appears complex and might require more in-depth assistance. I will escalate this to a human agent, or we can begin a multi-step problem-solving process. Would you like me to connect you?"
        else:
            return "I am unable to process this query at the moment."

if __name__ == "__main__":
    print("Welcome to the Adaptive Customer Support Chatbot! Type 'exit' to quit.")
    chatbot = ChatbotCore()

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break

        response = chatbot.process_customer_query(user_query)
        print(f"Chatbot: {response}")

    print("Goodbye!")