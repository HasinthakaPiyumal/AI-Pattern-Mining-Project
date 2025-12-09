class QueryClassifier:
    def classify(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ["hello", "hi", "thanks", "how are you", "bye"]):
            return "straightforward"
        elif any(word in query_lower for word in ["price", "cost", "availability", "stock", "details", "info"]):
            return "moderate"
        elif any(word in query_lower for word in ["compare", "recommend", "troubleshoot", "best for"]):
            return "complex"
        return "moderate"

class ProductRetriever:
    def __init__(self):
        self.products = {
            "laptop": [
                "High-performance laptop with 16GB RAM and 512GB SSD.",
                "Available in silver and space gray.",
                "Price: $1200"
            ],
            "smartphone": [
                "Latest model smartphone with a 6.7-inch OLED display and 5G capability.",
                "Comes with a 2-year warranty and fast charging support.",
                "Price: $800"
            ],
            "headphone": [
                "Noise-cancelling over-ear headphones with long battery life (30 hours).",
                "Comfortable fit, suitable for travel and daily use.",
                "Price: $250"
            ],
            "smartwatch": [
                "Fitness tracking smartwatch with heart rate monitor and GPS.",
                "Water-resistant up to 50 meters, compatible with iOS and Android.",
                "Price: $300"
            ]
        }

    def retrieve(self, query: str) -> list[str]:
        query_lower = query.lower()
        retrieved_info = []
        for product_name, details in self.products.items():
            if product_name in query_lower:
                retrieved_info.extend(details)
        if not retrieved_info:
            return ["No specific product information found in our catalog for your query."]
        return retrieved_info

class LLMService:
    def generate_response(self, prompt: str) -> str:
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! How can I assist you today?"
        elif "thanks" in prompt.lower():
            return "You're welcome!"
        elif "bye" in prompt.lower():
            return "Goodbye! Have a great day!"
        return f"I understand you're looking for a direct answer. {prompt}. Please specify your needs if you have a product in mind."

    def perform_retrieval_augmented_generation(self, query: str, retrieved_info: list[str]) -> str:
        if retrieved_info:
            info_str = "\n".join(retrieved_info)
            return f"Based on your query '{query}' and our product information:\n{info_str}\nIs there anything else you would like to know?"
        return f"I couldn't find specific information for '{query}'. Can you please be more specific?"

    def perform_multi_step_reasoning(self, query: str, retrieved_info: list[str]) -> str:
        response_parts = [f"Let's break down your complex query: '{query}'."]
        if "compare" in query.lower() and len(retrieved_info) > 1:
            response_parts.append("To compare, I need specific details on what aspects you're interested in (e.g., price, features, performance).")
            response_parts.append("Here's what I found related to your query:\n" + "\n".join(retrieved_info))
            response_parts.append("Could you please tell me which products you'd like to compare?")
        elif "recommend" in query.lower():
            response_parts.append("To give you the best recommendation, I need more information about your preferences, budget, and intended use.")
            if retrieved_info:
                response_parts.append("Perhaps you're looking for something similar to these options:\n" + "\n".join(retrieved_info))
            response_parts.append("What are your key requirements for a product?")
        elif "troubleshoot" in query.lower():
            response_parts.append("Troubleshooting can be complex. Please describe the issue in detail, including what you've already tried.")
            if retrieved_info:
                response_parts.append("Here's some general product information that might be relevant:\n" + "\n".join(retrieved_info))
            response_parts.append("What specific problem are you encountering?")
        else:
            response_parts.append("This query requires deeper analysis. I'll need to process multiple steps. Please confirm if you'd like me to proceed with a detailed inquiry based on the following available information:\n" + "\n".join(retrieved_info) if retrieved_info else "I'm ready to delve deeper. What specific aspects of your query should I focus on?")

        return "\n".join(response_parts)

class ECommerceChatbot:
    def __init__(self):
        self.query_classifier = QueryClassifier()
        self.product_retriever = ProductRetriever()
        self.llm_service = LLMService()

    def process_query(self, query: str) -> str:
        complexity = self.query_classifier.classify(query)
        retrieved_info = []

        if complexity == "straightforward":
            return self.llm_service.generate_response(query)
        elif complexity == "moderate":
            retrieved_info = self.product_retriever.retrieve(query)
            return self.llm_service.perform_retrieval_augmented_generation(query, retrieved_info)
        elif complexity == "complex":
            retrieved_info = self.product_retriever.retrieve(query)
            return self.llm_service.perform_multi_step_reasoning(query, retrieved_info)
        else:
            return "I am unable to process this query at the moment."

if __name__ == "__main__":
    chatbot = ECommerceChatbot()
    print("Welcome to the E-commerce Chatbot! Type 'exit' to end the conversation.")

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        response = chatbot.process_query(user_query)
        print(f"Chatbot: {response}")
