import streamlit as st
import random

class QueryComplexityClassifier:
    def __init__(self):
        pass

    def classify(self, query: str) -> str:
        # In a real application, this would be a fine-tuned smaller LLM
        # For demonstration, we'll use a simplified rule-based/random classification.
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["faq", "return policy", "shipping cost", "contact info"]):
            return "straightforward"
        elif any(keyword in query_lower for keyword in ["product comparison", "troubleshooting", "account issue"]):
            return "moderate"
        elif any(keyword in query_lower for keyword in ["custom order", "technical specifications detailed", "integration help"]):
            return "complex"
        else:
            # Randomly assign if no keywords match, to simulate a model's variability
            return random.choice(["straightforward", "moderate", "complex"])


class DirectFAQResponse:
    def __init__(self):
        self.faq_db = {
            "return policy": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
            "shipping cost": "Shipping costs vary based on your location and chosen shipping speed. Please see our shipping page for details.",
            "contact info": "You can reach our support team at support@ecommerce.com or call us at 1-800-123-4567.",
            "faq": "Please visit our Frequently Asked Questions section on our website for more information."
        }

    def generate_response(self, query: str) -> str:
        query_lower = query.lower()
        for keyword, answer in self.faq_db.items():
            if keyword in query_lower:
                return answer
        return "I'm sorry, I couldn't find a direct answer in our FAQ for that specific query."


class SingleStepRAG:
    def __init__(self):
        # In a real application, this would involve a vector database and an LLM
        pass

    def generate_response(self, query: str) -> str:
        # Simulate RAG by generating a more detailed, but still generic, response.
        return f"Based on a quick retrieval, I can tell you more about '{query}'. It involves looking up relevant product details or documentation and summarizing them for you."


class MultiStepRAG:
    def __init__(self):
        # In a real application, this would involve an agentic workflow with multiple retrieval and reasoning steps
        pass

    def generate_response(self, query: str) -> str:
        # Simulate multi-step RAG
        return f"For your complex query about '{query}', our advanced system performs multiple retrieval and reasoning steps to provide a comprehensive answer. This might involve cross-referencing several documents and external knowledge bases."


class HumanEscalation:
    def __init__(self):
        pass

    def escalate(self, query: str, context: str = "") -> str:
        return f"Your query: '{query}' has been escalated to a human agent. Please provide your contact details if you haven't already. An agent will get back to you shortly."


class StrategySelector:
    def __init__(self):
        self.direct_faq = DirectFAQResponse()
        self.single_step_rag = SingleStepRAG()
        self.multi_step_rag = MultiStepRAG()
        self.human_escalation = HumanEscalation()

    def select_and_execute_strategy(self, complexity_label: str, query: str) -> str:
        if complexity_label == "straightforward":
            return self.direct_faq.generate_response(query)
        elif complexity_label == "moderate":
            return self.single_step_rag.generate_response(query)
        elif complexity_label == "complex":
            # For complex, we could try multi-step RAG first, then escalate if needed
            rag_response = self.multi_step_rag.generate_response(query)
            # In a real system, there would be logic to determine if RAG was sufficient
            if "comprehensive answer" in rag_response: # Simple heuristic for demo
                return rag_response
            else:
                return self.human_escalation.escalate(query)
        else:
            return self.human_escalation.escalate(query)


def main():
    st.title("E-commerce Customer Support Chatbot")
    st.subheader("Dynamic Strategy Adaptation Demo")

    classifier = QueryComplexityClassifier()
    selector = StrategySelector()

    user_query = st.text_input("Ask a question about our products or services:")

    if user_query:
        st.write(f"You asked: '{user_query}'")

        # 1. Classify the query
        complexity = classifier.classify(user_query)
        st.info(f"Query classified as: **{complexity.upper()}**")

        # 2. Select and execute strategy
        response = selector.select_and_execute_strategy(complexity, user_query)
        st.success(f"Chatbot Response: {response}")

if __name__ == "__main__":
    main()
