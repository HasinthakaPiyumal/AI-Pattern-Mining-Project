from typing import Any, List, Mapping, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate

# 1. Mock LLM Core
class MockLLM(LLM):
    response_map = {
        "order status": "To check your order status, let's think step by step: First, navigate to the 'My Orders' section on our website. Second, locate the specific order you are inquiring about. Third, click on the order to view its detailed status, including tracking information if available.",
        "return policy": "To understand our return policy, let's think step by step: First, please visit our 'Returns & Refunds' page. Second, identify if your item meets the eligibility criteria (e.g., within X days of purchase, original condition). Third, follow the instructions for initiating a return, which usually involves filling out a form or contacting support.",
        "shipping cost": "To determine shipping costs, let's think step by step: First, add the desired items to your cart. Second, proceed to the checkout page. Third, enter your shipping address to calculate the exact cost based on your location and selected shipping method.",
        "product availability": "To check product availability, let's think step by step: First, search for the product using the search bar. Second, navigate to the product page. Third, check the 'Add to Cart' button or the stock indicator; if it says 'In Stock' or allows adding to cart, it's available."
    }

    @property
    def _llm_type(self) -> str:
        return "mock_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        # Simulate a ZeroShot CoT response based on keywords
        for keyword, response in self.response_map.items():
            if keyword in prompt.lower():
                return response
        return "Let's think step by step: I am a customer support chatbot. Please rephrase your query or ask a common e-commerce question like 'order status' or 'return policy' for a detailed step-by-step response."

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}

# 4. Simulated Knowledge Base (simple dictionary for context)
KNOWLEDGE_BASE = {
    "order status": "You can find your order status in the 'My Orders' section after logging in.",
    "return policy": "Our return policy allows returns within 30 days of purchase for most items, provided they are in original condition.",
    "shipping cost": "Shipping costs vary based on location and shipping speed. You can see the exact cost at checkout.",
    "product availability": "Product availability is displayed on each product page."
}

# 2. Prompt Engineering Module
def apply_zeroshot_cot(query: str) -> str:
    return f"Let's think step by step. {query}"

# 5. Orchestration Logic and 3. Chatbot Interface
def run_chatbot():
    print("Welcome to the E-commerce ZeroShot CoT Chatbot!")
    print("Ask me a question about your order, returns, shipping, or product availability. (Type 'exit' to quit)")

    llm = MockLLM()

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Thank you for using the chatbot. Goodbye!")
            break

        # Retrieve relevant context from simulated knowledge base (optional, for richer responses)
        context = ""
        for keyword, info in KNOWLEDGE_BASE.items():
            if keyword in user_query.lower():
                context = info
                break

        # Prompt Engineering
        engineered_prompt = apply_zeroshot_cot(f"Query: {user_query}. Context: {context}") if context else apply_zeroshot_cot(f"Query: {user_query}")

        print(f"Chatbot (thinking): Sending '{engineered_prompt}' to LLM...")

        # LLM Core interaction
        response = llm.invoke(engineered_prompt)

        print(f"Chatbot: {response}")

if __name__ == "__main__":
    run_chatbot()