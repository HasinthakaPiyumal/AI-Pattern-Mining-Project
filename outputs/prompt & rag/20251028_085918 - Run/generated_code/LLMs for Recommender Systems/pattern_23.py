
import os
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import OpenAI # Placeholder for actual LLM

# --- Mock Product Database ---
mock_products = [
    {
        "id": "p001",
        "name": "Wireless Bluetooth Headphones",
        "category": "Electronics",
        "price": 79.99,
        "description": "High-quality sound, comfortable fit, 20-hour battery life. Perfect for travel and workouts."
    },
    {
        "id": "p002",
        "name": "Organic Green Tea - 100 bags",
        "category": "Food & Beverage",
        "price": 12.50,
        "description": "Premium organic green tea, rich in antioxidants. Enjoy a soothing cup any time of the day."
    },
    {
        "id": "p003",
        "name": "Ergonomic Office Chair",
        "category": "Home & Office",
        "price": 249.00,
        "description": "Adjustable lumbar support, breathable mesh, suitable for long working hours. Boost your productivity."
    },
    {
        "id": "p004",
        "name": "Smartphone Tripod with Remote",
        "category": "Electronics",
        "price": 25.99,
        "description": "Lightweight and portable tripod for smartphones, includes a wireless remote for perfect selfies and videos."
    },
    {
        "id": "p005",
        "name": "Stainless Steel Water Bottle",
        "category": "Home & Kitchen",
        "price": 19.95,
        "description": "Double-walled insulation keeps drinks cold for 24 hours and hot for 12 hours. Eco-friendly and durable."
    },
]

class ECommerceAssistant:
    def __init__(self, api_key: str = None):
        # Initialize LLM. For production, replace with actual LLM like ChatOpenAI, Llama-2, etc.
        # Ensure OPENAI_API_KEY or equivalent environment variable is set for actual use.
        self.llm = OpenAI(openai_api_key=api_key) if api_key else self._mock_llm
        self.products = mock_products
        self.output_parser = StrOutputParser()

    def _mock_llm(self, prompt: str) -> str:
        """A simple mock LLM for demonstration purposes without an API key."""
        if "recommend products" in prompt.lower():
            return "Based on your preferences, I recommend: Wireless Bluetooth Headphones and Organic Green Tea."
        elif "answer query" in prompt.lower():
            return "Thank you for your query. How can I help you further? (This is a mock response)"
        elif "generate description" in prompt.lower():
            return "This is a dynamically generated product description for your item. (Mock)"
        elif "search for products" in prompt.lower():
            if "headphones" in prompt.lower():
                return "Found: Wireless Bluetooth Headphones."
            elif "tea" in prompt.lower():
                return "Found: Organic Green Tea - 100 bags."
            else:
                return "No specific product found for your search (mock)."
        return "Mock LLM Response: I am an AI assistant."

    def _invoke_llm_chain(self, prompt_template: ChatPromptTemplate, input_data: Dict[str, Any]) -> str:
        """Helper to invoke an LLM chain."""
        chain = prompt_template | self.llm | self.output_parser
        return chain.invoke(input_data)

    def get_personalized_recommendations(self, user_preferences: str) -> str:
        """Generates personalized product recommendations based on user preferences."""
        product_list_str = "\n".join([f"- {p['name']} ({p['category']}) - {p['description']}" for p in self.products])
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful e-commerce assistant. Provide personalized product recommendations from the given product list based on user preferences. List up to 3 relevant products."),
            ("human", "User preferences: {user_preferences}\n\nAvailable Products:\n{product_list}\n\nRecommendations:")
        ])
        return self._invoke_llm_chain(prompt_template, {"user_preferences": user_preferences, "product_list": product_list_str})

    def answer_customer_query(self, query: str, context: str = "") -> str:
        """Provides intelligent customer support by answering queries."""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an intelligent e-commerce customer support agent. Answer the user's question concisely and helpfully. If context is provided, use it to inform your answer."),
            ("human", "Context: {context}\nQuestion: {query}\nAnswer:")
        ])
        return self._invoke_llm_chain(prompt_template, {"query": query, "context": context})

    def generate_product_description(self, product_id: str) -> str:
        """Generates a dynamic and engaging product description for a given product ID."""
        product_info = next((p for p in self.products if p['id'] == product_id), None)
        if not product_info:
            return f"Product with ID '{product_id}' not found."

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a creative marketing copywriter for an e-commerce platform. Generate an engaging and descriptive product description (approx. 100-150 words) for the following product, highlighting its key features and benefits. Focus on appealing to the customer."),
            ("human", "Product Name: {name}\nCategory: {category}\nPrice: ${price:.2f}\nKey Info: {description}\n\nGenerated Description:")
        ])
        return self._invoke_llm_chain(prompt_template, product_info)

    def search_products_natural_language(self, natural_language_query: str) -> List[Dict[str, Any]]:
        """Facilitates natural language-based product search and discovery."""
        # First, use LLM to extract keywords/categories from the natural language query
        keyword_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a product search assistant. Extract 1-3 primary keywords or categories from the user's natural language search query that can be used to find products. Respond with comma-separated keywords."),
            ("human", "Search query: {query}")
        ])
        extracted_keywords_str = self._invoke_llm_chain(keyword_extraction_prompt, {"query": natural_language_query})
        extracted_keywords = [k.strip().lower() for k in extracted_keywords_str.split(',') if k.strip()]

        # Simulate search based on extracted keywords
        matching_products = []
        for product in self.products:
            product_text = f"{product['name']} {product['category']} {product['description']}".lower()
            if any(keyword in product_text for keyword in extracted_keywords):
                matching_products.append(product)
        
        # If LLM failed to extract keywords or no direct match, could fall back to a more general LLM-driven search
        if not matching_products and extracted_keywords:
             # This is a fallback/enhancement where LLM could directly 'find' product if it had deep knowledge
             # For now, we return empty if no direct keyword match.
             pass

        return matching_products

if __name__ == "__main__":
    # Example Usage
    # For actual LLM integration, set your OpenAI API key in the environment or pass it to the constructor
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

    # If using a real LLM, uncomment the line below and ensure API key is set.
    # assistant = ECommerceAssistant(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Using the mock LLM for demonstration without an actual API key
    assistant = ECommerceAssistant()
    print("--- E-commerce Assistant Demo ---\n")

    print("1. Personalized Product Recommendations:")
    user_prefs = "I'm looking for something to help me relax, maybe a healthy drink, or something for my home office setup."
    recommendations = assistant.get_personalized_recommendations(user_prefs)
    print(f"User preferences: '{user_prefs}'\nRecommendations: {recommendations}\n")

    print("2. Intelligent Customer Support:")
    customer_q1 = "What is the battery life of the wireless headphones?"
    support_answer1 = assistant.answer_customer_query(customer_q1, context=mock_products[0]['description'])
    print(f"Customer Q: '{customer_q1}'\nAnswer: {support_answer1}\n")
    
    customer_q2 = "Do you have organic teas?"
    support_answer2 = assistant.answer_customer_query(customer_q2)
    print(f"Customer Q: '{customer_q2}'\nAnswer: {support_answer2}\n")

    print("3. Dynamic Product Description Generation:")
    product_id_to_describe = "p003"
    description = assistant.generate_product_description(product_id_to_describe)
    print(f"Product ID '{product_id_to_describe}' description: {description}\n")

    print("4. Natural Language Product Search:")
    search_query1 = "I need some comfortable headphones for my daily commute."
    search_results1 = assistant.search_products_natural_language(search_query1)
    print(f"Search query: '{search_query1}'\nResults: {[p['name'] for p in search_results1]}\n")

    search_query2 = "Show me some healthy beverages."
    search_results2 = assistant.search_products_natural_language(search_query2)
    print(f"Search query: '{search_query2}'\nResults: {[p['name'] for p in search_results2]}\n")

    search_query3 = "Looking for something for photography."
    search_results3 = assistant.search_products_natural_language(search_query3)
    print(f"Search query: '{search_query3}'\nResults: {[p['name'] for p in search_results3]}\n")

    search_query4 = "Do you have any gaming chairs?"
    search_results4 = assistant.search_products_natural_language(search_query4)
    print(f"Search query: '{search_query4}'\nResults: {[p['name'] for p in search_results4]}\n")
