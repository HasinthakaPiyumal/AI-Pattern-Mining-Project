import json
from typing import Dict, List, Any
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.prompts import PromptTemplate

# --- Mock External Tools --- 

class MockSearchEngine:
    """Simulates a search engine for product information."""
    def search_products(self, query: str) -> List[Dict[str, Any]]:
        print(f"[DEBUG] MockSearchEngine: Searching for '{query}'")
        if "laptop" in query.lower():
            return [
                {"id": "prod101", "name": "Dell XPS 13 Laptop", "price": 1200, "description": "High-performance ultrabook.", "stock": 10, "rating": 4.7},
                {"id": "prod102", "name": "MacBook Air M2", "price": 1100, "description": "Thin and light, powerful M2 chip.", "stock": 15, "rating": 4.8}
            ]
        elif "headphone" in query.lower():
            return [
                {"id": "prod201", "name": "Sony WH-1000XM5", "price": 350, "description": "Industry-leading noise cancelling.", "stock": 20, "rating": 4.9}
            ]
        return []

class MockRecommendationEngine:
    """Simulates a recommendation engine."""
    def get_personalized_recommendations(self, user_id: str, last_viewed_product_id: str = None) -> List[Dict[str, Any]]:
        print(f"[DEBUG] MockRecommendationEngine: Getting recommendations for user '{user_id}'")
        if user_id == "user123":
            return [
                {"id": "prod301", "name": "Wireless Mouse", "price": 50, "category": "Accessories"},
                {"id": "prod302", "name": "External Monitor", "price": 250, "category": "Peripherals"}
            ]
        return []

class MockProductDatabase:
    """Simulates a product database (vector/relational)."""
    def get_product_details(self, product_id: str) -> Dict[str, Any]:
        print(f"[DEBUG] MockProductDatabase: Getting details for product '{product_id}'")
        details = {
            "prod101": {"brand": "Dell", "specs": "Intel i7, 16GB RAM, 512GB SSD", "reviews_summary": "Great performance, premium build.", "internal_reviews": ["Fast", "Good screen"]},
            "prod201": {"brand": "Sony", "specs": "Bluetooth 5.2, 30-hour battery", "reviews_summary": "Excellent noise cancellation, comfortable.", "internal_reviews": ["Amazing sound", "Comfortable fit"]}
        }
        return details.get(product_id, {})

class MockUserProfileDatabase:
    """Simulates a user profile database."""
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        print(f"[DEBUG] MockUserProfileDatabase: Getting preferences for user '{user_id}'")
        profiles = {
            "user123": {"preferred_categories": ["electronics", "books"], "budget_range": "medium", "last_purchased_item": "prod101"}
        }
        return profiles.get(user_id, {})

class MockInventoryManagementSystem:
    """Simulates an inventory system."""
    def check_stock(self, product_id: str) -> int:
        print(f"[DEBUG] MockInventoryManagementSystem: Checking stock for '{product_id}'")
        stock = {"prod101": 10, "prod201": 20, "prod301": 50}
        return stock.get(product_id, 0)

    def get_shipping_info(self, product_id: str, quantity: int) -> str:
        print(f"[DEBUG] MockInventoryManagementSystem: Getting shipping info for '{product_id}' (qty={quantity})'")
        if product_id == "prod101" and quantity <= 10: 
            return "Ships in 2-3 business days. Free standard shipping."
        return "Item out of stock or cannot be shipped to your location."

class MockSentimentAnalysisTool:
    """Simulates a sentiment analysis tool for reviews."""
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        print(f"[DEBUG] MockSentimentAnalysisTool: Analyzing sentiment for: '{text[:50]}...' ")
        if "excellent" in text.lower() or "amazing" in text.lower():
            return {"overall_sentiment": "positive", "score": 0.9}
        elif "bad" in text.lower() or "poor" in text.lower():
            return {"overall_sentiment": "negative", "score": 0.1}
        return {"overall_sentiment": "neutral", "score": 0.5}

# --- Initialize Tools --- 
search_engine = MockSearchEngine()
recommendation_engine = MockRecommendationEngine()
product_db = MockProductDatabase()
user_profile_db = MockUserProfileDatabase()
inventory_system = MockInventoryManagementSystem()
sentiment_analyzer = MockSentimentAnalysisTool()

# --- Create Langchain Tools --- 

tools = [
    Tool(
        name="SearchProducts",
        func=search_engine.search_products,
        description="Useful for searching for products based on keywords. Input should be a product query string."
    ),
    Tool(
        name="GetPersonalizedRecommendations",
        func=lambda user_id: recommendation_engine.get_personalized_recommendations(user_id),
        description="Useful for getting personalized product recommendations for a user. Input should be a user ID string."
    ),
    Tool(
        name="GetProductDetails",
        func=product_db.get_product_details,
        description="Useful for getting detailed specifications, brand, and review summaries for a specific product ID. Input should be a product ID string."
    ),
    Tool(
        name="GetUserPreferences",
        func=user_profile_db.get_user_preferences,
        description="Useful for retrieving a user's shopping preferences and history. Input should be a user ID string."
    ),
    Tool(
        name="CheckProductStock",
        func=inventory_system.check_stock,
        description="Useful for checking the current stock level of a product given its product ID. Input should be a product ID string."
    ),
    Tool(
        name="GetShippingInformation",
        func=lambda args: inventory_system.get_shipping_info(product_id=args.split(',')[0].strip(), quantity=int(args.split(',')[1].strip())),
        description="Useful for getting shipping details for a product. Input should be a comma-separated string of 'product_id, quantity'."
    ),
    Tool(
        name="AnalyzeReviewSentiment",
        func=sentiment_analyzer.analyze_sentiment,
        description="Useful for analyzing the sentiment of a given text, such as a product review. Input should be a review text string."
    ),
]

# --- LLM and Agent Setup --- 

# Replace with your actual OpenAI API Key, or use an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
llm = OpenAI(temperature=0, api_key="YOUR_OPENAI_API_KEY_HERE") # Placeholder

# Define the agent prompt
prompt = PromptTemplate.from_template("""You are a helpful e-commerce shopping assistant. 
Your goal is to assist users with their shopping queries by leveraging various tools.

Answer the following questions as best you can.

You's profile ID is 'user123'. Use this ID when requesting user-specific information.

TOOLS:
{tools}

FORMAT INSTRUCTIONS:
{format_instructions}

USER'S QUESTION:
{input}

{agent_scratchpad}""")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Main Application Loop --- 

def run_shopping_assistant():
    print("Hello! I am your personalized e-commerce shopping assistant. How can I help you today?")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nUser: ")
        if user_query.lower() == 'exit':
            print("Thank you for shopping with us! Goodbye.")
            break

        try:
            # The agent_executor will orchestrate the LLM and tool calls
            response = agent_executor.invoke({"input": user_query, "user_id": "user123"})
            print(f"Assistant: {response['output']}")
        except Exception as e:
            print(f"Assistant: An error occurred: {e}")
            print("Please try rephrasing your question or provide more details.")

if __name__ == "__main__":
    run_shopping_assistant()
