
import abc

class LLMAdapter:
    def summarize(self, text):
        return f"[Simulated LLM Summary]: {text[:50]}..."

    def generate_response(self, prompt):
        if "search for product" in prompt.lower():
            return "Intent: ProductSearch"
        elif "compare prices" in prompt.lower():
            return "Intent: PriceComparison"
        elif "summarize reviews" in prompt.lower():
            return "Intent: ReviewSummarization"
        elif "recommend" in prompt.lower():
            return "Intent: Recommendation"
        else:
            return "Intent: GeneralQuery"

class ExternalTool(abc.ABC):
    @abc.abstractmethod
    def execute(self, *args, **kwargs):
        pass

class EcommerceAPI(ExternalTool):
    def search_products(self, query, filters=None):
        print(f"[Simulated EcommerceAPI]: Searching for '{query}' with filters {filters}")
        if "laptop" in query.lower():
            return [
                {"id": "L101", "name": "Dell XPS 15", "price": 1800, "platform": "Amazon"},
                {"id": "L102", "name": "MacBook Air M2", "price": 1500, "platform": "Best Buy"}
            ]
        return []

    def get_product_details(self, product_id):
        print(f"[Simulated EcommerceAPI]: Getting details for product ID '{product_id}'")
        if product_id == "L101":
            return {"id": "L101", "name": "Dell XPS 15", "description": "Powerful laptop", "reviews_url": "http://example.com/dell-xps-reviews"}
        return None

    def execute(self, action, *args, **kwargs):
        if action == "search":
            return self.search_products(*args, **kwargs)
        elif action == "details":
            return self.get_product_details(*args, **kwargs)
        return "Unknown EcommerceAPI action"

class WebScraper(ExternalTool):
    def scrape_prices(self, product_name):
        print(f"[Simulated WebScraper]: Scraping prices for '{product_name}'")
        if "dell xps 15" in product_name.lower():
            return {"Amazon": 1800, "Newegg": 1750, "Walmart": 1820}
        return {}

    def scrape_reviews(self, product_url):
        print(f"[Simulated WebScraper]: Scraping reviews from '{product_url}'")
        if "dell-xps-reviews" in product_url:
            return [
                "Great laptop, very fast!",
                "Battery life could be better.",
                "Excellent screen quality."
            ]
        return []

    def execute(self, action, *args, **kwargs):
        if action == "scrape_prices":
            return self.scrape_prices(*args, **kwargs)
        elif action == "scrape_reviews":
            return self.scrape_reviews(*args, **kwargs)
        return "Unknown WebScraper action"

class MemoryManager:
    def __init__(self):
        self.session_data = {}
        self.user_preferences = {}

    def store(self, key, value, session_id="default"):
        if session_id not in self.session_data:
            self.session_data[session_id] = {}
        self.session_data[session_id][key] = value
        print(f"[MemoryManager]: Stored '{key}' for session '{session_id}'")

    def retrieve(self, key, session_id="default"):
        return self.session_data.get(session_id, {}).get(key)

    def update_session(self, session_id, data):
        if session_id not in self.session_data:
            self.session_data[session_id] = {}
        self.session_data[session_id].update(data)
        print(f"[MemoryManager]: Updated session '{session_id}' with data: {data}")

    def get_preferences(self, user_id="default"):
        return self.user_preferences.get(user_id, {})

    def set_preference(self, user_id, key, value):
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        self.user_preferences[user_id][key] = value
        print(f"[MemoryManager]: Set preference '{key}' for user '{user_id}'")

class BaseAgent(abc.ABC):
    def __init__(self, name, description, tools=None):
        self.name = name
        self.description = description
        self.tools = tools if tools is not None else []

    @abc.abstractmethod
    def can_handle(self, query, session_data):
        pass

    @abc.abstractmethod
    def execute(self, query, session_data):
        pass

class ProductSearchAgent(BaseAgent):
    def __init__(self, ecommerce_api):
        super().__init__("ProductSearchAgent", "Searches for products on e-commerce platforms.", tools=[ecommerce_api])
        self.ecommerce_api = ecommerce_api

    def can_handle(self, query, session_data):
        return "search for" in query.lower() or "find product" in query.lower()

    def execute(self, query, session_data):
        search_term = query.replace("search for", "").replace("find product", "").strip()
        products = self.ecommerce_api.execute("search", query=search_term)
        if products:
            return "Found the following products: " + ", ".join([p['name'] for p in products])
        return "Could not find any products matching your query."

class PriceComparisonAgent(BaseAgent):
    def __init__(self, ecommerce_api, web_scraper):
        super().__init__("PriceComparisonAgent", "Compares prices across different retailers.", tools=[ecommerce_api, web_scraper])
        self.ecommerce_api = ecommerce_api
        self.web_scraper = web_scraper

    def can_handle(self, query, session_data):
        return "compare prices" in query.lower() or "how much does" in query.lower()

    def execute(self, query, session_data):
        product_name = query.replace("compare prices for", "").replace("how much does", "").replace("cost", "").strip()
        prices = self.web_scraper.execute("scrape_prices", product_name=product_name)
        if prices:
            price_str = ", ".join([f"{platform}: ${price}" for platform, price in prices.items()])
            return f"Prices for {product_name}: {price_str}"
        return "Could not compare prices for this product."

class ReviewSummarizationAgent(BaseAgent):
    def __init__(self, web_scraper, llm_adapter):
        super().__init__("ReviewSummarizationAgent", "Summarizes product reviews.", tools=[web_scraper, llm_adapter])
        self.web_scraper = web_scraper
        self.llm_adapter = llm_adapter

    def can_handle(self, query, session_data):
        return "summarize reviews" in query.lower() or "tell me about reviews" in query.lower()

    def execute(self, query, session_data):
        product_id = session_data.get("last_product_id")
        if not product_id:
            return "Please search for a product first to get reviews."

        ecommerce_api = next((tool for tool in self.tools if isinstance(tool, EcommerceAPI)), None)
        if not ecommerce_api:
            return "Error: EcommerceAPI tool not available for review summarization."

        product_details = ecommerce_api.get_product_details(product_id)
        if product_details and "reviews_url" in product_details:
            reviews = self.web_scraper.execute("scrape_reviews", product_url=product_details["reviews_url"])
            if reviews:
                combined_reviews = " ".join(reviews)
                summary = self.llm_adapter.summarize(combined_reviews)
                return f"Here's a summary of reviews for {product_details['name']}: {summary}"
            return "No reviews found for this product."
        return "Could not retrieve product details or review URL."

class RecommendationAgent(BaseAgent):
    def __init__(self, memory_manager, ecommerce_api):
        super().__init__("RecommendationAgent", "Provides product recommendations.", tools=[memory_manager, ecommerce_api])
        self.memory_manager = memory_manager
        self.ecommerce_api = ecommerce_api

    def can_handle(self, query, session_data):
        return "recommend" in query.lower() or "suggest something" in query.lower()

    def execute(self, query, session_data):
        user_preferences = self.memory_manager.get_preferences(session_data.get("user_id", "default"))
        last_search_term = session_data.get("last_search_term", "")

        if "favorite_category" in user_preferences:
            recommend_query = user_preferences["favorite_category"]
        elif last_search_term:
            recommend_query = last_search_term
        else:
            return "I need more information about your preferences or recent searches to provide recommendations."

        products = self.ecommerce_api.execute("search", query=f"{recommend_query} recommendations")
        if products:
            return "Based on your interests, I recommend: " + ", ".join([p['name'] for p in products[:2]])
        return "Could not find recommendations at this time."

class TaskOrchestrator:
    def __init__(self, agents, llm_adapter):
        self.agents = agents
        self.llm_adapter = llm_adapter

    def process_user_query(self, query, session_data):
        # Simulate intent recognition using LLMAdapter
        intent = self.llm_adapter.generate_response(query).replace("Intent: ", "").strip()
        print(f"[TaskOrchestrator]: Detected intent: {intent}")

        for agent in self.agents:
            if agent.can_handle(query, session_data) or (intent == agent.name.replace("Agent", "").strip() and agent.name != "BaseAgent"):
                print(f"[TaskOrchestrator]: Dispatching to {agent.name}")
                result = agent.execute(query, session_data)
                if isinstance(agent, ProductSearchAgent) and "Found the following products" in result:
                    # Extract product ID from a simulated product list for follow-up actions
                    if "Dell XPS 15" in result:
                        session_data["last_product_id"] = "L101"
                return result
        return "I'm sorry, I couldn't understand your request or find a suitable agent."

class IntelligentShoppingAssistant:
    def __init__(self):
        self.llm_adapter = LLMAdapter()
        self.ecommerce_api = EcommerceAPI()
        self.web_scraper = WebScraper()
        self.memory_manager = MemoryManager()

        self.agents = [
            ProductSearchAgent(self.ecommerce_api),
            PriceComparisonAgent(self.ecommerce_api, self.web_scraper),
            ReviewSummarizationAgent(self.web_scraper, self.llm_adapter),
            RecommendationAgent(self.memory_manager, self.ecommerce_api)
        ]
        self.orchestrator = TaskOrchestrator(self.agents, self.llm_adapter)
        self.session_data = {"session_id": "user_123", "user_id": "user_123"}

    def start_conversation(self):
        print("Hello! I'm your Intelligent Shopping Assistant. How can I help you today?")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            response = self.handle_input(user_input)
            print(f"Assistant: {response}")

    def handle_input(self, user_input):
        response = self.orchestrator.process_user_query(user_input, self.session_data)
        return response

if __name__ == "__main__":
    assistant = IntelligentShoppingAssistant()
    assistant.start_conversation()
