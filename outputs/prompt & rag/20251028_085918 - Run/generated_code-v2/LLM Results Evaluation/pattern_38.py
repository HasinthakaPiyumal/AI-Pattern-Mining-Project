import re

class ProductDatabase:
    def __init__(self):
        self.products = [
            {"id": "P001", "name": "Smartphone X", "category": "Electronics", "description": "A high-end smartphone with a great camera."},
            {"id": "P002", "name": "Laptop Pro", "category": "Electronics", "description": "Powerful laptop for professionals and gamers."},
            {"id": "P003", "name": "Wireless Earbuds Z", "category": "Audio", "description": "Noise-cancelling earbuds with long battery life."},
            {"id": "P004", "name": "Smart Watch S", "category": "Wearable Tech", "description": "Fitness tracker and smart notifications."},               
            {"id": "P005", "name": "Coffee Maker Deluxe", "category": "Home Appliances", "description": "Automatic coffee maker with multiple brewing options."},            
        ]

    def get_product_by_id(self, product_id):
        for product in self.products:
            if product["id"] == product_id:
                return product
        return None

    def search_products_by_keyword(self, keyword):
        found_products = []
        keyword_lower = keyword.lower()
        for product in self.products:
            if keyword_lower in product["name"].lower() or \
               keyword_lower in product["category"].lower() or \
               keyword_lower in product["description"].lower():
                found_products.append(product)
        return found_products

class LLMService:
    def generate_response(self, prompt_template, customer_query):
        # This is a simulated LLM response. In a real application, this would call an actual LLM API.
        full_prompt = prompt_template.format(query=customer_query)
        if "recommend a product" in full_prompt.lower() or "looking for" in full_prompt.lower():
            if "smartphone" in customer_query.lower() or "phone" in customer_query.lower():
                return f"Based on your query '{customer_query}', here are some recommendations: Smartphone X (P001). It's a high-end device. You might also like Laptop Pro (P002) if you need computing."
            elif "laptop" in customer_query.lower():
                return f"For '{customer_query}', consider Laptop Pro (P002), a powerful machine for various tasks. Also, check out Wireless Earbuds Z (P003) for audio needs."
            elif "earbuds" in customer_query.lower() or "audio" in customer_query.lower():
                return f"Looking for '{customer_query}'? Wireless Earbuds Z (P003) are great for noise-cancelling. Maybe a Smart Watch S (P004) too?"
            elif "coffee" in customer_query.lower():
                 return f"For '{customer_query}', we recommend Coffee Maker Deluxe (P005), a top-tier appliance."
            else:
                return f"For your query '{customer_query}', our LLM suggests general electronics like Smartphone X (P001) or Laptop Pro (P002)."
        return f"I received your request: '{full_prompt}'. My LLM output is a general statement about helpfulness."

class MutualInformationScorer:
    def __init__(self, product_db):
        self.product_db = product_db

    def score_response(self, llm_response, customer_query):
        score = 0
        query_keywords = re.findall(r'\b\w+\b', customer_query.lower())
        response_lower = llm_response.lower()

        # Check for direct query keyword presence in response
        for keyword in query_keywords:
            if len(keyword) > 2 and keyword in response_lower:
                score += 1
        
        # Check for product names and categories mentioned in the response
        for product in self.product_db.products:
            product_name_lower = product["name"].lower()
            product_category_lower = product["category"].lower()
            if product_name_lower in response_lower:
                score += 2  # Higher score for explicit product mention
            if product_category_lower in response_lower:
                score += 1
        
        # Simple check for product IDs (e.g., P001)
        product_id_matches = re.findall(r'P\d{3}', llm_response)
        score += len(product_id_matches) * 2

        return score

class PromptTemplateOptimizer:
    def __init__(self, llm_service, mi_scorer, prompt_templates):
        self.llm_service = llm_service
        self.mi_scorer = mi_scorer
        self.prompt_templates = prompt_templates

    def optimize_template(self, customer_query):
        best_template = None
        max_score = -1
        
        for template in self.prompt_templates:
            llm_response = self.llm_service.generate_response(template, customer_query)
            score = self.mi_scorer.score_response(llm_response, customer_query)
            
            if score > max_score:
                max_score = score
                best_template = template

        return best_template

class CustomerSupportChatbot:
    def __init__(self, product_db, llm_service, mi_scorer, prompt_templates):
        self.product_db = product_db
        self.llm_service = llm_service
        self.mi_scorer = mi_scorer
        self.prompt_optimizer = PromptTemplateOptimizer(llm_service, mi_scorer, prompt_templates)

    def get_recommendations(self, customer_query):
        print(f"\nCustomer Query: {customer_query}")
        
        # Optimize the prompt template
        optimal_template = self.prompt_optimizer.optimize_template(customer_query)
        print(f"Optimal Prompt Template Selected: {optimal_template}")
        
        # Generate final recommendation using the optimal template
        final_prompt = optimal_template.format(query=customer_query)
        final_recommendation = self.llm_service.generate_response(optimal_template, customer_query) # Pass template and query separately to LLMService
        
        return final_recommendation

if __name__ == "__main__":
    # Initialize components
    product_db = ProductDatabase()
    llm_service = LLMService()
    mi_scorer = MutualInformationScorer(product_db)

    # Define multiple prompt templates
    prompt_templates = [
        "Please recommend a product based on the following customer query: {query}",
        "What kind of products would you suggest for a user asking about: {query}? Provide specific product IDs if possible.",
        "I need product suggestions for '{query}'. Give me the top 2 relevant items.",
        "Help me find products related to: {query}.",
    ]

    chatbot = CustomerSupportChatbot(product_db, llm_service, mi_scorer, prompt_templates)

    # Simulate customer interactions
    queries = [
        "I am looking for a new smartphone.",
        "Do you have any powerful laptops?",
        "What kind of audio devices do you recommend?",
        "I need a coffee machine.",
        "Something for fitness tracking.",
        "Tell me about electronics."
    ]

    for query in queries:
        recommendations = chatbot.get_recommendations(query)
        print(f"Chatbot Recommendation: {recommendations}")
        print("--------------------------------------------------")
