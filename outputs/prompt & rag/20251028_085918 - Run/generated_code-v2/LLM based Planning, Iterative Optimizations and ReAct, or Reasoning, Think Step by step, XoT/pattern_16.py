import random

class ECommerceEnvironment:
    def __init__(self):
        self.products = [
            {"id": "P001", "name": "Wireless Bluetooth Headphones", "category": "Electronics", "price": 79.99, "stock": 10, "reviews": 4.5},
            {"id": "P002", "name": "Ergonomic Office Chair", "category": "Home & Office", "price": 249.00, "stock": 5, "reviews": 4.8},
            {"id": "P003", "name": "Smart LED TV 55 inch", "category": "Electronics", "price": 699.50, "stock": 3, "reviews": 4.2},
            {"id": "P004", "name": "Stainless Steel Water Bottle", "category": "Kitchen & Dining", "price": 19.99, "stock": 50, "reviews": 4.7},
            {"id": "P005", "name": "Gaming Laptop RTX 3060", "category": "Electronics", "price": 1200.00, "stock": 0, "reviews": 4.6},
            {"id": "P006", "name": "Running Shoes Men's", "category": "Apparel", "price": 85.00, "stock": 20, "reviews": 4.3},
            {"id": "P007", "name": "Coffee Maker Drip", "category": "Kitchen & Dining", "price": 45.00, "stock": 12, "reviews": 4.0},
            {"id": "P008", "name": "Noise Cancelling Earbuds", "category": "Electronics", "price": 129.00, "stock": 8, "reviews": 4.4},
            {"id": "P009", "name": "Yoga Mat Eco-Friendly", "category": "Sports & Outdoors", "price": 30.00, "stock": 15, "reviews": 4.9},
            {"id": "P010", "name": "External SSD 1TB", "category": "Electronics", "price": 99.00, "stock": 0, "reviews": 4.7},
        ]

    def search_products(self, query: str):
        query_lower = query.lower()
        results = []
        for product in self.products:
            if query_lower in product["name"].lower() or \
               query_lower in product["category"].lower() or \
               (query_lower.replace(" ", "") in product["name"].lower().replace(" ", "") and len(query_lower) > 2):
                results.append(product)
        return results

    def get_product_details(self, product_id: str):
        for product in self.products:
            if product["id"] == product_id:
                return product
        return None

class PersonalShopperAgent:
    def __init__(self, environment: ECommerceEnvironment):
        self.environment = environment
        self.conversation_history = []

    def _understand_intent(self, user_query: str):
        intent = {"search_term": user_query}
        if "cheap" in user_query.lower() or "affordable" in user_query.lower():
            intent["price_constraint"] = "low"
        elif "expensive" in user_query.lower() or "premium" in user_query.lower():
            intent["price_constraint"] = "high"
        if "stock" in user_query.lower() or "available" in user_query.lower():
            intent["availability_constraint"] = "in_stock"
        return intent

    def _generate_initial_plan(self, intent: dict):
        search_query = intent.get("search_term", "")
        return search_query

    def _execute_plan(self, search_query: str):
        return self.environment.search_products(search_query)

    def _observe_feedback(self, search_results: list, current_query: str):
        feedback = {
            "found_products": len(search_results) > 0,
            "too_many_results": len(search_results) > 5,
            "product_categories": set(),
            "price_range": [float('inf'), float('-inf')],
            "in_stock_count": 0,
            "out_of_stock_count": 0,
            "all_out_of_stock": True
        }
        if feedback["found_products"]:
            for product in search_results:
                feedback["product_categories"].add(product["category"])
                if product["price"] < feedback["price_range"][0]:
                    feedback["price_range"][0] = product["price"]
                if product["price"] > feedback["price_range"][1]:
                    feedback["price_range"][1] = product["price"]
                if product["stock"] > 0:
                    feedback["in_stock_count"] += 1
                    feedback["all_out_of_stock"] = False
                else:
                    feedback["out_of_stock_count"] += 1
        return feedback

    def _reason_and_adjust_plan(self, current_query: str, feedback: dict, user_feedback: str = None):
        new_query = current_query
        if user_feedback:
            user_feedback_lower = user_feedback.lower()
            if "cheaper" in user_feedback_lower or "less expensive" in user_feedback_lower:
                new_query += " cheap"
            elif "more expensive" in user_feedback_lower or "premium" in user_feedback_lower:
                new_query += " premium"
            elif "category" in user_feedback_lower:
                categories = list(feedback["product_categories"])
                if categories:
                    new_query += f" in {categories[0]}"
            elif "in stock" in user_feedback_lower or "available" in user_feedback_lower:
                new_query += " in stock"
            elif "out of stock" in user_feedback_lower:
                new_query = new_query.replace(" in stock", "")

        if not feedback["found_products"]:
            print("No products found. Trying a broader search...")
            new_query = "product"
        elif feedback["too_many_results"] and not user_feedback:
            print("Many results found. Can you specify a category or price range?")
            if feedback["product_categories"]:
                new_query += f" in {list(feedback["product_categories"])[0]}"
            elif feedback["price_range"][0] != float('inf'):
                new_query += f" under ${feedback["price_range"][1] + 1}"
        elif feedback["all_out_of_stock"] and not user_feedback:
            print("All found products are out of stock. Looking for available alternatives.")
            new_query += " in stock"

        return new_query.strip()

    def chat(self, user_query: str, max_iterations: int = 5):
        print(f"Hello! I'm your personal shopper assistant. Let's find some products for you.")
        print(f"Initial query: {user_query}")

        intent = self._understand_intent(user_query)
        current_search_query = self._generate_initial_plan(intent)
        self.conversation_history.append((user_query, current_search_query))

        for i in range(max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            print(f"Executing plan with query: '{current_search_query}'")
            search_results = self._execute_plan(current_search_query)
            feedback = self._observe_feedback(search_results, current_search_query)

            if feedback["found_products"]:
                print("Found the following products:")
                for product in search_results:
                    print(f"  - {product['name']} ({product['category']}) - ${product['price']:.2f} (Stock: {product['stock']}, Reviews: {product['reviews']})")
            else:
                print("No products found for your query.")

            user_input = input("Are you satisfied, or would you like to refine the search (e.g., 'cheaper', 'in stock', 'in electronics')? (type 'yes' to finish): ").strip().lower()

            if user_input == "yes":
                print("Great! Happy shopping!")
                break
            else:
                previous_query = current_search_query
                current_search_query = self._reason_and_adjust_plan(current_search_query, feedback, user_input)
                if previous_query == current_search_query:
                    print("Could not refine the search further with your input. Please try a different input or 'yes' to finish.")

        else:
            print("Max iterations reached. I hope I was able to assist you!")

if __name__ == "__main__":
    environment = ECommerceEnvironment()
    agent = PersonalShopperAgent(environment)
    agent.chat("I am looking for headphones")
    print("\n---")
    agent.chat("Show me a cheap office chair")
    print("\n---")
    agent.chat("gaming laptop in stock")
    print("\n---")
    agent.chat("water bottle")