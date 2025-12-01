
class User:
    def __init__(self, user_id, preferences):
        self.user_id = user_id
        self.preferences = preferences # e.g., {'fav_brands': ['Nike', 'Adidas'], 'budget': {'min': 50, 'max': 200}, 'style': 'casual', 'fav_platforms': ['Amazon', 'Zappos']}
        self.history = [] # e.g., [{'product_id': 'P101', 'action': 'viewed', 'timestamp': '...'}, {'product_id': 'P102', 'action': 'purchased', 'timestamp': '...'}]

class Product:
    def __init__(self, product_id, name, brand, price, category, color, style, platform, description):
        self.product_id = product_id
        self.name = name
        self.brand = brand
        self.price = price
        self.category = category
        self.color = color
        self.style = style
        self.platform = platform
        self.description = description

    def __repr__(self):
        return f"Product(ID: {self.product_id}, Name: {self.name}, Brand: {self.brand}, Price: ${self.price}, Platform: {self.platform})"

class PersonalizedShoppingAssistant:
    def __init__(self, user: User, product_catalog: list[Product]):
        self.user = user
        self.product_catalog = product_catalog
        self._model_user_info() # Initialize user preferences from the provided user object

    def _model_user_info(self):
        """Simulates modeling diverse user information. In a real system, this would involve NLP, embeddings, etc."""
        print(f"[User Info Modeling] Initializing preferences for user {self.user.user_id}: {self.user.preferences}")
        # In a real system, this would process raw data into a unified semantic space.
        # For this simulation, we assume user.preferences is already structured.

    def _personalized_tool_planning(self, query: str):
        """Decides which tool (search, filter, compare) to use based on query and user preferences."""
        plan = []
        if 'search' in query.lower() or 'find' in query.lower():
            plan.append('search')
        if 'filter' in query.lower() or 'show me' in query.lower():
            plan.append('filter')
        # Add more sophisticated planning based on user preferences here
        # For example, if user prefers 'deals', plan to use a 'deals' tool
        return plan if plan else ['search'] # Default to search if no specific tool mentioned

    def _personalized_tool_call(self, tool_name: str, parameters: dict):
        """Generates personalized inputs for a given tool and simulates its execution."""
        print(f"[Tool Call] Executing '{tool_name}' with personalized parameters: {parameters}")
        
        # Simulate tool execution based on tool_name
        if tool_name == 'search':
            return self._execute_search_tool(**parameters)
        elif tool_name == 'filter':
            return self._execute_filter_tool(**parameters)
        else:
            print(f"[Tool Call] Unknown tool: {tool_name}")
            return []

    def _execute_search_tool(self, keyword: str, platform: str = None):
        """Simulates searching the product catalog with personalization."""
        results = []
        for product in self.product_catalog:
            if keyword.lower() in product.name.lower() or keyword.lower() in product.description.lower():
                if platform and product.platform.lower() != platform.lower():
                    continue
                results.append(product)
        return results

    def _execute_filter_tool(self, products: list[Product], brand: str = None, price_range: tuple = None, category: str = None, color: str = None, style: str = None, platform: str = None):
        """Simulates filtering products with personalization."""
        filtered_products = []
        for product in products:
            match = True
            if brand and product.brand.lower() != brand.lower():
                match = False
            if price_range and not (price_range[0] <= product.price <= price_range[1]):
                match = False
            if category and product.category.lower() != category.lower():
                match = False
            if color and product.color.lower() != color.lower():
                match = False
            if style and product.style.lower() != style.lower():
                match = False
            if platform and product.platform.lower() != platform.lower():
                match = False
            
            if match:
                filtered_products.append(product)
        return filtered_products

    def get_personalized_recommendations(self, query: str):
        """Provides personalized recommendations based on the query and user preferences."""
        print(f"\n--- Generating personalized recommendations for '{query}' ---")
        
        # 1. Personalized Tool Planning
        plan = self._personalized_tool_planning(query)
        print(f"[Tool Planning] Generated plan: {plan}")

        initial_results = self.product_catalog # Start with all products

        # 2. Personalized Tool Call & Execution
        for tool in plan:
            if tool == 'search':
                # Extract keyword from query (simple parsing for demonstration)
                keyword = next((word for word in query.split() if word.lower() not in ['search', 'find', 'for', 'a', 'an']), '')
                
                # Personalize platform preference
                preferred_platform = self.user.preferences.get('fav_platforms', [None])[0] # Use first preferred platform or None

                search_params = {'keyword': keyword, 'platform': preferred_platform}
                initial_results = self._personalized_tool_call('search', search_params)
            elif tool == 'filter':
                filter_params = {}
                # Apply user's budget preferences
                if self.user.preferences.get('budget'):
                    filter_params['price_range'] = (self.user.preferences['budget']['min'], self.user.preferences['budget']['max'])
                # Apply user's favorite brands
                if self.user.preferences.get('fav_brands'):
                    # For simplicity, if multiple brands, we'll try to filter by the first one that matches the current query context
                    # A real system would have more complex brand matching logic
                    pass # Filtering by brand directly here is tricky without specific query mentioning it. Handled later in ranking.
                
                # Simple extraction of color/style from query
                if 'blue' in query.lower(): filter_params['color'] = 'blue'
                if 'casual' in query.lower(): filter_params['style'] = 'casual'

                initial_results = self._personalized_tool_call('filter', {'products': initial_results, **filter_params})

        # 3. Apply further personalization for ranking and final recommendations
        final_recommendations = []
        for product in initial_results:
            score = 0
            # Prefer favorite brands
            if product.brand in self.user.preferences.get('fav_brands', []):
                score += 5
            # Prefer preferred platforms
            if product.platform in self.user.preferences.get('fav_platforms', []):
                score += 3
            # Prefer items within budget (already filtered, but can add score for being closer to ideal price within budget)
            if self.user.preferences.get('budget'):
                min_b, max_b = self.user.preferences['budget']['min'], self.user.preferences['budget']['max']
                if min_b <= product.price <= max_b:
                    score += 1
            
            final_recommendations.append((product, score))
        
        # Sort by score (higher is better)
        final_recommendations.sort(key=lambda x: x[1], reverse=True)

        return [prod for prod, score in final_recommendations]

    def proactive_suggestion(self, trigger_event: str):
        """Simulates a proactive system offering suggestions based on events."""
        print(f"\n--- Proactive System Triggered by: '{trigger_event}' ---")
        suggestions = []
        
        if "new arrivals" in trigger_event.lower() or "sale" in trigger_event.lower():
            print("[Proactive] Checking for new items or sales based on user preferences...")
            # Simulate finding new items from favorite brands or within budget
            for product in self.product_catalog:
                is_new_or_on_sale = True # This would come from an actual API in a real system
                if is_new_or_on_sale:
                    if product.brand in self.user.preferences.get('fav_brands', []) and \
                       self.user.preferences.get('budget', {}).get('min', 0) <= product.price <= self.user.preferences.get('budget', {}).get('max', float('inf')):
                        suggestions.append(product)
            
            if suggestions:
                print(f"[Proactive] Found {len(suggestions)} new/sale items tailored to your preferences:")
                for s in suggestions[:3]: # Show top 3 proactive suggestions
                    print(f"  - {s.name} ({s.brand}) - ${s.price} on {s.platform}")
            else:
                print("[Proactive] No new/sale items found matching your specific preferences at this time.")

        elif "browsing" in trigger_event.lower() and "shoes" in trigger_event.lower():
            print("[Proactive] Noticed you're browsing shoes. Here are some popular styles from your preferred brands:")
            shoe_suggestions = []
            for product in self.product_catalog:
                if product.category.lower() == 'shoes' and product.brand in self.user.preferences.get('fav_brands', []):
                    shoe_suggestions.append(product)
            
            if shoe_suggestions:
                for s in shoe_suggestions[:2]:
                    print(f"  - {s.name} ({s.brand}) - ${s.price} on {s.platform}")
            else:
                print("[Proactive] No shoe suggestions from your favorite brands at this moment.")
        
        return suggestions

# --- Example Usage ---
if __name__ == "__main__":
    # Mock Product Catalog
    mock_products = [
        Product('P101', 'Running Shoes', 'Nike', 120.0, 'Shoes', 'Black', 'sporty', 'Amazon', 'Comfortable running shoes.'),
        Product('P102', 'Casual T-Shirt', 'Adidas', 35.0, 'Apparel', 'White', 'casual', 'Zappos', 'Soft cotton t-shirt.'),
        Product('P103', 'Denim Jeans', 'Leviathan', 80.0, 'Apparel', 'Blue', 'casual', 'Amazon', 'Classic denim jeans.'),
        Product('P104', 'Formal Dress', 'Zara', 150.0, 'Apparel', 'Red', 'elegant', 'Zappos', 'Elegant evening dress.'),
        Product('P105', 'Sports Watch', 'Garmin', 250.0, 'Electronics', 'Silver', 'sporty', 'Amazon', 'Advanced GPS sports watch.'),
        Product('P106', 'Casual Sneakers', 'Nike', 90.0, 'Shoes', 'White', 'casual', 'Zappos', 'Stylish everyday sneakers.'),
        Product('P107', 'Graphic Tee', 'Adidas', 40.0, 'Apparel', 'Black', 'streetwear', 'Amazon', 'Trendy graphic t-shirt.'),
        Product('P108', 'Smartwatch', 'Apple', 399.0, 'Electronics', 'Space Gray', 'modern', 'BestBuy', 'Latest Apple Smartwatch.'),
        Product('P109', 'Hiking Boots', 'Timberwolf', 180.0, 'Shoes', 'Brown', 'outdoor', 'REI', 'Durable hiking boots.'),
        Product('P110', 'Summer Dress', 'H&M', 45.0, 'Apparel', 'Yellow', 'casual', 'Zappos', 'Light and breezy summer dress.'),
        Product('P111', 'Designer Bag', 'Gucci', 1200.0, 'Accessories', 'Black', 'luxury', 'Farfetch', 'High-end designer handbag.'),
    ]

    # Example User Profile
    user1_preferences = {
        'fav_brands': ['Nike', 'Adidas', 'Zappos'], # Zappos is a platform, but user might 'prefer' buying from it
        'budget': {'min': 30, 'max': 150},
        'style': 'casual',
        'fav_platforms': ['Amazon', 'Zappos']
    }
    user1 = User('user_alice', user1_preferences)

    user2_preferences = {
        'fav_brands': ['Apple', 'Garmin'],
        'budget': {'min': 200, 'max': 500},
        'style': 'sporty',
        'fav_platforms': ['BestBuy', 'Amazon']
    }
    user2 = User('user_bob', user2_preferences)

    # Initialize the assistant for User Alice
    assistant_alice = PersonalizedShoppingAssistant(user1, mock_products)

    # Test Personalized Recommendations for Alice
    print("\n--- Alice's Personalized Shopping Experience ---")
    recommendations_alice_1 = assistant_alice.get_personalized_recommendations("Find me some casual white shoes.")
    print("\nRecommendations for Alice (Casual White Shoes):")
    for r in recommendations_alice_1:
        print(f"- {r.name} ({r.brand}) - ${r.price} on {r.platform}")

    recommendations_alice_2 = assistant_alice.get_personalized_recommendations("Search for a blue dress.")
    print("\nRecommendations for Alice (Blue Dress):")
    for r in recommendations_alice_2:
        print(f"- {r.name} ({r.brand}) - ${r.price} on {r.platform}")
    
    # Test Proactive System for Alice
    assistant_alice.proactive_suggestion("New arrivals from Nike are out!")
    assistant_alice.proactive_suggestion("User has been browsing shoes for 10 minutes.")

    # Initialize the assistant for User Bob
    assistant_bob = PersonalizedShoppingAssistant(user2, mock_products)

    # Test Personalized Recommendations for Bob
    print("\n--- Bob's Personalized Shopping Experience ---")
    recommendations_bob_1 = assistant_bob.get_personalized_recommendations("Show me a sports watch.")
    print("\nRecommendations for Bob (Sports Watch):")
    for r in recommendations_bob_1:
        print(f"- {r.name} ({r.brand}) - ${r.price} on {r.platform}")

    recommendations_bob_2 = assistant_bob.get_personalized_recommendations("Find me an Apple smartwatch.")
    print("\nRecommendations for Bob (Apple Smartwatch):")
    for r in recommendations_bob_2:
        print(f"- {r.name} ({r.brand}) - ${r.price} on {r.platform}")
    
    # Test Proactive System for Bob
    assistant_bob.proactive_suggestion("Sale on Garmin products!")
    assistant_bob.proactive_suggestion("User just viewed a new high-end electronic device.")
