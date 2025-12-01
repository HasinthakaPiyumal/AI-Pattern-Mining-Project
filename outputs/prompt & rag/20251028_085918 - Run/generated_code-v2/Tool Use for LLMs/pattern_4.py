import abc

class UserProfile:
    """Models diverse user information including preferences, history, and style."""
    def __init__(self, user_id: str, preferences: dict = None, shopping_history: list = None):
        self.user_id = user_id
        self.preferences = preferences if preferences is not None else {}
        self.shopping_history = shopping_history if shopping_history is not None else []

    def update_preferences(self, new_preferences: dict):
        self.preferences.update(new_preferences)
        print(f"User {self.user_id} preferences updated: {self.preferences}")

    def add_to_history(self, item_searched_or_purchased: str):
        self.shopping_history.append(item_searched_or_purchased)
        print(f"User {self.user_id} history updated with: {item_searched_or_purchased}")

    def __str__(self):
        return f"UserProfile(ID: {self.user_id}, Prefs: {self.preferences}, History: {self.shopping_history})"


class ECommerceTool(abc.ABC):
    """Abstract base class for any e-commerce platform tool."""
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def search_product(self, query: str) -> list:
        pass

    @abc.abstractmethod
    def get_deals(self, category: str = None) -> list:
        pass


class AmazonTool(ECommerceTool):
    """Simulated Amazon tool with basic search and deals functionality."""
    def __init__(self):
        super().__init__("Amazon")
        self.products = {
            "laptop": ["Dell XPS 13", "MacBook Air"],
            "headphone": ["Sony WH-1000XM4", "Bose QuietComfort 35 II"],
            "t-shirt": ["Graphic Tee A", "Plain Cotton Tee B"]
        }
        self.deals = {
            "electronics": ["20% off selected laptops"],
            "fashion": ["15% off summer collection"]
        }

    def search_product(self, query: str) -> list:
        print(f"[{self.name}] Searching for: {query}")
        results = [p for k, v in self.products.items() for p in v if query.lower() in k.lower() or query.lower() in p.lower()]
        return results if results else [f"No product found for '{query}' on {self.name}"]

    def get_deals(self, category: str = None) -> list:
        print(f"[{self.name}] Getting deals for category: {category}")
        return self.deals.get(category.lower(), [f"No deals for '{category}' on {self.name}"])


class EbayTool(ECommerceTool):
    """Simulated eBay tool with basic search and deals functionality."""
    def __init__(self):
        super().__init__("eBay")
        self.products = {
            "vintage camera": ["Canon AE-1 Program", "Nikon F2"],
            "collectible coin": ["1909 VDB Penny", "Silver Eagle"],
            "laptop": ["Used ThinkPad X1 Carbon", "Refurbished MacBook Pro"]
        }
        self.deals = {
            "collectibles": ["10% off rare coins"],
            "electronics": ["Free shipping on refurbished laptops"]
        }

    def search_product(self, query: str) -> list:
        print(f"[{self.name}] Searching for: {query}")
        results = [p for k, v in self.products.items() for p in v if query.lower() in k.lower() or query.lower() in p.lower()]
        return results if results else [f"No product found for '{query}' on {self.name}"]

    def get_deals(self, category: str = None) -> list:
        print(f"[{self.name}] Getting deals for category: {category}")
        return self.deals.get(category.lower(), [f"No deals for '{category}' on {self.name}"])


class ToolPlanner:
    """Develops tool execution plans and selects tools based on individual user preferences."""
    def __init__(self, available_tools: list[ECommerceTool]):
        self.available_tools = {tool.name.lower(): tool for tool in available_tools}

    def plan_tools(self, user_profile: UserProfile, shopping_goal: str) -> list[tuple[ECommerceTool, str, str]]:
        """Determines which tools to use and what type of action (search/deals) based on goal and preferences."""
        plan = []
        preferred_platform = user_profile.preferences.get("preferred_platform", "amazon").lower()
        print(f"\nPlanning tools for goal '{shopping_goal}' with preferred platform: {preferred_platform}")

        # Simple planning logic: if a preferred platform exists and is available, use it first.
        # Otherwise, try a default or all available.
        tool_to_use = self.available_tools.get(preferred_platform) or self.available_tools.get("amazon")

        if "search for" in shopping_goal.lower():
            query = shopping_goal.replace("search for", "").strip()
            if tool_to_use:
                plan.append((tool_to_use, "search", query))
            else:
                print("No suitable tool found for the preferred platform. Trying all available tools.")
                for tool_name, tool_obj in self.available_tools.items():
                    plan.append((tool_obj, "search", query))

        elif "find deals for" in shopping_goal.lower():
            category = shopping_goal.replace("find deals for", "").strip()
            if tool_to_use:
                plan.append((tool_to_use, "get_deals", category))
            else:
                print("No suitable tool found for the preferred platform. Trying all available tools.")
                for tool_name, tool_obj in self.available_tools.items():
                    plan.append((tool_obj, "get_deals", category))
        else:
            print(f"Could not understand shopping goal: {shopping_goal}. Defaulting to search.")
            query = shopping_goal
            if tool_to_use:
                plan.append((tool_to_use, "search", query))
            else:
                for tool_name, tool_obj in self.available_tools.items():
                    plan.append((tool_obj, "search", query))

        return plan

    def generate_personalized_input(self, user_profile: UserProfile, tool_name: str, original_query: str) -> str:
        """Generates different inputs for tools based on user preferences."""
        personalized_query = original_query
        user_style = user_profile.preferences.get("language_style", "formal")
        preferred_brand = user_profile.preferences.get("preferred_brand")
        max_price = user_profile.preferences.get("max_price")

        if preferred_brand and preferred_brand.lower() not in personalized_query.lower():
            personalized_query = f"{preferred_brand} {personalized_query}"
        if max_price:
            # This is a simplification; a real tool would parse price filters.
            if "under" not in personalized_query.lower() and "less than" not in personalized_query.lower():
                personalized_query = f"{personalized_query} under ${max_price}"

        # Example of language style personalization (simplified)
        if user_style == "casual" and "search for" in personalized_query.lower():
            personalized_query = personalized_query.replace("search for", "find me some")

        print(f"Generating personalized input for {tool_name}: Original='{original_query}', Personalized='{personalized_query}'")
        return personalized_query


class PersonalizedShoppingAssistant:
    """The main AI assistant integrating all components for a personalized experience."""
    def __init__(self, user_profile: UserProfile, available_tools: list[ECommerceTool]):
        self.user_profile = user_profile
        self.tool_planner = ToolPlanner(available_tools)
        self.available_tools = {tool.name.lower(): tool for tool in available_tools}
        print(f"\nPersonalized Shopping Assistant initialized for user: {user_profile.user_id}")

    def assist(self, shopping_goal: str):
        """Handles a shopping request, plans, executes, and updates user history."""
        print(f"\nUser Request: '{shopping_goal}'")
        plan = self.tool_planner.plan_tools(self.user_profile, shopping_goal)
        all_results = []

        if not plan:
            print("No plan could be generated for your request.")
            return all_results

        for tool_obj, action_type, query_or_category in plan:
            personalized_input = self.tool_planner.generate_personalized_input(
                self.user_profile, tool_obj.name, query_or_category
            )
            if action_type == "search":
                results = tool_obj.search_product(personalized_input)
            elif action_type == "get_deals":
                results = tool_obj.get_deals(personalized_input)
            else:
                results = [f"Unknown action type: {action_type}"]

            all_results.extend(results)
            print(f"Results from {tool_obj.name}: {results}")

            # Update user history with the query/category used
            self.user_profile.add_to_history(f"{action_type}: {query_or_category}")
        
        print(f"\n--- Overall Results for '{shopping_goal}' ---")
        for result in all_results:
            print(f"- {result}")
        print("-------------------------------------")
        return all_results

    def proactive_suggestions(self):
        """Provides proactive suggestions based on user's history and preferences."""
        print(f"\n--- Proactive Suggestions for {self.user_profile.user_id} ---")
        if not self.user_profile.shopping_history:
            print("No history yet to provide proactive suggestions.")
            return
        
        # Simple proactive suggestion based on last item in history
        last_item = self.user_profile.shopping_history[-1]
        print(f"Based on your recent interest in '{last_item}', you might like:")
        
        # Simulate finding related items or deals from a preferred platform
        preferred_platform_name = self.user_profile.preferences.get("preferred_platform", "amazon").lower()
        tool_obj = self.available_tools.get(preferred_platform_name)
        
        if tool_obj:
            # Try to infer a category or keyword from the last item
            suggested_query = "laptop" if "laptop" in last_item.lower() else \
                              "headphone" if "headphone" in last_item.lower() else \
                              "deals"
            
            if suggested_query == "deals":
                deals = tool_obj.get_deals(category="electronics") # Default category
                for deal in deals:
                    print(f"- {deal} on {tool_obj.name}")
            else:
                suggestions = tool_obj.search_product(f"best {suggested_query}")
                for s in suggestions:
                    print(f"- {s} on {tool_obj.name}")
        else:
            print("Could not find preferred platform for proactive suggestions.")
        print("-------------------------------------")

    def learn_from_interaction(self, new_interaction_data: dict):
        """Simulates continuous learning by updating user preferences/history."""
        if "new_preference" in new_interaction_data:
            self.user_profile.update_preferences(new_interaction_data["new_preference"])
        if "purchased_item" in new_interaction_data:
            self.user_profile.add_to_history(f"Purchased: {new_interaction_data['purchased_item']}")
        print("Assistant learned from interaction.")


# --- Demo Usage ---
if __name__ == "__main__":
    # 1. Initialize User Profile
    user1 = UserProfile(
        user_id="Alice",
        preferences={
            "language_style": "casual",
            "preferred_platform": "Amazon",
            "preferred_brand": "Dell",
            "max_price": 1200,
            "favorite_category": "electronics"
        }
    )
    user2 = UserProfile(
        user_id="Bob",
        preferences={
            "language_style": "formal",
            "preferred_platform": "eBay",
            "favorite_category": "collectibles"
        }
    )

    # 2. Initialize Available E-commerce Tools
    amazon = AmazonTool()
    ebay = EbayTool()
    available_tools = [amazon, ebay]

    # 3. Initialize Personalized Shopping Assistant
    alice_assistant = PersonalizedShoppingAssistant(user1, available_tools)
    bob_assistant = PersonalizedShoppingAssistant(user2, available_tools)

    print("\n--- Alice's Shopping Session ---")
    # Alice's first request: personalized search
    alice_assistant.assist("search for a laptop")

    # Alice's second request: find deals in her favorite category
    alice_assistant.assist("find deals for electronics")

    # Alice's proactive suggestions
    alice_assistant.proactive_suggestions()

    # Simulate learning from interaction (e.g., she liked a new brand)
    alice_assistant.learn_from_interaction({"new_preference": {"preferred_brand": "HP"}})
    alice_assistant.assist("search for a workstation laptop") # See if the new preference is used

    print("\n--- Bob's Shopping Session ---")
    # Bob's request: personalized search
    bob_assistant.assist("search for vintage camera")

    # Bob's request: find deals
    bob_assistant.assist("find deals for collectibles")

    # Bob's proactive suggestions
    bob_assistant.proactive_suggestions()

    print("\n--- End of Demo ---")