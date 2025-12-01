import random

class ECommerceEnv:
    def __init__(self):
        self.products = {
            "101": {"name": "Waterproof Jacket", "price": 85.00, "category": "Apparel", "features": "waterproof, breathable", "in_stock": 10},
            "102": {"name": "Running Shoes", "price": 60.00, "category": "Footwear", "features": "lightweight, cushioned", "in_stock": 20},
            "201": {"name": "Laptop Pro X", "price": 1200.00, "category": "Electronics", "features": "16GB RAM, 512GB SSD", "in_stock": 5},
            "202": {"name": "Smartwatch Series 5", "price": 250.00, "category": "Electronics", "features": "heart rate, GPS", "in_stock": 15},
            "301": {"name": "Coffee Maker Deluxe", "price": 75.00, "category": "Home & Kitchen", "features": "programmable, 12-cup", "in_stock": 8},
        }
        self.cart = {}
        self.current_page = "home"
        self.last_search_results = []

    def _get_product_details(self, product_id):
        return self.products.get(product_id)

    def search_products(self, query, max_price=None, category=None):
        results = []
        query_lower = query.lower()
        for pid, product in self.products.items():
            match = False
            if query_lower in product["name"].lower() or query_lower in product["features"].lower():
                match = True
            if category and product["category"].lower() != category.lower():
                match = False
            if max_price is not None and product["price"] > max_price:
                match = False
            if match:
                results.append({"id": pid, "name": product["name"], "price": product["price"]})
        self.last_search_results = results
        self.current_page = "search_results"
        return results

    def view_product(self, product_id):
        product = self._get_product_details(product_id)
        if product:
            self.current_page = f"product_page_{product_id}"
            return product
        return None

    def add_to_cart(self, product_id, quantity=1):
        product = self._get_product_details(product_id)
        if product and product["in_stock"] >= quantity:
            self.cart[product_id] = self.cart.get(product_id, 0) + quantity
            product["in_stock"] -= quantity
            return f"{quantity} of {product["name"]} added to cart."
        return "Failed to add to cart: product not found or out of stock."

    def get_cart_contents(self):
        return self.cart

    def checkout(self):
        if not self.cart:
            return "Your cart is empty."
        total = sum(self.products[pid]["price"] * qty for pid, qty in self.cart.items())
        self.cart = {}
        self.current_page = "checkout_success"
        return f"Checkout successful! Total: ${total:.2f}"

    def get_current_state(self):
        return {"page": self.current_page, "cart": self.cart, "last_search_results": self.last_search_results}


class DemonstrationCollector:
    def __init__(self):
        self.demonstrations = []

    def record_step(self, action, observation):
        self.demonstrations.append({"action": action, "observation": observation})

    def get_demonstrations(self):
        return self.demonstrations

    def save_demonstrations(self, filename="demonstrations.json"):
        print(f"Saving {len(self.demonstrations)} demonstrations to {filename} (simulated).")
        # In a real scenario, this would write to a JSON/CSV file
        # import json
        # with open(filename, "w") as f:
        #     json.dump(self.demonstrations, f, indent=4)


class ComparisonCollector:
    def __init__(self):
        self.comparisons = []

    def record_comparison(self, query, output_a, output_b, preferred_output):
        self.comparisons.append({"query": query, "output_A": output_a, "output_B": output_b, "preferred_output": preferred_output})

    def get_comparisons(self):
        return self.comparisons

    def save_comparisons(self, filename="comparisons.json"):
        print(f"Saving {len(self.comparisons)} comparisons to {filename} (simulated).")
        # In a real scenario, this would write to a JSON/CSV file
        # import json
        # with open(filename, "w") as f:
        #     json.dump(self.comparisons, f, indent=4)


class RewardModel:
    def __init__(self):
        self.trained = False

    def train(self, comparison_data):
        print(f"Training Reward Model with {len(comparison_data)} comparison examples (simulated).")
        self.trained = True

    def predict_reward(self, query, output):
        if not self.trained:
            return random.uniform(-1, 1)
        # Simulate a reward prediction based on output length or keywords
        reward = len(output) * 0.1 + (0.5 if "found" in output.lower() else 0)
        return min(max(reward, -1), 1) # Clamp between -1 and 1


class LLMShoppingAssistant:
    def __init__(self, env):
        self.env = env
        self.model = None # Represents the underlying LLM
        self.tokenizer = None
        self.bc_trained = False
        self.rlhf_trained = False

    def _simulate_llm_response(self, prompt, context=None):
        # A very basic simulation of an LLM response
        if "search" in prompt.lower():
            return "ACTION: search_products(" + prompt.split("search ", 1)[1].split(" for",1)[0].strip() + ")"
        elif "product details" in prompt.lower():
            return "ACTION: view_product(" + prompt.split("details for ", 1)[1].strip() + ")"
        elif "add to cart" in prompt.lower():
            parts = prompt.split("add ", 1)[1].split(" to cart", 1)
            qty_name = parts[0].strip().split(" ", 1)
            qty = int(qty_name[0]) if qty_name[0].isdigit() else 1
            name = qty_name[1] if len(qty_name) > 1 else qty_name[0]
            return f"ACTION: add_to_cart(name='{name}', quantity={qty})"
        elif "checkout" in prompt.lower():
            return "ACTION: checkout()"
        elif "cart contents" in prompt.lower() or "what's in my cart" in prompt.lower():
            return f"Your cart contains: {self.env.get_cart_contents()}"
        return "I'm not sure how to respond to that. Can you rephrase?"

    def behavior_cloning_train(self, demonstration_data):
        print(f"Starting Behavior Cloning training with {len(demonstration_data)} demonstrations (simulated)...")
        # In a real scenario, this would involve loading a base LLM,
        # tokenizing actions/observations, and fine-tuning using PyTorch/TensorFlow and transformers.
        self.bc_trained = True
        print("Behavior Cloning training complete.")

    def rlhf_train(self, reward_model, num_iterations=2):
        if not self.bc_trained:
            print("Please train with Behavior Cloning first.")
            return
        print(f"Starting RLHF training with Reward Model for {num_iterations} iterations (simulated)...")
        # This would involve: generate responses, get rewards from RewardModel,
        # update policy using PPO/DPO from trl library.
        for i in range(num_iterations):
            print(f"RLHF Iteration {i+1}/{num_iterations}...")
            # Simulate interaction and reward feedback
            dummy_query = "Find me a cheap laptop."
            dummy_output = self._simulate_llm_response(dummy_query)
            reward = reward_model.predict_reward(dummy_query, dummy_output)
            print(f"  Generated output: \"{dummy_output}\" (simulated), Reward: {reward:.2f}")
        self.rlhf_trained = True
        print("RLHF training complete.")

    def chat(self, user_query):
        print(f"User: {user_query}")
        # Simulate LLM reasoning and tool use
        llm_output_action = self._simulate_llm_response(user_query, context=self.env.get_current_state())

        if llm_output_action.startswith("ACTION:"):
            action_str = llm_output_action[len("ACTION:"):].strip()
            print(f"Assistant (decided action): {action_str}")
            try:
                # A very crude way to parse and execute actions
                if action_str.startswith("search_products("):
                    parts = action_str.split("search_products(", 1)[1][:-1].split(",", 2)
                    query = parts[0].strip("'")
                    max_price = float(parts[1].split("=")[1].strip()) if len(parts) > 1 and "max_price" in parts[1] else None
                    category = parts[2].split("=")[1].strip("')") if len(parts) > 2 and "category" in parts[2] else None
                    results = self.env.search_products(query, max_price=max_price, category=category)
                    return f"Found {len(results)} products: {[r['name'] for r in results]}"
                elif action_str.startswith("view_product(name="):
                    product_name = action_str.split("name=", 1)[1].strip("')")
                    # Find product by name, this is simplified
                    product_id = next((pid for pid, p in self.env.products.items() if p['name'].lower() == product_name.lower()), None)
                    if product_id:
                        details = self.env.view_product(product_id)
                        return f"Product Details: {details['name']} - ${details['price']:.2f} - {details['features']}"
                    return f"Could not find details for {product_name}."
                elif action_str.startswith("add_to_cart(name="):
                    parts = action_str.split("name=", 1)[1].split(", quantity=")
                    product_name = parts[0].strip("')")
                    quantity = int(parts[1].strip(")")) if len(parts) > 1 else 1
                    product_id = next((pid for pid, p in self.env.products.items() if p['name'].lower() == product_name.lower()), None)
                    if product_id:
                        return self.env.add_to_cart(product_id, quantity)
                    return f"Could not add {product_name} to cart."
                elif action_str.startswith("checkout("):
                    return self.env.checkout()
                else:
                    return f"Assistant (unknown action): {action_str}"
            except Exception as e:
                return f"Assistant (action execution error): {e}"
        else:
            # If LLM doesn't decide on an action, it generates a direct response
            return f"Assistant: {llm_output_action}"


class APIServer:
    def __init__(self, assistant):
        self.assistant = assistant
        print("API Server initialized. Use .run() to simulate interaction.")

    def _handle_chat_request(self, user_query):
        print(f"API received chat request: '{user_query}'")
        response = self.assistant.chat(user_query)
        print(f"API sending response: '{response}'")
        return {"response": response}

    def run(self):
        print("Simulating API endpoint '/chat'. Type 'exit' to quit.")
        while True:
            user_input = input("Enter user query (or 'exit'): ")
            if user_input.lower() == 'exit':
                break
            self._handle_chat_request(user_input)


# --- Main execution flow simulation --- 

if __name__ == "__main__":
    print("\n--- Setting up E-commerce Shopping Assistant Project ---")

    # 1. Initialize Environment and Collectors
    env = ECommerceEnv()
    demo_collector = DemonstrationCollector()
    comp_collector = ComparisonCollector()

    # 2. Simulate Demonstration Data Collection
    print("\n--- Simulating Demonstration Data Collection ---")
    expert_actions = [
        {"query": "Find a waterproof jacket", "action": "search_products(\'waterproof jacket\', category=\'Apparel\')"},
        {"query": "Tell me about Laptop Pro X", "action": "view_product(name=\'Laptop Pro X\')"},
        {"query": "Add 2 coffee makers to cart", "action": "add_to_cart(name=\'Coffee Maker Deluxe\', quantity=2)"},
    ]

    for i, expert_task in enumerate(expert_actions):
        print(f"Expert performing task {i+1}: {expert_task['query']}")
        # Simulate expert interaction with the environment and record it
        initial_state = env.get_current_state()
        # In a real scenario, the 'action' would be parsed and executed by the environment
        # and the observation would be the resulting state after execution.
        # Here, we'll just record a mock action and observation.
        mock_observation = {"state_after_action": "simulated_state_change", "search_results_count": random.randint(1,5)}
        demo_collector.record_step(action=expert_task['action'], observation=mock_observation)
    demo_collector.save_demonstrations()

    # 3. Initialize and Train LLM Agent (Behavior Cloning)
    assistant = LLMShoppingAssistant(env)
    assistant.behavior_cloning_train(demo_collector.get_demonstrations())

    # 4. Simulate Comparison Data Collection (after initial BC training)
    print("\n--- Simulating Comparison Data Collection ---")
    llm_output_1 = assistant._simulate_llm_response("Find a cheap smartwatch")
    llm_output_2 = assistant._simulate_llm_response("Show me smartwatches under $300")
    comp_collector.record_comparison(
        query="Find a cheap smartwatch", 
        output_a=llm_output_1, 
        output_b=llm_output_2, 
        preferred_output=("A" if len(llm_output_1) > len(llm_output_2) else "B") # Mock preference
    )

    human_expert_output = "ACTION: search_products('smartwatch', max_price=300)"
    llm_output_3 = assistant._simulate_llm_response("What's a good smartwatch?")
    comp_collector.record_comparison(
        query="What's a good smartwatch?", 
        output_a=llm_output_3, 
        output_b=human_expert_output, 
        preferred_output="B" # Human output preferred
    )
    comp_collector.save_comparisons()

    # 5. Initialize and Train Reward Model
    reward_model = RewardModel()
    reward_model.train(comp_collector.get_comparisons())

    # 6. Fine-tune LLM Agent with RLHF
    assistant.rlhf_train(reward_model)

    # 7. Deploy and Interact via Simulated API
    print("\n--- Starting Simulated API Interaction ---")
    api_server = APIServer(assistant)
    api_server.run()

    print("\n--- Simulation Complete ---")
