
class Product:
    def __init__(self, name: str, price: float, categories: list):
        self.name = name
        self.price = price
        self.categories = categories

    def __repr__(self):
        return f"Product(name='{self.name}', price=${self.price:.2f}, categories={self.categories})"

class UserRequirements:
    def __init__(self, required_items: list = None, budget: float = None, desired_categories: list = None):
        self.required_items = required_items if required_items is not None else []
        self.budget = budget
        self.desired_categories = desired_categories if desired_categories is not None else []

    def __repr__(self):
        return (f"UserRequirements(required_items={self.required_items}, "
                f"budget={self.budget if self.budget is not None else 'No Budget'}, "
                f"desired_categories={self.desired_categories})")

class ShoppingAssistant:
    def __init__(self, product_catalog: list[Product]):
        self.product_catalog = product_catalog
        self.current_reasoning = None  # Stores the agent's thought process
        self.last_action = None       # Stores the agent's last executed action

    def reason(self, user_reqs: UserRequirements) -> dict:
        """
        Simulates the agent's internal reasoning process to find the most cost-optimized
        selection of products that fulfill user requirements. This is the 'Thought' phase.
        """
        print(f"\n[Agent Thought Process] Analyzing user requirements: {user_reqs}")
        optimal_selection = []
        total_cost = 0.0
        reasoning_steps = []

        # Step 1: Prioritize required items, finding the cheapest option for each
        for req_item in user_reqs.required_items:
            matching_products = [p for p in self.product_catalog if req_item.lower() in p.name.lower()]
            if not matching_products:
                reasoning_steps.append(f"Could not find '{req_item}'.")
                continue
            
            cheapest_option = min(matching_products, key=lambda p: p.price)
            optimal_selection.append(cheapest_option)
            total_cost += cheapest_option.price
            reasoning_steps.append(f"Identified '{cheapest_option.name}' (price: ${cheapest_option.price:.2f}) for '{req_item}'.")

        # Step 2: Consider desired categories, adding items if within budget
        remaining_budget = user_reqs.budget - total_cost if user_reqs.budget is not None else float('inf')
        if user_reqs.budget is not None and total_cost > user_reqs.budget:
            reasoning_steps.append(f"Initial selection cost (${total_cost:.2f}) exceeds budget (${user_reqs.budget:.2f}).")

        for desired_cat in user_reqs.desired_categories:
            if user_reqs.budget is not None and remaining_budget <= 0:
                reasoning_steps.append(f"Budget exhausted for desired category '{desired_cat}'.")
                break

            potential_products = [
                p for p in self.product_catalog
                if desired_cat in p.categories and p not in optimal_selection
            ]
            if potential_products:
                cheapest_in_category = min(potential_products, key=lambda p: p.price)
                if cheapest_in_category.price <= remaining_budget:
                    optimal_selection.append(cheapest_in_category)
                    total_cost += cheapest_in_category.price
                    remaining_budget -= cheapest_in_category.price
                    reasoning_steps.append(f"Added '{cheapest_in_category.name}' (price: ${cheapest_in_category.price:.2f}) from desired category '{desired_cat}'. Remaining budget: ${remaining_budget:.2f}.")
                else:
                    reasoning_steps.append(f"Cheapest item for category '{desired_cat}' ('{cheapest_in_category.name}', ${cheapest_in_category.price:.2f}) exceeds remaining budget (${remaining_budget:.2f}).")
            else:
                reasoning_steps.append(f"No suitable products found for desired category '{desired_cat}'.")

        self.current_reasoning = {
            "optimal_selection": optimal_selection,
            "total_cost": total_cost,
            "user_requirements": user_reqs,
            "reasoning_steps": reasoning_steps
        }
        print(f"[Agent Thought Process] Concluded optimal selection: {[p.name for p in optimal_selection]} with total cost ${total_cost:.2f}.")
        return self.current_reasoning

    def act(self) -> dict:
        """
        Executes the agent's action (product recommendations) based on its last reasoning.
        This is the 'Action' phase.
        """
        if not self.current_reasoning:
            print("[Agent Action] No reasoning performed yet. Cannot act.")
            return None

        print("\n[Agent Action] Generating recommendations based on recent reasoning...")
        recommended_products = self.current_reasoning["optimal_selection"]
        recommended_cost = self.current_reasoning["total_cost"]
        user_reqs = self.current_reasoning["user_requirements"]
        
        action_explanation = (f"Based on the goal to minimize cost while fulfilling requirements "
                              f"({user_reqs}), the following products are recommended:\n")
        
        for p in recommended_products:
            action_explanation += f"- {p.name} (${p.price:.2f})\n"
        
        action_explanation += f"Total Recommended Cost: ${recommended_cost:.2f}"

        if user_reqs.budget is not None:
            if recommended_cost > user_reqs.budget:
                action_explanation += f"\nWarning: Recommended cost (${recommended_cost:.2f}) exceeds your budget (${user_reqs.budget:.2f})."
            else:
                action_explanation += f"\nThis selection is within your budget (${user_reqs.budget:.2f})."

        self.last_action = {
            "recommended_products": recommended_products,
            "recommended_cost": recommended_cost,
            "explanation": action_explanation
        }
        print("[Agent Action] Action generated successfully.")
        return self.last_action

    def validate_action(self) -> tuple[bool, str]:
        """
        Internally validates if the last action aligns with the last reasoning.
        This is the 'Synchronization Check'.
        """
        if not self.current_reasoning or not self.last_action:
            return False, "No reasoning or action to validate."

        print("\n[Agent Validation] Validating action against reasoning...")

        reasoning_optimal_selection = self.current_reasoning["optimal_selection"]
        reasoning_total_cost = self.current_reasoning["total_cost"]
        action_recommended_products = self.last_action["recommended_products"]
        action_recommended_cost = self.last_action["recommended_cost"]

        # Check if recommended products and cost match the reasoning's output
        selection_matches = sorted([p.name for p in reasoning_optimal_selection]) == sorted([p.name for p in action_recommended_products])
        cost_matches = abs(reasoning_total_cost - action_recommended_cost) < 0.01

        validation_report = []
        is_aligned = True

        if not selection_matches:
            is_aligned = False
            validation_report.append("Discrepancy: Recommended product selection does NOT match the optimal selection derived from reasoning.")
            validation_report.append(f"  Reasoning thought: {[p.name for p in reasoning_optimal_selection]}")
            validation_report.append(f"  Action recommended: {[p.name for p in action_recommended_products]}")
        else:
            validation_report.append("Alignment: Recommended product selection matches reasoning.")

        if not cost_matches:
            is_aligned = False
            validation_report.append("Discrepancy: Recommended total cost does NOT match the total cost derived from reasoning.")
            validation_report.append(f"  Reasoning cost: ${reasoning_total_cost:.2f}")
            validation_report.append(f"  Action cost: ${action_recommended_cost:.2f}")
        else:
            validation_report.append("Alignment: Recommended total cost matches reasoning.")

        if is_aligned:
            validation_report.append("Conclusion: Agent's action is successfully synchronized with its reasoning.")
        else:
            validation_report.append("Conclusion: Agent's action is NOT synchronized with its reasoning. Further investigation/correction needed.")
        
        print("\n".join(validation_report))
        return is_aligned, "\n".join(validation_report)

# --- Example Usage ---
# Product Catalog
product_catalog = [
    Product("Laptop Pro", 1200.00, ["electronics", "computers"]),
    Product("Laptop Lite", 800.00, ["electronics", "computers"]),
    Product("Wireless Mouse", 25.00, ["electronics", "accessories"]),
    Product("Mechanical Keyboard", 75.00, ["electronics", "accessories"]),
    Product("Headphones X", 150.00, ["electronics", "audio"]),
    Product("Coffee Mug", 15.00, ["home", "kitchen"]),
    Product("Desk Lamp", 40.00, ["home", "lighting"]),
    Product("Notebook 100pg", 5.00, ["office", "stationary"]),
    Product("Pen Set", 8.00, ["office", "stationary"]),
]

# Initialize the assistant
assistant = ShoppingAssistant(product_catalog)

# Scenario 1: Basic requirement with cost minimization and desired categories within budget
print("\n=== Scenario 1: Synchronized Action ===")
user_requirements_1 = UserRequirements(
    required_items=["laptop", "mouse"],
    budget=1000.00,
    desired_categories=["audio"]
)

reasoning_output_1 = assistant.reason(user_requirements_1)
action_output_1 = assistant.act()
is_aligned_1, validation_report_1 = assistant.validate_action()

# Scenario 2: Budget Exceeded - Agent identifies the issue in reasoning and reports in action
print("\n=== Scenario 2: Budget Exceeded (Reported) ===")
user_requirements_2 = UserRequirements(
    required_items=["Laptop Pro", "Headphones X"],
    budget=1300.00,
    desired_categories=["stationary"]
)

reasoning_output_2 = assistant.reason(user_requirements_2)
action_output_2 = assistant.act()
is_aligned_2, validation_report_2 = assistant.validate_action()

# Scenario 3: Intentional Misalignment (for demonstration purposes)
print("\n=== Scenario 3: Intentional Misalignment (Detected by Validation) ===")
user_requirements_3 = UserRequirements(
    required_items=["laptop"],
    budget=1000.00
)
reasoning_output_3 = assistant.reason(user_requirements_3)

# Manually override action to simulate a discrepancy
assistant.last_action = {
    "recommended_products": [Product("Laptop Pro", 1200.00, ["electronics", "computers"])], 
    "recommended_cost": 1200.00,
    "explanation": "Simulated misaligned action: Recommended a more expensive laptop than reasoned."
}
# In a real system, the 'act' method would generate this, here we bypass it for demonstration.
# action_output_3 = assistant.act() 

is_aligned_3, validation_report_3 = assistant.validate_action()