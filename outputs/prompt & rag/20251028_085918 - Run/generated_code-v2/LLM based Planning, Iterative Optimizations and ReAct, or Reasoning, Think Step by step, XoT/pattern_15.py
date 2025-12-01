import re
from typing import Dict, List

class InternalKnowledgeBase:
    def __init__(self):
        self.products = {
            "Tent": {
                "price_range": (100, 300),
                "delivery_days_range": (2, 5),
                "attributes": {
                    "2-person": ["2-person camping tent", "lightweight backpacking tent"],
                    "4-person": ["family camping tent", "large dome tent"],
                    "moderate_climate": ["3-season tent"],
                    "cold_climate": ["4-season tent"],
                }
            },
            "Sleeping Bag": {
                "price_range": (50, 150),
                "delivery_days_range": (3, 6),
                "attributes": {
                    "moderate_climate": ["20-degree sleeping bag", "30-degree sleeping bag"],
                    "cold_climate": ["0-degree sleeping bag", "winter sleeping bag"],
                }
            },
            "Cooking Gear": {
                "price_range": (30, 100),
                "delivery_days_range": (2, 4),
                "attributes": {
                    "basic": ["camping stove", "cookware set"],
                    "full": ["portable grill", "utensil kit"],
                }
            },
            "Clothing": {
                "price_range": (20, 80),
                "delivery_days_range": (3, 7),
                "attributes": {
                    "moderate_climate": ["hiking pants", "fleece jacket"],
                    "cold_climate": ["insulated jacket", "thermal base layer"],
                }
            }
        }

    def get_product_info(self, category: str) -> Dict:
        return self.products.get(category, {})


class IntrospectivePlanner:
    def __init__(self, knowledge_base: InternalKnowledgeBase):
        self.knowledge_base = knowledge_base

    def _parse_goal(self, user_goal: str) -> Dict:
        parsed_data = {
            "items": [],
            "people": 1,
            "budget": float('inf'),
            "delivery_days": float('inf'),
            "climate": "moderate_climate"
        }

        # Extract items (simplified)
        if "tent" in user_goal.lower():
            parsed_data["items"].append("Tent")
        if "sleeping bag" in user_goal.lower() or "sleeping bags" in user_goal.lower():
            parsed_data["items"].append("Sleeping Bag")
        if "cooking gear" in user_goal.lower():
            parsed_data["items"].append("Cooking Gear")
        if "clothing" in user_goal.lower():
            parsed_data["items"].append("Clothing")

        # Extract number of people
        people_match = re.search(r"(\d+)\s*people", user_goal, re.IGNORECASE)
        if people_match: 
            parsed_data["people"] = int(people_match.group(1))

        # Extract budget
        budget_match = re.search(r"under\s*\$(\d+)", user_goal, re.IGNORECASE)
        if budget_match:
            parsed_data["budget"] = float(budget_match.group(1))

        # Extract delivery time
        delivery_match = re.search(r"delivered within\s*(\d+)\s*week", user_goal, re.IGNORECASE)
        if delivery_match:
            parsed_data["delivery_days"] = int(delivery_match.group(1)) * 7
        else:
            delivery_match = re.search(r"delivered within\s*(\d+)\s*day", user_goal, re.IGNORECASE)
            if delivery_match:
                parsed_data["delivery_days"] = int(delivery_match.group(1))

        # Extract climate
        if "cold climate" in user_goal.lower():
            parsed_data["climate"] = "cold_climate"
        elif "moderate climate" in user_goal.lower():
            parsed_data["climate"] = "moderate_climate"

        return parsed_data

    def generate_plan(self, user_goal: str) -> Dict:
        parsed_data = self._parse_goal(user_goal)
        items_to_plan = parsed_data["items"]
        num_people = parsed_data["people"]
        max_budget = parsed_data["budget"]
        max_delivery_days = parsed_data["delivery_days"]
        climate = parsed_data["climate"]

        plan_steps = []
        total_estimated_cost = 0
        max_estimated_delivery_days = 0
        step_id_counter = 1

        for item_category in items_to_plan:
            product_info = self.knowledge_base.get_product_info(item_category)
            if not product_info:
                continue

            avg_price = sum(product_info["price_range"]) / 2
            avg_delivery = sum(product_info["delivery_days_range"]) / 2
            
            item_cost = avg_price
            item_delivery = avg_delivery

            suggested_items = []
            if item_category == "Tent":
                suggested_items.extend(product_info["attributes"].get(f"{num_people}-person", []))
                suggested_items.extend(product_info["attributes"].get(climate, []))
            elif item_category == "Sleeping Bag":
                item_cost *= num_people # Each person needs a sleeping bag
                suggested_items.extend(product_info["attributes"].get(climate, []))
                suggested_items = [f"{item} (x{num_people})" for item in suggested_items]
            elif item_category == "Cooking Gear":
                suggested_items.extend(product_info["attributes"].get("basic", []))
            elif item_category == "Clothing":
                item_cost *= num_people # Each person needs clothing
                suggested_items.extend(product_info["attributes"].get(climate, []))
                suggested_items = [f"{item} (x{num_people})" for item in suggested_items]

            plan_steps.append({
                "step_id": step_id_counter,
                "task": f"Find {item_category}",
                "action_code": f"FIND_PRODUCTS(category='{item_category}', criteria={{'people': {num_people}, 'climate': '{climate}'}}, budget_max={avg_price:.2f})",
                "estimated_budget": item_cost,
                "estimated_delivery_days": item_delivery,
                "suggested_items": list(set(suggested_items)) if suggested_items else [f"General {item_category}"]
            })
            total_estimated_cost += item_cost
            max_estimated_delivery_days = max(max_estimated_delivery_days, item_delivery)
            step_id_counter += 1

        summary = "Plan generated successfully."
        if total_estimated_cost > max_budget:
            summary = f"Warning: Estimated cost (${total_estimated_cost:.2f}) exceeds budget (${max_budget:.2f})."
        if max_estimated_delivery_days > max_delivery_days:
            summary = f"Warning: Estimated delivery ({max_estimated_delivery_days:.0f} days) exceeds target ({max_delivery_days:.0f} days)."
        if total_estimated_cost > max_budget and max_estimated_delivery_days > max_delivery_days:
            summary = f"Warning: Estimated cost (${total_estimated_cost:.2f}) exceeds budget (${max_budget:.2f}) AND estimated delivery ({max_estimated_delivery_days:.0f} days) exceeds target ({max_delivery_days:.0f} days)."
        if not plan_steps:
            summary = "Could not generate a plan for the given goal. Please refine your request."

        return {
            "goal": user_goal,
            "total_estimated_cost": round(total_estimated_cost, 2),
            "delivery_estimate_days": round(max_estimated_delivery_days),
            "plan_steps": plan_steps,
            "summary": summary
        }


# Example Usage:
if __name__ == "__main__":
    kb = InternalKnowledgeBase()
    planner = IntrospectivePlanner(kb)

    user_goal_1 = "Plan a camping trip for two people, including tent, sleeping bags, cooking gear, and appropriate clothing for a moderate climate, all under $500, and delivered within a week."
    plan_1 = planner.generate_plan(user_goal_1)
    import json
    print("\n--- Plan 1 ---")
    print(json.dumps(plan_1, indent=2))

    user_goal_2 = "I need a tent for 4 people for a cold climate, delivered within 3 days. My budget is $300."
    plan_2 = planner.generate_plan(user_goal_2)
    print("\n--- Plan 2 ---")
    print(json.dumps(plan_2, indent=2))

    user_goal_3 = "Just some cooking gear, under $50."
    plan_3 = planner.generate_plan(user_goal_3)
    print("\n--- Plan 3 ---")
    print(json.dumps(plan_3, indent=2))

    user_goal_4 = "I need a yacht."
    plan_4 = planner.generate_plan(user_goal_4)
    print("\n--- Plan 4 ---")
    print(json.dumps(plan_4, indent=2))