import streamlit as st
import requests
import random

class MockAPI:
    def get_attractions(self, destination, interests):
        all_attractions = {
            "Paris": [
                {"name": "Eiffel Tower", "type": "landmark", "cost": 20, "duration": 3, "tags": ["iconic", "sightseeing"]},
                {"name": "Louvre Museum", "type": "museum", "cost": 17, "duration": 4, "tags": ["art", "culture"]},
                {"name": "Notre Dame Cathedral", "type": "landmark", "cost": 0, "duration": 2, "tags": ["history", "sightseeing"]},
                {"name": "Montmartre", "type": "neighborhood", "cost": 0, "duration": 3, "tags": ["art", "views"]},
                {"name": "Seine River Cruise", "type": "activity", "cost": 15, "duration": 1.5, "tags": ["sightseeing", "leisure"]}
            ],
            "Tokyo": [
                {"name": "Shibuya Crossing", "type": "landmark", "cost": 0, "duration": 1, "tags": ["iconic", "shopping"]},
                {"name": "Senso-ji Temple", "type": "temple", "cost": 0, "duration": 2, "tags": ["history", "culture"]},
                {"name": "Tokyo Skytree", "type": "landmark", "cost": 30, "duration": 2, "tags": ["views", "sightseeing"]},
                {"name": "Ghibli Museum", "type": "museum", "cost": 10, "duration": 3, "tags": ["art", "anime"]},
                {"name": "Akihabara", "type": "neighborhood", "cost": 0, "duration": 3, "tags": ["electronics", "anime"]}
            ],
            "New York City": [
                {"name": "Statue of Liberty", "type": "landmark", "cost": 25, "duration": 4, "tags": ["iconic", "history"]},
                {"name": "Central Park", "type": "park", "cost": 0, "duration": 3, "tags": ["nature", "leisure"]},
                {"name": "Times Square", "type": "landmark", "cost": 0, "duration": 2, "tags": ["iconic", "entertainment"]},
                {"name": "Metropolitan Museum of Art", "type": "museum", "cost": 25, "duration": 4, "tags": ["art", "culture"]},
                {"name": "Broadway Show", "type": "activity", "cost": 100, "duration": 3, "tags": ["entertainment", "culture"]}
            ]
        }
        dest_attractions = all_attractions.get(destination, [])
        if interests:
            filtered_attractions = [att for att in dest_attractions if any(tag in interests for tag in att.get("tags", []))]
            return filtered_attractions if filtered_attractions else dest_attractions
        return dest_attractions

    def get_restaurants(self, destination, budget_level):
        all_restaurants = {
            "Paris": [
                {"name": "Le Relais de l'Entrecôte", "cuisine": "French", "cost_level": "medium"},
                {"name": "L'As du Fallafel", "cuisine": "Middle Eastern", "cost_level": "low"},
                {"name": "Septime", "cuisine": "Modern French", "cost_level": "high"}
            ],
            "Tokyo": [
                {"name": "Sushi Dai", "cuisine": "Sushi", "cost_level": "high"},
                {"name": "Ichiran Ramen", "cuisine": "Ramen", "cost_level": "medium"},
                {"name": "Tsukiji Outer Market", "cuisine": "Street Food", "cost_level": "low"}
            ],
            "New York City": [
                {"name": "Joe's Shanghai", "cuisine": "Chinese", "cost_level": "medium"},
                {"name": "Katz's Delicatessen", "cuisine": "Deli", "cost_level": "medium"},
                {"name": "Per Se", "cuisine": "Fine Dining", "cost_level": "high"}
            ]
        }
        dest_restaurants = all_restaurants.get(destination, [])
        if budget_level:
            filtered_restaurants = [res for res in dest_restaurants if res.get("cost_level") == budget_level]
            return filtered_restaurants if filtered_restaurants else dest_restaurants
        return dest_restaurants

    def get_lodging(self, destination, budget_per_night):
        lodgings = {
            "Paris": [{"name": "Hotel Rivoli", "cost_per_night": 150}, {"name": "Hostel Generator", "cost_per_night": 50}, {"name": "Ritz Paris", "cost_per_night": 1000}],
            "Tokyo": [{"name": "Park Hyatt Tokyo", "cost_per_night": 400}, {"name": "Capsule Hotel", "cost_per_night": 40}, {"name": "Hotel Gracery Shinjuku", "cost_per_night": 180}],
            "New York City": [{"name": "The Plaza Hotel", "cost_per_night": 700}, {"name": "Pod 39", "cost_per_night": 120}, {"name": "YOTEL New York", "cost_per_night": 190}]
        }
        available_lodging = []
        for lodging in lodgings.get(destination, []):
            if lodging["cost_per_night"] <= budget_per_night:
                available_lodging.append(lodging)
        if not available_lodging and lodgings.get(destination):
            return [min(lodgings.get(destination), key=lambda x: x['cost_per_night'])]
        return available_lodging

    def get_weather(self, destination):
        return {"Paris": "Sunny", "Tokyo": "Cloudy", "New York City": "Rainy"}.get(destination, "Clear")

class PlanningAgent:
    def __init__(self, mock_api):
        self.mock_api = mock_api
        self.memory = {}

    def _get_llm_suggestion(self, prompt):
        # Mock LLM response for suggestions and descriptions
        if "day overview" in prompt:
            return "A perfect blend of culture and iconic sights. Prepare for a day of exploration!"
        elif "attraction" in prompt:
            return "This attraction offers a unique blend of history and breathtaking views."
        elif "restaurant" in prompt:
            return "Enjoy a delightful meal at this highly-rated spot."
        return "A great choice for your trip!"

    def _decompose_task(self, duration):
        return [f"Day {i+1}" for i in range(duration)]

    def _manage_memory(self, key, value=None):
        if value:
            self.memory[key] = value
        return self.memory.get(key)

    def _satisfy_constraints(self, current_budget, item_cost):
        return current_budget >= item_cost

    def _plan_day(self, day_num, destination, interests, daily_budget, visited_attractions):
        day_itinerary = {"day": day_num, "activities": [], "lodging": None, "meals": [], "daily_cost": 0}
        current_day_budget = daily_budget

        st.sidebar.write(f"Planning {day_num} with budget: {current_day_budget:.2f}")

        # Try to find an attraction
        available_attractions = [att for att in self.mock_api.get_attractions(destination, interests) if att["name"] not in visited_attractions]
        if available_attractions:
            selected_attraction = random.choice(available_attractions)
            if self._satisfy_constraints(current_day_budget, selected_attraction["cost"]):
                day_itinerary["activities"].append({
                    "type": "Attraction",
                    "name": selected_attraction["name"],
                    "cost": selected_attraction["cost"],
                    "description": self._get_llm_suggestion(f"Describe {selected_attraction['name']} attraction")
                })
                current_day_budget -= selected_attraction["cost"]
                day_itinerary["daily_cost"] += selected_attraction["cost"]
                visited_attractions.add(selected_attraction["name"])
                st.sidebar.write(f"  Added attraction: {selected_attraction['name']}. Remaining budget: {current_day_budget:.2f}")

        # Try to find a restaurant (lunch/dinner)
        budget_level = "medium" if current_day_budget > daily_budget * 0.2 else "low"
        available_restaurants = self.mock_api.get_restaurants(destination, budget_level)
        if available_restaurants:
            selected_restaurant = random.choice(available_restaurants)
            # Mock restaurant cost based on level
            restaurant_cost = {"low": 20, "medium": 40, "high": 80}.get(selected_restaurant["cost_level"], 30)

            if self._satisfy_constraints(current_day_budget, restaurant_cost):
                day_itinerary["meals"].append({
                    "type": "Lunch/Dinner",
                    "name": selected_restaurant["name"],
                    "cost": restaurant_cost,
                    "description": self._get_llm_suggestion(f"Describe {selected_restaurant['name']} restaurant")
                })
                current_day_budget -= restaurant_cost
                day_itinerary["daily_cost"] += restaurant_cost
                st.sidebar.write(f"  Added restaurant: {selected_restaurant['name']}. Remaining budget: {current_day_budget:.2f}")

        return day_itinerary, current_day_budget, visited_attractions

    def generate_itinerary(self, destination, duration, interests, total_budget):
        itinerary = []
        daily_budget_estimate = total_budget / duration if duration > 0 else total_budget
        remaining_total_budget = total_budget
        visited_attractions = set()

        st.sidebar.title("Planning Progress")

        for i, day_task in enumerate(self._decompose_task(duration)):
            st.sidebar.write(f"\nProcessing {day_task}...")
            current_day_plan, day_spent_budget, updated_visited = self._plan_day(
                day_task,
                destination,
                interests,
                daily_budget_estimate, # Use estimated daily budget for planning individual day
                visited_attractions
            )
            visited_attractions = updated_visited
            remaining_total_budget -= current_day_plan["daily_cost"]
            itinerary.append(current_day_plan)

            # Adjust daily budget for subsequent days if current day was over/under budget
            if duration - (i + 1) > 0:
                daily_budget_estimate = remaining_total_budget / (duration - (i + 1))

        # Add lodging for the whole trip (simplified)
        if duration > 1:
            lodging_budget_per_night = (total_budget * 0.3) / (duration - 1) if duration > 1 else 0 # Allocate 30% of total budget for lodging
            available_lodging = self.mock_api.get_lodging(destination, lodging_budget_per_night)
            if available_lodging:
                selected_lodging = random.choice(available_lodging)
                lodging_cost_total = selected_lodging["cost_per_night"] * (duration - 1)
                if lodging_cost_total <= total_budget * 0.3: # Check if within the allocated percentage
                    for day_plan in itinerary[:-1]: # Assign lodging to all but the last day
                        day_plan["lodging"] = {"name": selected_lodging["name"], "cost": selected_lodging["cost_per_night"], "total_cost": lodging_cost_total}
                else:
                    st.sidebar.warning("Could not find lodging within allocated budget. Adjusting...")
                    # Fallback to cheapest lodging if primary choice is too expensive, re-evaluate total budget impact
                    cheapest_lodging = min(self.mock_api.get_lodging(destination, 99999), key=lambda x: x['cost_per_night'])
                    if cheapest_lodging:
                        lodging_cost_total = cheapest_lodging["cost_per_night"] * (duration - 1)
                        if lodging_cost_total < remaining_total_budget + (total_budget * 0.7): # Check if total trip budget can absorb it
                            for day_plan in itinerary[:-1]:
                                day_plan["lodging"] = {"name": cheapest_lodging["name"], "cost": cheapest_lodging["cost_per_night"], "total_cost": lodging_cost_total}

        return itinerary


def main():
    st.set_page_config(page_title="AI Travel Itinerary Planner", layout="wide")
    st.title("AI-Powered Multi-Day Travel Itinerary Planner")

    with st.sidebar:
        st.header("Plan Your Trip")
        destination = st.selectbox("Destination", ["Paris", "Tokyo", "New York City"])
        duration = st.slider("Duration (days)", 1, 7, 3)
        interests = st.multiselect("Interests (optional)", ["art", "culture", "sightseeing", "history", "food", "shopping", "nature", "entertainment", "views", "iconic", "leisure", "anime", "electronics"])
        budget = st.number_input("Total Budget ($)", min_value=100, value=1000, step=50)

        if st.button("Generate Itinerary"):
            st.session_state["show_itinerary"] = True
            mock_api = MockAPI()
            agent = PlanningAgent(mock_api)
            st.session_state["itinerary"] = agent.generate_itinerary(destination, duration, interests, budget)

    if st.session_state.get("show_itinerary") and st.session_state.get("itinerary"):
        st.header(f"Your {duration}-Day Trip to {destination}")
        total_estimated_cost = 0

        for day_plan in st.session_state["itinerary"]:
            st.subheader(day_plan["day"])
            st.write(f"Weather: {MockAPI().get_weather(destination)}")
            st.markdown(f"**Overview:** {PlanningAgent(MockAPI())._get_llm_suggestion('day overview')}")

            daily_cost = 0

            if day_plan["lodging"]:
                st.write(f"**Lodging:** {day_plan['lodging']['name']} (${day_plan['lodging']['cost']} per night)")
                # Daily cost for lodging is only for the nights, not the full duration days
                # This needs careful handling for multi-day where lodging cost is for previous night
                # For simplicity, assuming lodging cost is calculated for (duration - 1) nights and distributed.
                # However, for display we can show daily cost as part of sum.

            if day_plan["activities"]:
                st.write("**Activities:**")
                for activity in day_plan["activities"]:
                    st.write(f"- {activity['name']} (${activity['cost']:.2f}) - {activity['description']}")
                    daily_cost += activity["cost"]

            if day_plan["meals"]:
                st.write("**Meals:**")
                for meal in day_plan["meals"]:
                    st.write(f"- {meal['name']} (${meal['cost']:.2f}) - {meal['description']}")
                    daily_cost += meal["cost"]

            st.markdown(f"**Estimated Daily Activity/Meal Cost: ${daily_cost:.2f}**")
            total_estimated_cost += daily_cost

        # Calculate total lodging cost separately
        if st.session_state["itinerary"] and st.session_state["itinerary"][0]["lodging"]:
            total_lodging_cost = st.session_state["itinerary"][0]["lodging"]["cost"] * (duration - 1) if duration > 1 else 0
            total_estimated_cost += total_lodging_cost
            st.subheader("Total Lodging Cost")
            st.write(f"Estimated Total Lodging: ${total_lodging_cost:.2f}")

        st.subheader("Total Estimated Trip Cost")
        st.write(f"Your estimated total trip cost (activities + meals + lodging): **${total_estimated_cost:.2f}**")
        if total_estimated_cost > budget:
            st.error(f"Warning: Your estimated trip cost exceeds your total budget by ${total_estimated_cost - budget:.2f}!")


if __name__ == "__main__":
    main()
