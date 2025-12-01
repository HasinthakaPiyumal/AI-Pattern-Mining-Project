import json
import re

def generate_travel_plan_structured_output(user_preferences, llm_natural_language_plan):
    structured_plan = {
        "destination": user_preferences.get("destination"),
        "dates": user_preferences.get("dates"),
        "travelers": user_preferences.get("travel_companions"),
        "interests": user_preferences.get("interests"),
        "itinerary": []
    }

    # Split the plan by days
    day_sections = re.split(r"\n\s*Day\s*(\d+):", llm_natural_language_plan)
    # The first element before 