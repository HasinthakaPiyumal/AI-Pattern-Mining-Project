import re
import json

def get_user_preferences():
    print("\n--- User Input Module ---")
    preferences = input("Please tell me your travel preferences (e.g., destination, dates, interests, budget): ")
    return preferences

def simulate_llm_plan(preferences):
    print("\n--- LLM Interaction Module (Simulated) ---")
    print(f"Simulating LLM generating a plan based on: '{preferences}'")
    # This is a hardcoded example of an LLM-generated natural language travel plan.
    # In a real application, this would be an API call to an LLM.
    simulated_plan = """
Day 1:
Transportation: Fly from London to Paris. Take a taxi from Charles de Gaulle Airport to the hotel.
Accommodation: Check into Hotel Louvre in the 1st arrondissement.
Meals: Breakfast at a local boulangerie. Lunch at Le Comptoir du Relais (French bistro). Dinner at Septime (Michelin-starred).
Attractions: Visit the Louvre Museum, stroll along the Seine River, see the Eiffel Tower illuminated at night.

Day 2:
Transportation: Use the Paris Metro for inter-city travel.
Accommodation: Stay at Hotel Louvre.
Meals: Breakfast at the hotel. Lunch near Notre Dame at a small cafe. Dinner cruise on the Seine.
Attractions: Explore Notre Dame Cathedral (exterior), walk through the Latin Quarter, visit the Musée d'Orsay.

Day 3:
Transportation: High-speed train (TGV) from Paris to Nice. Taxi to hotel.
Accommodation: Check into Hotel Negresco on the Promenade des Anglais.
Meals: Breakfast at the hotel. Lunch at a beachfront restaurant in Nice. Dinner at Le Chantecler (fine dining).
Attractions: Relax on the beach, explore Vieux Nice (Old Town), visit Cours Saleya flower market.
"""
    print("LLM Generated Plan (Natural Language):\n" + simulated_plan)
    return simulated_plan

def post_process_plan(natural_language_plan):
    print("\n--- Post-processing Module ---")
    structured_itinerary = []
    days = re.split(r'\nDay (\d+):', natural_language_plan)[1:] # Split by 'Day X:' and keep day numbers

    for i in range(0, len(days), 2):
        day_num = int(days[i])
        day_content = days[i+1].strip()

        transportation_match = re.search(r'Transportation: (.+?)(?=\nAccommodation:|\nMeals:|\nAttractions:|$)', day_content, re.DOTALL)
        accommodation_match = re.search(r'Accommodation: (.+?)(?=\nMeals:|\nAttractions:|$)', day_content, re.DOTALL)
        meals_match = re.search(r'Meals: (.+?)(?=\nAttractions:|$)', day_content, re.DOTALL)
        attractions_match = re.search(r'Attractions: (.+)', day_content, re.DOTALL)

        transportation = transportation_match.group(1).strip() if transportation_match else "N/A"
        accommodation = accommodation_match.group(1).strip() if accommodation_match else "N/A"
        raw_meals = meals_match.group(1).strip() if meals_match else "N/A"
        raw_attractions = attractions_match.group(1).strip() if attractions_match else "N/A"

        parsed_meals = {"breakfast": "N/A", "lunch": "N/A", "dinner": "N/A"}
        if raw_meals != "N/A":
            meal_parts = raw_meals.split('. ')
            for part in meal_parts:
                if "Breakfast at" in part:
                    parsed_meals["breakfast"] = part.replace("Breakfast at ", "").strip().rstrip('.')
                elif "Lunch at" in part:
                    parsed_meals["lunch"] = part.replace("Lunch at ", "").strip().rstrip('.')
                elif "Dinner at" in part:
                    parsed_meals["dinner"] = part.replace("Dinner at ", "").strip().rstrip('.')
                elif "Breakfast at the hotel" in part:
                    parsed_meals["breakfast"] = "at the hotel"
                elif "Dinner cruise on the Seine" in part:
                    parsed_meals["dinner"] = "cruise on the Seine"

        parsed_attractions = [a.strip() for a in raw_attractions.split(',')] if raw_attractions != "N/A" else []

        structured_itinerary.append({
            "day": day_num,
            "transportation": transportation,
            "accommodation": accommodation,
            "meals": parsed_meals,
            "attractions": parsed_attractions
        })
    return structured_itinerary

def display_structured_output(structured_itinerary):
    print("\n--- Output Module ---")
    print("Structured Travel Itinerary (JSON):\n")
    print(json.dumps(structured_itinerary, indent=2))


if __name__ == "__main__":
    user_preferences = get_user_preferences()
    natural_language_plan = simulate_llm_plan(user_preferences)
    structured_plan = post_process_plan(natural_language_plan)
    display_structured_output(structured_plan)
