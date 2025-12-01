import json
import re

def simulate_user_input(preferences: str) -> str:
    return preferences

def simulate_llm_itinerary_generation(preferences: str) -> str:
    # This function simulates an LLM generating a natural language itinerary.
    # In a real application, this would involve an actual LLM call.
    if "paris" in preferences.lower() and "3-day" in preferences.lower():
        return (
            "Here is your 3-day trip to Paris!\n\n"
            "Day 1: Arrival in Paris\n"
            "Morning: Arrive at Charles de Gaulle Airport, transfer to Hotel Le Bristol. Check-in.\n"
            "Afternoon: Visit the Louvre Museum. Explore the art collections.\n"
            "Evening: Dinner at a traditional French bistro near the hotel. Stroll along the Seine River.\n"
            "Accommodation: Hotel Le Bristol.\n"
            "Transportation: Taxi from airport, Metro for museum.\n\n"
            "Day 2: Culture and Views\n"
            "Morning: Explore Notre Dame Cathedral (exterior view) and Île de la Cité. Walk around the Latin Quarter.\n"
            "Afternoon: Visit the Eiffel Tower. Enjoy panoramic views of the city.\n"
            "Evening: Dinner at a rooftop restaurant with Eiffel Tower views. Attend a cabaret show.\n"
            "Accommodation: Hotel Le Bristol.\n"
            "Transportation: Metro throughout the day.\n\n"
            "Day 3: Art and Departure\n"
            "Morning: Visit Musée d'Orsay. Admire impressionist masterpieces.\n"
            "Afternoon: Last-minute souvenir shopping. Transfer to Charles de Gaulle Airport for departure.\n"
            "Evening: Fly back home.\n"
            "Accommodation: N/A\n"
            "Transportation: Metro for shopping, Taxi to airport.\n"
        )
    else:
        return "I'm sorry, I can only generate a 3-day trip to Paris for this demo."

def parse_and_structure_itinerary(natural_language_itinerary: str) -> dict:
    structured_plan = {
        "destination": "",
        "duration_days": 0,
        "daily_plan": []
    }

    # Extract destination and duration
    destination_match = re.search(r"trip to (.*?)[!\n]", natural_language_itinerary)
    if destination_match:
        structured_plan["destination"] = destination_match.group(1).strip()
    
    duration_match = re.search(r"(\d+)-day trip", natural_language_itinerary)
    if duration_match:
        structured_plan["duration_days"] = int(duration_match.group(1))

    # Split itinerary into days
    day_sections = re.split(r"\n\nDay \d+: ", natural_language_itinerary)
    # The first element is usually the intro, skip it.
    if day_sections and "Day 1" in day_sections[0]: # Handle case where Day 1 starts right away
        day_sections[0] = day_sections[0][day_sections[0].find("Day 1:"):] # keep Day 1 content
    elif len(day_sections) > 1:
        day_sections = day_sections[1:] # Skip the intro part
    else:
        return structured_plan # No daily plans found

    for i, day_content_raw in enumerate(day_sections):
        day_num_match = re.match(r"Day (\d+): (.+?)\n", day_content_raw)
        if not day_num_match:
            continue
        
        day_num = int(day_num_match.group(1))
        day_title = day_num_match.group(2).strip()
        day_content = day_content_raw[day_content_raw.find('\n') + 1:] # Remove the 