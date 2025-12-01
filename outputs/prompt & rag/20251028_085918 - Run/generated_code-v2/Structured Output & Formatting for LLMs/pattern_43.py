import json
import re

def simulate_llm_response(user_request):
    """
    Simulates an LLM generating a natural language event itinerary.
    In a real application, this would be an actual LLM API call.
    """
    if "tech conference" in user_request.lower():
        return """Day 1: Arrival and Keynote. Attendees arrive by 9:00 AM. Opening Keynote by Dr. Alice Smith at 10:00 AM in Main Hall. Lunch Break at 12:00 PM. Afternoon workshops from 1:30 PM to 4:30 PM in various breakout rooms. Networking Reception at 6:00 PM in the Atrium.
Day 2: Workshops and Panel Discussion. Morning workshops start at 9:00 AM. Panel Discussion: "Future of AI" at 11:00 AM in Main Hall. Lunch Break at 1:00 PM. Afternoon sessions on specific tech topics from 2:30 PM to 5:00 PM. Evening Gala Dinner at 7:00 PM at The Grand Hotel.
Day 3: Deep Dives and Closing Remarks. Deep Dive sessions on advanced topics from 9:30 AM to 12:00 PM. Closing Remarks by CEO Bob Johnson at 1:30 PM in Main Hall. Event concludes at 2:30 PM."""
    else:
        return """Day 1: Morning exploration. Visit local museum at 10:00 AM. Lunch at a local cafe at 1:00 PM. Evening concert at 7:00 PM.
Day 2: Outdoor adventure. Hiking trip at 9:00 AM. Picnic lunch at 12:30 PM. Stargazing at 9:00 PM."""

def parse_natural_language_itinerary(natural_language_itinerary):
    """
    Parses a natural language itinerary into a structured JSON format.
    This function uses rule-based parsing suitable for a predictable LLM output structure.
    """
    structured_itinerary = {"days": []}
    days = natural_language_itinerary.strip().split('\n')

    for day_entry in days:
        day_match = re.match(r"Day (\d+):\s*(.*)", day_entry)
        if not day_match:
            continue

        day_number = int(day_match.group(1))
        day_content = day_match.group(2).strip()
        activities = []

        # Split activities using common separators like '.' followed by a new capitalized word or just '.'
        # This regex tries to split at the end of a sentence that isn't the end of the entire day's content
        activity_parts = re.split(r'\. (?=[A-Z])|\.(?!$)', day_content)
        activity_parts = [part.strip() for part in activity_parts if part.strip()]

        for activity_text in activity_parts:
            time_match = re.match(r"(.*?)\s*at\s*(\d{1,2}:\d{2}\s*(?:AM|PM)?)\s*(?:in|at)?\s*(.*)", activity_text, re.IGNORECASE)
            if time_match:
                description = time_match.group(1).strip()
                time = time_match.group(2).strip()
                location = time_match.group(3).strip() if time_match.group(3) else ""
                activities.append({"time": time, "description": description, "location": location})
            else:
                # Handle activities without explicit time/location in the simple regex
                activities.append({"time": "N/A", "description": activity_text, "location": "N/A"})

        structured_itinerary["days"].append({"day_number": day_number, "activities": activities})

    return structured_itinerary

def main():
    user_request = input("Enter your event planning request: ")

    print("\n--- Simulating LLM Response ---")
    llm_output = simulate_llm_response(user_request)
    print("Natural Language Itinerary:\n", llm_output)

    print("\n--- Post-processing to Structured Output ---")
    structured_output = parse_natural_language_itinerary(llm_output)
    print("Structured JSON Itinerary:")
    print(json.dumps(structured_output, indent=2))

if __name__ == "__main__":
    main()