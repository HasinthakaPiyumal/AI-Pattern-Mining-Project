import random

def load_travel_data():
    return [
        {"name": "Eiffel Tower", "type": "landmark", "duration": 3, "location": "Paris"},
        {"name": "Louvre Museum", "type": "museum", "duration": 4, "location": "Paris"},
        {"name": "Notre Dame Cathedral", "type": "landmark", "duration": 2, "location": "Paris"},
        {"name": "Sacre-Coeur Basilica", "type": "landmark", "duration": 2, "location": "Paris"},
        {"name": "Disneyland Paris", "type": "amusement_park", "duration": 8, "location": "Paris"},
        {"name": "Champs-Élysées", "type": "shopping", "duration": 3, "location": "Paris"},
        {"name": "Musée d'Orsay", "type": "museum", "duration": 3, "location": "Paris"},
        {"name": "Arc de Triomphe", "type": "landmark", "duration": 1, "location": "Paris"},
        {"name": "Seine River Cruise", "type": "activity", "duration": 2, "location": "Paris"},
        {"name": "Montmartre", "type": "district", "duration": 4, "location": "Paris"},
    ]

def generate_unconstrained_itinerary(attractions, num_days=3):
    itinerary = {day: [] for day in range(1, num_days + 1)}
    available_attractions = list(attractions)

    for day in range(1, num_days + 1):
        day_attractions_types = set()
        for _ in range(3):
            if not available_attractions:
                break
            potential_attraction = random.choice(available_attractions)
            itinerary[day].append(potential_attraction)
            available_attractions.remove(potential_attraction)
    return itinerary

def avoid_repeated_attractions_on_same_day(itinerary):
    new_itinerary = {day: [] for day, activities in itinerary.items()}
    for day, activities in itinerary.items():
        seen_on_day = set()
        for activity in activities:
            if activity["name"] not in seen_on_day:
                new_itinerary[day].append(activity)
                seen_on_day.add(activity["name"])
    return new_itinerary

def ensure_reasonable_travel_time(itinerary):
    # Placeholder for actual travel time calculation based on location
    # For this example, we'll assume a fixed buffer between activities
    # and ensure total duration for the day is not excessive.
    MAX_DAILY_DURATION = 8  # hours
    revised_itinerary = {day: [] for day, activities in itinerary.items()}

    for day, activities in itinerary.items():
        current_day_duration = 0
        for activity in activities:
            if current_day_duration + activity["duration"] <= MAX_DAILY_DURATION:
                revised_itinerary[day].append(activity)
                current_day_duration += activity["duration"]
    return revised_itinerary

def diversify_activities(itinerary):
    revised_itinerary = {day: [] for day, activities in itinerary.items()}

    for day, activities in itinerary.items():
        seen_types = set()
        temp_activities = []
        for activity in activities:
            if activity["type"] not in seen_types:
                temp_activities.append(activity)
                seen_types.add(activity["type"])
        # If not enough diversity, add more distinct activities if available from original pool
        # This simple version just filters out duplicates within a day to enforce some diversity
        # A more complex version would re-plan or swap activities.
        revised_itinerary[day] = temp_activities
    return revised_itinerary

def avoid_overcrowding_day(itinerary, max_activities_per_day=4):
    revised_itinerary = {day: [] for day, activities in itinerary.items()}
    for day, activities in itinerary.items():
        revised_itinerary[day] = activities[:max_activities_per_per_day]
    return revised_itinerary

def main():
    attractions = load_travel_data()
    print("\n--- Generating Unconstrained Itinerary ---")
    unconstrained_itinerary = generate_unconstrained_itinerary(attractions, num_days=3)
    for day, activities in unconstrained_itinerary.items():
        print(f"Day {day}:")
        for activity in activities:
            print(f"  - {activity['name']} ({activity['type']}) - Duration: {activity['duration']} hours")

    print("\n--- Applying Commonsense Constraints ---")

    # Constraint 1: Avoid repeated attractions on the same day
    itinerary_c1 = avoid_repeated_attractions_on_same_day(unconstrained_itinerary)
    print("\nAfter avoiding repeated attractions on same day:")
    for day, activities in itinerary_c1.items():
        print(f"Day {day}:")
        for activity in activities:
            print(f"  - {activity['name']} ({activity['type']}) - Duration: {activity['duration']} hours")

    # Constraint 2: Ensure reasonable travel time (placeholder implementation)
    itinerary_c2 = ensure_reasonable_travel_time(itinerary_c1)
    print("\nAfter ensuring reasonable daily activity duration:")
    for day, activities in itinerary_c2.items():
        print(f"Day {day}:")
        for activity in activities:
            print(f"  - {activity['name']} ({activity['type']}) - Duration: {activity['duration']} hours")

    # Constraint 3: Diversify activities
    itinerary_c3 = diversify_activities(itinerary_c2)
    print("\nAfter diversifying activities:")
    for day, activities in itinerary_c3.items():
        print(f"Day {day}:")
        for activity in activities:
            print(f"  - {activity['name']} ({activity['type']}) - Duration: {activity['duration']} hours")

    # Constraint 4: Avoid overcrowding day
    itinerary_final = avoid_overcrowding_day(itinerary_c3, max_activities_per_day=3)
    print("\n--- Final Itinerary (with Commonsense Constraints) ---")
    for day, activities in itinerary_final.items():
        print(f"Day {day}:")
        if not activities:
            print("  (No activities planned for this day)")
        for activity in activities:
            print(f"  - {activity['name']} ({activity['type']}) - Duration: {activity['duration']} hours")

if __name__ == "__main__":
    main()