import datetime

def get_user_input():
    print("\n--- Smart Travel Planner ---\n")
    destination = input("Enter your desired destination: ")
    start_date_str = input("Enter your start date (YYYY-MM-DD): ")
    end_date_str = input("Enter your end date (YYYY-MM-DD): ")
    budget = float(input("Enter your total budget (e.g., 2000): "))
    preferences = input("Any specific preferences (e.g., 'luxury hotel', 'direct flights')? (Optional): ")

    try:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return get_user_input()

    return {
        "destination": destination,
        "start_date": start_date,
        "end_date": end_date,
        "budget": budget,
        "preferences": preferences.lower()
    }

def generate_initial_itinerary(user_preferences):
    return {
        "destination": user_preferences["destination"],
        "start_date": user_preferences["start_date"],
        "end_date": user_preferences["end_date"],
        "flight": {"cost": 0, "available": False, "details": "TBD"},
        "hotel": {"cost": 0, "available": False, "details": "TBD"},
        "total_cost": 0
    }

def mock_flight_api(destination, start_date, end_date, current_budget_consideration, preferences):
    # Simulate flight availability and pricing
    base_cost = 500 # Base cost for a flight
    if "europe" in destination.lower():
        base_cost = 800
    elif "asia" in destination.lower():
        base_cost = 1000
    
    # Introduce some randomness or budget-based unavailability
    if current_budget_consideration < base_cost * 1.2:
        if "direct flights" in preferences:
            return {"cost": base_cost * 1.5, "available": False, "details": "Direct flights too expensive for budget, consider indirect or different dates"}
        return {"cost": base_cost * 1.1, "available": False, "details": "Flights found but might be slightly above initial budget expectation, consider other options"}
    
    # Simulate some dates being unavailable
    if (start_date.month == 7 or start_date.month == 8) and "europe" in destination.lower():
        if "direct flights" in preferences:
            return {"cost": base_cost * 1.8, "available": False, "details": "Peak season for direct flights to Europe, very expensive or unavailable"}
        return {"cost": base_cost * 1.5, "available": True, "details": f"Flight to {destination} ({start_date} - {end_date}) - {base_cost * 1.5:.2f}"}

    return {"cost": base_cost, "available": True, "details": f"Flight to {destination} ({start_date} - {end_date}) - {base_cost:.2f}"}

def mock_hotel_api(destination, start_date, end_date, current_budget_consideration, preferences):
    # Simulate hotel availability and pricing
    num_nights = (end_date - start_date).days
    base_cost_per_night = 100
    if "luxury hotel" in preferences:
        base_cost_per_night = 250
    elif "europe" in destination.lower():
        base_cost_per_night = 150

    total_hotel_cost = base_cost_per_night * num_nights

    if current_budget_consideration < total_hotel_cost * 1.1:
        return {"cost": total_hotel_cost * 1.2, "available": False, "details": "Hotels found but might be slightly above initial budget expectation"}

    # Simulate some hotels being fully booked
    if (start_date.day % 7 == 0 and start_date.day > 0): # Simulate weekends being popular
        if "luxury hotel" in preferences:
            return {"cost": total_hotel_cost * 1.5, "available": False, "details": "Luxury hotels fully booked or very expensive on these dates"}
        return {"cost": total_hotel_cost * 1.3, "available": True, "details": f"Hotel in {destination} ({num_nights} nights) - {total_hotel_cost * 1.3:.2f}"}

    return {"cost": total_hotel_cost, "available": True, "details": f"Hotel in {destination} ({num_nights} nights) - {total_hotel_cost:.2f}"}

class SmartTravelPlanner:
    def __init__(self, user_preferences):
        self.user_preferences = user_preferences
        self.itinerary = generate_initial_itinerary(user_preferences)
        self.iteration_count = 0
        self.max_iterations = 5

    def evaluate_and_correct_itinerary(self):
        print(f"\n--- Iteration {self.iteration_count + 1}: Evaluating and Correcting Plan ---")
        needs_correction = False
        current_flight_budget = self.user_preferences["budget"] * 0.5 # Allocate half budget for flights initially
        current_hotel_budget = self.user_preferences["budget"] * 0.5 # Allocate half budget for hotels initially

        # Simulate API calls with current budget considerations
        flight_feedback = mock_flight_api(
            self.itinerary["destination"],
            self.itinerary["start_date"],
            self.itinerary["end_date"],
            current_flight_budget,
            self.user_preferences["preferences"]
        )
        hotel_feedback = mock_hotel_api(
            self.itinerary["destination"],
            self.itinerary["start_date"],
            self.itinerary["end_date"],
            current_hotel_budget,
            self.user_preferences["preferences"]
        )

        # Update itinerary with feedback
        self.itinerary["flight"] = flight_feedback
        self.itinerary["hotel"] = hotel_feedback
        self.itinerary["total_cost"] = flight_feedback["cost"] + hotel_feedback["cost"]

        print(f"Current flight status: {flight_feedback['details']} (Available: {flight_feedback['available']})")
        print(f"Current hotel status: {hotel_feedback['details']} (Available: {hotel_feedback['available']})")
        print(f"Current total estimated cost: {self.itinerary['total_cost']:.2f} (Budget: {self.user_preferences['budget']:.2f})")

        # Self-correction logic
        if not flight_feedback["available"] or not hotel_feedback["available"] or self.itinerary["total_cost"] > self.user_preferences["budget"]:
            needs_correction = True
            print("Issues detected. Attempting to self-correct...")

            # Attempt to correct flight issues
            if not flight_feedback["available"] or flight_feedback["cost"] > current_flight_budget:
                print("  -> Adjusting flight search criteria (e.g., slightly later departure, indirect flights).")
                # Simple correction: If flight is too expensive/unavailable, try increasing the implicit budget for flights for the next iteration
                current_flight_budget *= 1.1
                # For a real system, this would involve trying different dates, airports, or flight types
                self.user_preferences["preferences"] = self.user_preferences["preferences"].replace("direct flights", "") # Remove direct flight preference for flexibility

            # Attempt to correct hotel issues
            if not hotel_feedback["available"] or hotel_feedback["cost"] > current_hotel_budget:
                print("  -> Adjusting hotel search criteria (e.g., different hotel class, slightly different dates).")
                # Simple correction: If hotel is too expensive/unavailable, try increasing the implicit budget for hotels
                current_hotel_budget *= 1.1
                # For a real system, this would involve trying different neighborhoods, star ratings, or dates
                self.user_preferences["preferences"] = self.user_preferences["preferences"].replace("luxury hotel", "") # Remove luxury preference

            # If budget is exceeded, try to be more flexible with dates if possible
            if self.itinerary["total_cost"] > self.user_preferences["budget"] * 1.1: # 10% buffer
                print("  -> Overall budget exceeded significantly. Suggesting date flexibility.")
                # Simple correction: Shift dates by one day to see if prices change
                self.itinerary["start_date"] += datetime.timedelta(days=1)
                self.itinerary["end_date"] += datetime.timedelta(days=1)
                print(f"     New proposed dates: {self.itinerary['start_date']} to {self.itinerary['end_date']}")

        return needs_correction

    def run_planner(self):
        print("\n--- Starting Travel Plan Generation ---")
        while self.iteration_count < self.max_iterations:
            if not self.evaluate_and_correct_itinerary():
                print("\n--- Plan deemed satisfactory after corrections. ---")
                break
            self.iteration_count += 1
        else:
            print(f"\n--- Max iterations ({self.max_iterations}) reached. Presenting best plan found. ---")

        self.display_itinerary()

    def display_itinerary(self):
        print("\n--- Final Proposed Itinerary ---")
        print(f"Destination: {self.itinerary['destination']}")
        print(f"Dates: {self.itinerary['start_date']} to {self.itinerary['end_date']}")
        print(f"Flight: {self.itinerary['flight']['details']} (Cost: {self.itinerary['flight']['cost']:.2f})")
        print(f"Hotel: {self.itinerary['hotel']['details']} (Cost: {self.itinerary['hotel']['cost']:.2f})")
        print(f"Total Estimated Cost: {self.itinerary['total_cost']:.2f}")
        print(f"User Budget: {self.user_preferences['budget']:.2f}")
        if self.itinerary['total_cost'] > self.user_preferences['budget']:
            print("WARNING: Total estimated cost exceeds your budget.")
        print("------------------------------------\n")

if __name__ == "__main__":
    user_prefs = get_user_input()
    planner = SmartTravelPlanner(user_prefs)
    planner.run_planner()