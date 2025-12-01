import json

class Flight:
    def __init__(self, airline: str, flight_number: str, departure_airport: str, arrival_airport: str, departure_time: str, arrival_time: str):
        self.airline = airline
        self.flight_number = flight_number
        self.departure_airport = departure_airport
        self.arrival_airport = arrival_airport
        self.departure_time = departure_time
        self.arrival_time = arrival_time

    def to_dict(self):
        return {
            "airline": self.airline,
            "flight_number": self.flight_number,
            "departure_airport": self.departure_airport,
            "arrival_airport": self.arrival_airport,
            "departure_time": self.departure_time,
            "arrival_time": self.arrival_time,
        }

class Accommodation:
    def __init__(self, name: str, location: str, check_in: str, check_out: str):
        self.name = name
        self.location = location
        self.check_in = check_in
        self.check_out = check_out

    def to_dict(self):
        return {
            "name": self.name,
            "location": self.location,
            "check_in": self.check_in,
            "check_out": self.check_out,
        }

class Activity:
    def __init__(self, name: str, time: str, description: str):
        self.name = name
        self.time = time
        self.description = description

    def to_dict(self):
        return {
            "name": self.name,
            "time": self.time,
            "description": self.description,
        }

class Dining:
    def __init__(self, name: str, time: str, cuisine: str):
        self.name = name
        self.time = time
        self.cuisine = cuisine

    def to_dict(self):
        return {
            "name": self.name,
            "time": self.time,
            "cuisine": self.cuisine,
        }

class DayPlan:
    def __init__(self, date: str, activities: list, dining_options: list):
        self.date = date
        self.activities = activities
        self.dining_options = dining_options

    def to_dict(self):
        return {
            "date": self.date,
            "activities": [act.to_dict() for act in self.activities],
            "dining_options": [din.to_dict() for din in self.dining_options],
        }

class TravelPlan:
    def __init__(self, destination: str, start_date: str, end_date: str, flights: list, accommodation: Accommodation, daily_plans: list):
        self.destination = destination
        self.start_date = start_date
        self.end_date = end_date
        self.flights = flights
        self.accommodation = accommodation
        self.daily_plans = daily_plans

    def to_dict(self):
        return {
            "destination": self.destination,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "flights": [f.to_dict() for f in self.flights],
            "accommodation": self.accommodation.to_dict() if self.accommodation else None,
            "daily_plans": [dp.to_dict() for dp in self.daily_plans],
        }

# --- LLM Service (Simulated) ---
def generate_natural_language_plan(destination: str, dates: str, interests: str, budget: str) -> str:
    return f"""Here is a wonderful travel plan to {destination} from {dates} with your interests in {interests} on a {budget} budget.

Day 1 (2024-08-01): Arrive at {destination}. Check into The Grand Hotel. In the afternoon, visit the Central Museum. For dinner, enjoy Italian food at Pasta Paradise at 7 PM.

Day 2 (2024-08-02): Morning, explore the Old Town. Have lunch at the Riverside Cafe (local cuisine) at 1 PM. In the evening, attend a concert at the City Music Hall at 8 PM.

Flight details: Fly from SFO to {destination} on AirlineX, flight AX123, departing 2024-07-31 10:00 AM, arriving 2024-07-31 6:00 PM. Return flight from {destination} to SFO on AirlineY, flight YZ456, departing 2024-08-03 9:00 AM, arriving 2024-08-03 5:00 PM.
"""

# --- Parser Service (Simulated) ---
def parse_and_structure_plan(natural_language_plan: str, destination: str, start_date: str, end_date: str) -> TravelPlan:
    # This is a highly simplified parser. In a real application, an LLM with 'instructor' or a robust NLP parser would be used.
    # We'll extract some fixed details based on the simulated natural language plan.

    # Flights
    flight1 = Flight(
        airline="AirlineX", 
        flight_number="AX123", 
        departure_airport="SFO", 
        arrival_airport=destination, 
        departure_time="2024-07-31 10:00 AM", 
        arrival_time="2024-07-31 6:00 PM"
    )
    flight2 = Flight(
        airline="AirlineY", 
        flight_number="YZ456", 
        departure_airport=destination, 
        arrival_airport="SFO", 
        departure_time="2024-08-03 9:00 AM", 
        arrival_time="2024-08-03 5:00 PM"
    )

    # Accommodation
    accommodation = Accommodation(
        name="The Grand Hotel", 
        location="Downtown", 
        check_in="2024-08-01", 
        check_out="2024-08-03"
    )

    # Day 1
    activity1_day1 = Activity(name="Visit Central Museum", time="Afternoon", description="Explore art and history.")
    dining1_day1 = Dining(name="Pasta Paradise", time="7 PM", cuisine="Italian")
    day1_plan = DayPlan("2024-08-01", [activity1_day1], [dining1_day1])

    # Day 2
    activity1_day2 = Activity(name="Explore Old Town", time="Morning", description="Wander through historic streets.")
    dining1_day2 = Dining(name="Riverside Cafe", time="1 PM", cuisine="Local Cuisine")
    activity2_day2 = Activity(name="Attend City Music Hall Concert", time="8 PM", description="Enjoy live music.")
    day2_plan = DayPlan("2024-08-02", [activity1_day2, activity2_day2], [dining1_day2])

    return TravelPlan(
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        flights=[flight1, flight2],
        accommodation=accommodation,
        daily_plans=[day1_plan, day2_plan]
    )

# --- Main Application Logic ---
def main():
    print("--- Automated Travel Itinerary Generator ---")
    destination = input("Enter your desired destination (e.g., Paris): ")
    dates = input("Enter your travel dates (e.g., Aug 1-3, 2024): ")
    interests = input("Enter your interests (e.g., museums, food, concerts): ")
    budget = input("Enter your budget (e.g., moderate, luxury): ")

    print("\nGenerating natural language travel plan...")
    natural_plan = generate_natural_language_plan(destination, dates, interests, budget)
    print("\n--- Natural Language Plan ---")
    print(natural_plan)

    print("\nParsing and structuring the plan...")
    # Extracting start and end dates simply for the parser's constructor for this demo
    # In a real scenario, these would be parsed more robustly or passed directly.
    start_date = dates.split("-")[0].strip() + ", " + dates.split(",")[-1].strip()
    end_date = dates.split("-")[-1].strip()
    
    structured_plan = parse_and_structure_plan(natural_plan, destination, start_date, end_date)

    print("\n--- Structured Travel Plan (JSON) ---")
    print(json.dumps(structured_plan.to_dict(), indent=2))

if __name__ == "__main__":
    main()