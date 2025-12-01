
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import re
import json

# 3. Structured Output Definition (Pydantic Models)
class DailyPlan(BaseModel):
    day: int
    date: Optional[str] = None # Could be inferred or explicitly provided
    activities: List[str] = Field(default_factory=list)
    transportation: Optional[str] = None
    accommodation: Optional[str] = None
    meals: List[str] = Field(default_factory=list)

class TravelItinerary(BaseModel):
    destination: str
    duration_days: int
    daily_plans: List[DailyPlan] = Field(default_factory=list)

class SmartTravelPlanner:
    def __init__(self, llm_model: str = "simulated-llm"):
        self.llm_model = llm_model
        # In a real application, you would initialize your LLM client here
        # e.g., self.openai_client = OpenAI(api_key="YOUR_API_KEY")

    # 2. LLM Interaction Module (Simulated for this example)
    def _generate_natural_language_itinerary(self, destination: str, duration_days: int, interests: Optional[List[str]] = None) -> str:
        """
        Simulates an LLM generating a natural language travel itinerary.
        In a real scenario, this would involve an API call to an actual LLM.
        """
        print(f"Simulating LLM call for {destination} for {duration_days} days...")
        if destination.lower() == "paris" and duration_days == 3:
            return """Here is your 3-day trip to Paris:

Day 1: Arrival and Eiffel Tower
Activities: Arrive in Paris, check into hotel, visit Eiffel Tower, stroll along the Seine.
Transportation: Taxi from airport to hotel, Metro for sightseeing.
Accommodation: Hotel Le Littré
Meals: Dinner at a local bistro near the Eiffel Tower.

Day 2: Museums and Montmartre
Activities: Explore the Louvre Museum, walk through Tuileries Garden, visit Notre Dame Cathedral (exterior), explore Montmartre, Sacré-Cœur Basilica.
Transportation: Metro, walking.
Accommodation: Hotel Le Littré
Meals: Breakfast at hotel, Lunch near Louvre, Dinner in Montmartre.

Day 3: Versailles and Departure
Activities: Day trip to Versailles Palace and Gardens, return to Paris, last-minute souvenir shopping.
Transportation: RER train to Versailles, Metro in Paris, taxi to airport.
Accommodation: N/A
Meals: Breakfast at hotel, Lunch in Versailles.
"""
        else:
            return f"A {duration_days}-day trip to {destination} is being planned. (Simulated content not detailed for this destination/duration.)"

    # 4. Natural Language Parser / Post-processing Module
    def _parse_natural_language_itinerary(self, nl_itinerary: str, destination: str, duration_days: int) -> TravelItinerary:
        """
        Parses the natural language itinerary into a structured Pydantic model.
        """
        daily_plans: List[DailyPlan] = []
        
        # Split the itinerary into daily blocks
        day_blocks = re.split(r"\n\s*Day (\d+):", nl_itinerary)[1:] # [1:] to remove the part before "Day 1"

        for i in range(0, len(day_blocks), 2):
            day_num_str = day_blocks[i].strip()
            day_content = day_blocks[i+1].strip()
            
            day_num = int(day_num_str)
            
            activities_match = re.search(r"Activities: ([^\n]+)", day_content)
            transportation_match = re.search(r"Transportation: ([^\n]+)", day_content)
            accommodation_match = re.search(r"Accommodation: ([^\n]+)", day_content)
            meals_match = re.search(r"Meals: ([^\n]+)", day_content)

            daily_plans.append(DailyPlan(
                day=day_num,
                activities=[act.strip() for act in activities_match.group(1).split(',')] if activities_match else [],
                transportation=transportation_match.group(1).strip() if transportation_match else None,
                accommodation=accommodation_match.group(1).strip() if accommodation_match else None,
                meals=[meal.strip() for meal in meals_match.group(1).split(',')] if meals_match else [],
            ))
        
        return TravelItinerary(
            destination=destination,
            duration_days=duration_days,
            daily_plans=daily_plans
        )

    def plan_trip(self, destination: str, duration_days: int, interests: Optional[List[str]] = None) -> Dict:
        """
        Generates a structured travel plan for the given destination and duration.
        """
        # 2. LLM Interaction Module
        nl_itinerary = self._generate_natural_language_itinerary(destination, duration_days, interests)
        
        # 4. Natural Language Parser / Post-processing Module
        structured_itinerary = self._parse_natural_language_itinerary(nl_itinerary, destination, duration_days)
        
        # 5. Output Module
        return structured_itinerary.model_dump()

# 1. User Interface (Example using basic print/input for demonstration)
if __name__ == "__main__":
    planner = SmartTravelPlanner()

    print("Welcome to the Smart Travel Planner!")
    destination = input("Enter your desired destination (e.g., Paris): ")
    duration_days = int(input("Enter the duration of your trip in days (e.g., 3): "))
    # interests = input("Enter your interests (comma-separated, optional): ").split(',') if input else None

    try:
        structured_plan_json = planner.plan_trip(destination, duration_days)
        print("\n--- Generated Structured Travel Plan (JSON) ---")
        print(json.dumps(structured_plan_json, indent=2))
        print("\n--- Explanation ---")
        print("The Smart Travel Planner successfully generated a travel itinerary in a structured JSON format.")
        print("This allows for easy integration with other systems, automated evaluation, and consistent data representation.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your input matches the expected format for parsing.")

