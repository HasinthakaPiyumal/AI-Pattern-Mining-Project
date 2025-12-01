import json
from pydantic import BaseModel, Field
from typing import List, Optional

class Activity(BaseModel):
    name: str = Field(..., description="Name of the activity or attraction")
    time: str = Field(..., description="Suggested time for the activity (e.g., 'Morning', '10:00 AM')")
    description: Optional[str] = Field(None, description="Brief description of the activity")

class DayPlan(BaseModel):
    date: str = Field(..., description="Date of the day's plan (e.g., 'Day 1', '2023-10-27')")
    activities: List[Activity] = Field(..., description="List of activities planned for the day")
    meals: List[str] = Field(..., description="List of meal suggestions for the day")

class Itinerary(BaseModel):
    destination: str = Field(..., description="The travel destination")
    duration_days: int = Field(..., description="Total duration of the trip in days")
    plan: List[DayPlan] = Field(..., description="Detailed plan for each day of the trip")

def simulate_llm_itinerary_generation(destination: str, dates: str, interests: str, budget: str) -> str:
    """Simulates an LLM generating a natural language travel itinerary."""
    return f"""Here is a personalized {destination} travel plan for your {dates} trip, focusing on {interests} within your {budget} budget:

Day 1: Arrival and City Exploration
Morning: Arrive in {destination}, check into hotel.
Afternoon: Explore the historic city center, visit local markets.
Evening: Dinner at a highly-rated seafood restaurant. Stroll along the waterfront.

Day 2: Cultural Immersion
Morning: Visit the main historical museum. Attend a local art workshop.
Afternoon: Enjoy a traditional local lunch. Discover hidden alleyways and artisan shops.
Evening: Experience a live cultural performance. Try street food for dinner.

Day 3: Nature and Relaxation
Morning: Take a scenic boat tour to a nearby island or nature reserve.
Afternoon: Relax on the beach or hike a gentle nature trail. Picnic lunch.
Evening: Farewell dinner at a restaurant with panoramic views. Enjoy a quiet evening.

Enjoy your trip to {destination}!
"""

def parse_natural_language_itinerary(nl_itinerary: str) -> Optional[Itinerary]:
    """Parses a natural language itinerary into a structured Pydantic Itinerary model."""
    destination = "Unknown"
    duration_days = 0
    plan_days = []

    # Extract destination and duration (basic parsing)
    if "travel plan for your" in nl_itinerary and "trip" in nl_itinerary:
        parts = nl_itinerary.split("travel plan for your", 1)[1].split("trip", 1)
        if len(parts) > 0:
            duration_str = parts[0].strip().split(" ")[0] # e.g., 