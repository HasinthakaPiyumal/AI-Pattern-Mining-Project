from pydantic import BaseModel, Field
import json
from typing import List, Optional

class Activity(BaseModel):
    name: str = Field(..., description="Name of the activity")
    type: str = Field(..., description="Type of activity, e.g., Sightseeing, Culture, Leisure")
    duration_hours: Optional[float] = Field(None, description="Estimated duration in hours")

class Meal(BaseModel):
    time_of_day: str = Field(..., description="Time of the meal, e.g., Breakfast, Lunch, Dinner")
    description: str = Field(..., description="Brief description of the meal")
    cuisine: Optional[str] = Field(None, description="Type of cuisine")

class DayPlan(BaseModel):
    day_number: int = Field(..., description="The day number in the itinerary")
    summary: str = Field(..., description="A brief summary of the day's plan")
    activities: List[Activity] = Field(..., description="List of activities for the day")
    meals: List[Meal] = Field(..., description="List of meals for the day")
    accommodation: str = Field(..., description="Accommodation for the night")
    transportation: str = Field(..., description="Primary mode of transportation for the day")

class TravelItinerary(BaseModel):
    destination: str = Field(..., description="The travel destination")
    days: List[DayPlan] = Field(..., description="List of daily plans for the itinerary")

llm_generated_json_string = """
{
  "destination": "London",
  "days": [
    {
      "day_number": 1,
      "summary": "Explore iconic landmarks and markets",
      "activities": [
        {
          "name": "Tower of London",
          "type": "Sightseeing",
          "duration_hours": 3.0
        },
        {
          "name": "Walk across Tower Bridge and visit Borough Market",
          "type": "Sightseeing/Food",
          "duration_hours": 2.5
        }
      ],
      "meals": [
        {
          "time_of_day": "Lunch",
          "description": "Grab a sandwich at a local cafe near the Tower"
        },
        {
          "time_of_day": "Dinner",
          "description": "Dinner at The Shard",
          "cuisine": "Modern European"
        }
      ],
      "accommodation": "The Savoy",
      "transportation": "Tube"
    }
  ,
    {
      "day_number": 2,
      "summary": "Museums and Parks",
      "activities": [
        {
          "name": "British Museum",
          "type": "Culture",
          "duration_hours": 4.0
        },
        {
          "name": "Stroll through Hyde Park",
          "type": "Leisure",
          "duration_hours": 2.0
        }
      ],
      "meals": [
        {
          "time_of_day": "Lunch",
          "description": "Cafe inside the museum"
        },
        {
          "time_of_day": "Dinner",
          "description": "Dinner in Soho"
        }
      ],
      "accommodation": "The Savoy",
      "transportation": "Bus"
    }
  ]
}
"""

parsed_data = json.loads(llm_generated_json_string)
travel_itinerary = TravelItinerary(**parsed_data)

print(json.dumps(travel_itinerary.model_dump(), indent=2))