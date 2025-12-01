import json
import re
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, validator, ValidationError
from fastapi import FastAPI, HTTPException

class Activity(BaseModel):
    time: str = Field(..., description="Time of the activity, e.g., \"09:00 AM - 12:00 PM\"")
    description: str = Field(..., description="Description of the activity, e.g., \"Outdoor Team Building Games\"")
    location: str = Field(..., description="Location of the activity, e.g., \"Central Park\"")
    type: str = Field(..., description="Type of activity, e.g., \"Team Building\", \"Meal\", \"Accommodation\", \"Transport\"")

class Meal(BaseModel):
    time: str = Field(..., description="Time of the meal, e.g., \"01:00 PM - 02:00 PM\"")
    description: str = Field(..., description="Description of the meal, e.g., \"Lunch at The Garden Restaurant\"")
    cuisine: Optional[str] = Field(None, description="Cuisine type, e.g., \"Italian\"")

class Accommodation(BaseModel):
    check_in_date: str = Field(..., description="Check-in date, e.g., \"2024-10-26\"")
    check_out_date: str = Field(..., description="Check-out date, e.g., \"2024-10-27\"")
    hotel_name: str = Field(..., description="Name of the hotel, e.g., \"Grand City Hotel\"")
    address: str = Field(..., description="Address of the hotel, e.g., \"123 Main St\"")

class Transportation(BaseModel):
    time: str = Field(..., description="Time of transport, e.g., \"08:00 AM\"")
    mode: str = Field(..., description="Mode of transport, e.g., \"Bus\"")
    route: str = Field(..., description="Route of transport, e.g., \"Hotel to Central Park\"")

class DailySchedule(BaseModel):
    day: str = Field(..., description="Day of the plan, e.g., \"Day 1\"")
    activities: List[Activity] = Field(..., description="List of activities for the day")
    meals: List[Meal] = Field([], description="List of meals for the day")
    accommodation: Optional[Accommodation] = Field(None, description="Accommodation for the day, if applicable")
    transportation: List[Transportation] = Field([], description="List of transport details for the day")

class EventItinerary(BaseModel):
    event_name: str = Field(..., description="Name of the event, e.g., \"Annual Team Building Retreat\"")
    dates: List[str] = Field(..., description="List of dates for the event, e.g., [\"2024-10-26\", \"2024-10-27\"]")
    attendees: int = Field(..., description="Number of attendees, e.g., 50")
    budget: Optional[str] = Field(None, description="Optional budget for the event")
    daily_schedule: List[DailySchedule] = Field(..., description="Detailed daily schedule for the event")

    @validator("dates", pre=True, each_item=True)
    def validate_date_format(cls, v):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

class ItineraryGenerator:
    def __init__(self):
        pass

    def generate_itinerary(self, event_description: str) -> Dict[str, Any]:
        simulated_raw_json = """
{
    "event_name": "Corporate Team Building Retreat",
    "dates": ["2024-11-15", "2024-11-16"],
    "attendees": 50,
    "budget": "$10,000 - $15,000",
    "daily_schedule": [
        {
            "day": "Day 1",
            "activities": [
                {
                    "time": "09:00 AM - 10:00 AM",
                    "description": "Welcome and Registration",
                    "location": "Hotel Lobby",
                    "type": "Logistics"
                },
                {
                    "time": "10:00 AM - 12:30 PM",
                    "description": "Outdoor Obstacle Course",
                    "location": "Adventure Park",
                    "type": "Team Building"
                }
            ],
            "meals": [
                {
                    "time": "12:30 PM - 01:30 PM",
                    "description": "Lunch Buffet",
                    "cuisine": "International"
                },
                {
                    "time": "07:00 PM - 09:00 PM",
                    "description": "Gala Dinner",
                    "cuisine": "Fine Dining"
                }
            ],
            "accommodation": {
                "check_in_date": "2024-11-15",
                "check_out_date": "2024-11-16",
                "hotel_name": "Grand Lux Hotel",
                "address": "100 Grand Blvd"
            },
            "transportation": [
                {
                    "time": "08:30 AM",
                    "mode": "Bus",
                    "route": "Hotel to Adventure Park"
                }
            ]
        },
        {
            "day": "Day 2",
            "activities": [
                {
                    "time": "09:00 AM - 12:00 PM",
                    "description": "Strategy Workshop",
                    "location": "Hotel Conference Room",
                    "type": "Workshop"
                },
                {
                    "time": "01:30 PM - 03:00 PM",
                    "description": "Team Reflection and Wrap-up",
                    "location": "Hotel Conference Room",
                    "type": "Team Building"
                }
            ],
            "meals": [
                {
                    "time": "12:00 PM - 01:00 PM",
                    "description": "Lunch",
                    "cuisine": "Local"
                }
            ],
            "accommodation": None,
            "transportation": [
                {
                    "time": "03:30 PM",
                    "mode": "Bus",
                    "route": "Hotel to Airport"
                }
            ]
        }
    ]
}
        """
        
        try:
            parsed_data = EventItinerary.model_validate_json(simulated_raw_json)
            return parsed_data.model_dump()
        except ValidationError as e:
            print(f"Validation Error during simulated parsing: {e}")
            raise

app = FastAPI(
    title="AI Event Itinerary Generator",
    description="Generates structured event itineraries from natural language descriptions."
)

class EventRequest(BaseModel):
    event_description: str

generator = ItineraryGenerator()

@app.post("/generate-itinerary", response_model=EventItinerary)
async def generate_event_itinerary(request: EventRequest) -> Dict[str, Any]:
    try:
        itinerary = generator.generate_itinerary(request.event_description)
        return itinerary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate itinerary: {str(e)}")