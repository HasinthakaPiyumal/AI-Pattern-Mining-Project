from pydantic import BaseModel, Field
from typing import List, Union, Optional
import datetime

class Transportation(BaseModel):
    type: str = Field(..., description="Type of transportation (e.g., 'Flight', 'Train', 'Taxi', 'Metro')")
    details: str = Field(..., description="Specific details about the transportation (e.g., flight number, departure/arrival times, route)")
    time: Optional[datetime.time] = Field(None, description="Time of transportation activity")

class Accommodation(BaseModel):
    name: str = Field(..., description="Name of the accommodation (e.g., hotel, Airbnb)")
    check_in: Optional[datetime.date] = Field(None, description="Check-in date")
    check_out: Optional[datetime.date] = Field(None, description="Check-out date")
    location: Optional[str] = Field(None, description="Location or address of the accommodation")
    time: Optional[datetime.time] = Field(None, description="Time for check-in/check-out, if specified")

class Meal(BaseModel):
    type: str = Field(..., description="Type of meal (e.g., 'Breakfast', 'Lunch', 'Dinner', 'Snack')")
    restaurant: Optional[str] = Field(None, description="Name of the restaurant or place to eat")
    time: Optional[datetime.time] = Field(None, description="Time of the meal")
    details: Optional[str] = Field(None, description="Additional details about the meal")

class Attraction(BaseModel):
    name: str = Field(..., description="Name of the attraction or activity")
    time: Optional[datetime.time] = Field(None, description="Suggested time for the attraction")
    details: Optional[str] = Field(None, description="Additional details about the attraction")

# Define a Union type for all possible activity types
Activity = Union[Transportation, Accommodation, Meal, Attraction]

class DayPlan(BaseModel):
    date: datetime.date = Field(..., description="Date of the plan for this day")
    activities: List[Activity] = Field(..., description="List of activities planned for the day")

class TravelItinerary(BaseModel):
    destination: str = Field(..., description="The travel destination")
    start_date: datetime.date = Field(..., description="The start date of the trip")
    end_date: datetime.date = Field(..., description="The end date of the trip")
    budget: Optional[Union[float, str]] = Field(None, description="Estimated budget for the trip (e.g., 'Luxury', 'Mid-range', 'Economy' or a numerical value)")
    summary: Optional[str] = Field(None, description="A brief summary of the itinerary")
    days: List[DayPlan] = Field(..., description="List of daily plans for the itinerary")

# Example usage (for testing/demonstration):
if __name__ == "__main__":
    # Create an example itinerary
    itinerary = TravelItinerary(
        destination="Paris, France",
        start_date=datetime.date(2023, 10, 26),
        end_date=datetime.date(2023, 10, 28),
        budget=1500.00,
        summary="A lovely 3-day trip to explore the best of Paris.",
        days=[
            DayPlan(
                date=datetime.date(2023, 10, 26),
                activities=[
                    Transportation(type="Flight", details="Flight AA123, arrive 10:00 AM"),
                    Accommodation(name="Hotel Le Littré", check_in=datetime.date(2023, 10, 26), location="9 Rue Jean-Jacques Rousseau"),
                    Meal(type="Dinner", restaurant="Le Comptoir du Relais", time=datetime.time(19, 0)),
                ],
            ),
            DayPlan(
                date=datetime.date(2023, 10, 27),
                activities=[
                    Attraction(name="Louvre Museum", time=datetime.time(9, 0)),
                    Meal(type="Lunch", details="Local bistro"),
                    Attraction(name="Montmartre Exploration"),
                    Meal(type="Dinner", restaurant="Charming restaurant in Le Marais", time=datetime.time(20, 0)),
                ],
            ),
            DayPlan(
                date=datetime.date(2023, 10, 28),
                activities=[
                    Attraction(name="Seine River Cruise", time=datetime.time(10, 0)),
                    Meal(type="Lunch"),
                    Transportation(type="Flight", details="Flight AA456 from Charles de Gaulle Airport, depart 4:00 PM"),
                ],
            ),
        ],
    )
    print(itinerary.model_dump_json(indent=2))
