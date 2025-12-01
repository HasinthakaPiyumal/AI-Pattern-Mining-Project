import json
from datetime import date, timedelta
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="AI-powered Travel Itinerary Generator",
    description="Generate structured travel itineraries using an LLM based on user preferences."
)

# Pydantic Models for structured input and output
class Activity(BaseModel):
    name: str = Field(..., description="Name of the activity or attraction.")
    time: str = Field(..., description="Suggested time for the activity (e.g., '9:00 AM', 'Morning', 'Lunch').")
    description: str = Field(..., description="Brief description of the activity.")
    location: Optional[str] = Field(None, description="Specific location or address of the activity.")
    estimated_cost_usd: Optional[float] = Field(None, description="Estimated cost of the activity in USD.")

class DayPlan(BaseModel):
    date: date = Field(..., description="Date for this day's plan.")
    theme: Optional[str] = Field(None, description="Optional theme or focus for the day.")
    activities: List[Activity] = Field(..., description="List of activities planned for the day.")
    accommodation: Optional[str] = Field(None, description="Name or description of accommodation for the night.")
    transportation_notes: Optional[str] = Field(None, description="Notes on transportation for the day.")
    meal_suggestions: Optional[List[str]] = Field(None, description="Suggestions for meals (e.g., 'Breakfast at hotel', 'Lunch at local cafe').")

class TravelItinerary(BaseModel):
    destination: str = Field(..., description="The travel destination.")
    start_date: date = Field(..., description="Start date of the trip.")
    end_date: date = Field(..., description="End date of the trip.")
    total_budget_usd: Optional[float] = Field(None, description="Estimated total budget for the trip in USD.")
    daily_plans: List[DayPlan] = Field(..., description="A detailed plan for each day of the trip.")
    notes: Optional[str] = Field(None, description="Any additional notes or tips for the trip.")

class TravelPreferences(BaseModel):
    destination: str = Field(..., example="Paris, France", description="Desired travel destination.")
    start_date: date = Field(..., example="2024-09-01", description="Start date of the trip.")
    end_date: date = Field(..., example="2024-09-07", description="End date of the trip.")
    interests: List[str] = Field(..., example=["museums", "food", "sightseeing"], description="List of user interests (e.g., 'history', 'art', 'adventure', 'relaxing').")
    budget_level: str = Field(..., example="medium", description="Budget level (e.g., 'low', 'medium', 'high', 'luxury').")
    number_of_travelers: int = Field(1, example=2, description="Number of people traveling.")
    specific_requests: Optional[str] = Field(None, example="Prefer walkable areas, avoid crowded tourist traps.", description="Any specific requests or preferences.")

# Mock LLM interaction function (replace with actual LLM call)
def get_llm_itinerary(preferences: TravelPreferences, output_schema: str) -> str:
    """
    Simulates an LLM call to generate a travel itinerary in JSON format.
    In a real application, this would integrate with an actual LLM service (e.g., OpenAI, Google Gemini).
    The prompt engineering here is crucial to instruct the LLM to output structured JSON.
    """
    prompt_template = f"""
    You are a world-class travel agent AI. Your task is to generate a detailed, multi-day travel itinerary in JSON format.
    The itinerary should be based on the following user preferences:

    Destination: {preferences.destination}
    Start Date: {preferences.start_date}
    End Date: {preferences.end_date}
    Interests: {', '.join(preferences.interests)}
    Budget Level: {preferences.budget_level}
    Number of Travelers: {preferences.number_of_travelers}
    Specific Requests: {preferences.specific_requests if preferences.specific_requests else 'None'}

    The output MUST be a JSON object that strictly conforms to the following Pydantic schema.
    Do not include any other text, comments, or explanations outside the JSON block.
    Ensure all required fields are present and data types match.

    JSON Schema:
    {output_schema}

    Generate the JSON itinerary now:
    """

    # This is a mock response. An actual LLM call would look something like:
    # from openai import OpenAI
    # client = OpenAI()
    # response = client.chat.completions.create(
    #     model="gpt-4-turbo-preview", # Or another suitable model
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
    #         {"role": "user", "content": prompt_template}
    #     ],
    #     response_format={ "type": "json_object" }
    # )
    # return response.choices[0].message.content

    # Mock structured response for demonstration:
    num_days = (preferences.end_date - preferences.start_date).days + 1
    mock_daily_plans = []
    for i in range(num_days):
        current_date = preferences.start_date + timedelta(days=i)
        mock_daily_plans.append({
            "date": current_date.isoformat(),
            "theme": f"Day {i+1}: Explore {preferences.destination}",
            "activities": [
                {
                    "name": "Morning Activity",
                    "time": "9:00 AM",
                    "description": "Visit a landmark related to user interests.",
                    "location": f"Landmark {i+1}",
                    "estimated_cost_usd": 30.0
                },
                {
                    "name": "Lunch",
                    "time": "1:00 PM",
                    "description": "Enjoy local cuisine.",
                    "location": "Local Restaurant",
                    "estimated_cost_usd": 25.0
                },
                {
                    "name": "Afternoon Activity",
                    "time": "3:00 PM",
                    "description": "Explore another point of interest or relax.",
                    "location": f"Attraction {i+1}",
                    "estimated_cost_usd": 20.0
                }
            ],
            "accommodation": "Cozy Hotel",
            "transportation_notes": "Public transport or walking.",
            "meal_suggestions": ["Breakfast at hotel", "Lunch at local cafe", "Dinner near hotel"]
        })

    mock_llm_response_data = {
        "destination": preferences.destination,
        "start_date": preferences.start_date.isoformat(),
        "end_date": preferences.end_date.isoformat(),
        "total_budget_usd": 100.0 * num_days, # Simple budget estimation
        "daily_plans": mock_daily_plans,
        "notes": "Enjoy your custom-generated trip to {preferences.destination}!"
    }
    return json.dumps(mock_llm_response_data, indent=2)

# FastAPI endpoint
@app.post("/generate_itinerary", response_model=TravelItinerary)
async def generate_itinerary(preferences: TravelPreferences):
    """
    Generates a structured multi-day travel itinerary based on user preferences.
    The output is a JSON object conforming to the TravelItinerary schema.
    """
    try:
        # Generate the JSON schema dynamically from the Pydantic model
        # This helps the LLM understand the expected structure
        output_schema_json = json.dumps(TravelItinerary.schema(), indent=2)

        # Get LLM response (mocked or actual API call)
        llm_raw_json_output = get_llm_itinerary(preferences, output_schema_json)

        # Parse the LLM's JSON output into our Pydantic model for validation and structured access
        itinerary = TravelItinerary.parse_raw(llm_raw_json_output)

        return itinerary
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM generated invalid JSON: {e}. Raw output: {llm_raw_json_output}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
