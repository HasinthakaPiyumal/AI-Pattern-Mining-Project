import json
from typing import List, Optional
from pydantic import BaseModel, ValidationError


class Activity(BaseModel):
    time: str
    description: str


class DailyPlan(BaseModel):
    day: str
    date: str
    activities: List[Activity]


class Accommodation(BaseModel):
    type: str
    name: str
    location: str


class Transportation(BaseModel):
    mode: str
    details: str


class TravelPlan(BaseModel):
    destination: str
    start_date: str
    end_date: str
    travelers: int
    accommodation: Optional[Accommodation] = None
    transportation: List[Transportation]
    daily_plan: List[DailyPlan]


def simulate_llm_free_form_plan(user_request: str) -> str:
    if "Paris" in user_request and "7-day" in user_request:
        return """Here's a wonderful 7-day trip to Paris for 2 people in July, focusing on culture and good food. You'll stay at Hotel du Louvre. For transportation, mostly metro and some walking. 
Day 1 (July 1st): Arrive, check in. Evening: Eiffel Tower visit. Dinner at a classic French bistro. 
Day 2 (July 2nd): Morning: Louvre Museum. Afternoon: Walk around Tuileries Garden. Evening: Seine River cruise. 
Day 3 (July 3rd): Morning: Notre Dame (exterior). Afternoon: Latin Quarter exploration. Dinner in Saint-Germain-des-Prés. 
Day 4 (July 4th): Day trip to Versailles Palace. 
Day 5 (July 5th): Morning: Montmartre, Sacré-Cœur Basilica. Afternoon: Explore art galleries. Evening: Moulin Rouge show (optional). 
Day 6 (July 6th): Morning: Musée d'Orsay. Afternoon: Shopping on Champs-Élysées. Farewell dinner at a Michelin-star restaurant. 
Day 7 (July 7th): Departure.
"""
    else:
        return "I can't generate a detailed plan for that request yet. Please try Paris for 7 days."


def simulate_llm_structured_output(free_form_plan: str, schema: BaseModel) -> Optional[dict]:
    # In a real application, this would be an LLM call instructed to output JSON.
    # For simulation, we'll try to parse a predefined JSON string.
    
    # Simulate LLM trying to generate JSON based on the free-form plan
    if "Eiffel Tower visit" in free_form_plan:
        json_output = {
            "destination": "Paris, France",
            "start_date": "2023-07-01",
            "end_date": "2023-07-07",
            "travelers": 2,
            "accommodation": {
                "type": "Hotel",
                "name": "Hotel du Louvre",
                "location": "Paris"
            },
            "transportation": [
                {"mode": "Metro", "details": "Paris Metro system"},
                {"mode": "Walking", "details": "Around city centers"}
            ],
            "daily_plan": [
                {
                    "day": "Day 1",
                    "date": "2023-07-01",
                    "activities": [
                        {"time": "Evening", "description": "Eiffel Tower visit"},
                        {"time": "Evening", "description": "Dinner at a classic French bistro"}
                    ]
                },
                {
                    "day": "Day 2",
                    "date": "2023-07-02",
                    "activities": [
                        {"time": "Morning", "description": "Louvre Museum"},
                        {"time": "Afternoon", "description": "Walk around Tuileries Garden"},
                        {"time": "Evening", "description": "Seine River cruise"}
                    ]
                },
                {
                    "day": "Day 3",
                    "date": "2023-07-03",
                    "activities": [
                        {"time": "Morning", "description": "Notre Dame (exterior)"},
                        {"time": "Afternoon", "description": "Latin Quarter exploration"},
                        {"time": "Evening", "description": "Dinner in Saint-Germain-des-Prés"}
                    ]
                },
                {
                    "day": "Day 4",
                    "date": "2023-07-04",
                    "activities": [
                        {"time": "Full Day", "description": "Day trip to Versailles Palace"}
                    ]
                },
                {
                    "day": "Day 5",
                    "date": "2023-07-05",
                    "activities": [
                        {"time": "Morning", "description": "Montmartre, Sacré-Cœur Basilica"},
                        {"time": "Afternoon", "description": "Explore art galleries"},
                        {"time": "Evening", "description": "Moulin Rouge show (optional)"}
                    ]
                },
                {
                    "day": "Day 6",
                    "date": "2023-07-06",
                    "activities": [
                        {"time": "Morning", "description": "Musée d'Orsay"},
                        {"time": "Afternoon", "description": "Shopping on Champs-Élysées"},
                        {"time": "Evening", "description": "Farewell dinner at a Michelin-star restaurant"}
                    ]
                },
                {
                    "day": "Day 7",
                    "date": "2023-07-07",
                    "activities": [
                        {"time": "Morning", "description": "Departure"}
                    ]
                }
            ]
        }
    else:
        # Simulate an LLM failure to produce valid JSON or follow the plan
        print("Simulating LLM failure to generate valid JSON...")
        return {"error": "Could not generate structured plan"}

    try:
        # Validate against the Pydantic schema
        validated_plan = schema.parse_obj(json_output)
        return validated_plan.dict()
    except ValidationError as e:
        print(f"Validation Error: {e.json()}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during JSON parsing/validation: {e}")
        return None


if __name__ == "__main__":
    user_request = "Plan a 7-day trip to Paris for 2 people in July, focusing on culture and good food."
    print(f"\nUser Request: {user_request}")

    # Step 1: Generate free-form natural language plan
    print("\n--- Generating Free-form Plan ---")
    free_form_plan = simulate_llm_free_form_plan(user_request)
    print(free_form_plan)

    if "can't generate" not in free_form_plan:
        # Step 2: Convert to structured plan using LLM and Pydantic
        print("\n--- Converting to Structured Plan ---")
        structured_plan_data = simulate_llm_structured_output(free_form_plan, TravelPlan)

        if structured_plan_data:
            print("\n--- Structured Travel Plan (JSON) ---")
            print(json.dumps(structured_plan_data, indent=2))
            
            # Example of accessing structured data
            print(f"\nDestination: {structured_plan_data['destination']}")
            print(f"Number of Travelers: {structured_plan_data['travelers']}")
            print(f"First activity on Day 1: {structured_plan_data['daily_plan'][0]['activities'][0]['description']}")
        else:
            print("Failed to generate a valid structured travel plan.")
    else:
        print("Skipping structured conversion due to initial plan generation failure.")

    print("\n--- Demonstrating failure case (simulated) ---")
    bad_free_form_plan = "Just some random text without structure."
    print(f"Attempting to convert: '{bad_free_form_plan}'")
    failed_structured_plan = simulate_llm_structured_output(bad_free_form_plan, TravelPlan)
    if not failed_structured_plan:
        print("Successfully handled simulated failure to produce valid structured output.")
