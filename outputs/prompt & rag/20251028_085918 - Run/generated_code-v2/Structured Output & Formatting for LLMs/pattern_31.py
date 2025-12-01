import os
import json
from typing import List, Literal

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import create_structured_output_runnable

class Activity(BaseModel):
    name: str = Field(description="Name of the activity or event")
    time: str = Field(description="Time of the activity, e.g., '9:00 AM' or 'Lunchtime'")
    description: str = Field(description="A brief description of the activity")
    category: Literal["transport", "food", "attraction", "accommodation", "other"] = Field(description="Category of the activity")

class DayPlan(BaseModel):
    date: str = Field(description="Date of the plan for the day, in YYYY-MM-DD format")
    activities: List[Activity] = Field(description="List of activities planned for the day")

class TravelPlan(BaseModel):
    destination: str = Field(description="The travel destination")
    duration_days: int = Field(description="Total number of days for the trip")
    days: List[DayPlan] = Field(description="A list of daily plans for the entire trip")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that generates detailed travel plans in JSON format."),
    ("human", "Generate a travel plan for {user_preferences}.")
])

structured_llm_chain = create_structured_output_runnable(TravelPlan, llm, prompt)

def generate_travel_plan(user_preferences: str) -> TravelPlan:
    return structured_llm_chain.invoke({"user_preferences": user_preferences})

if __name__ == "__main__":
    # Make sure to set your OPENAI_API_KEY environment variable
    # For example: os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

    user_input = "plan a 3-day trip to Paris, interested in museums and good food, moderate budget"
    print(f"Generating travel plan for: {user_input}")
    travel_plan = generate_travel_plan(user_input)
    print("\n--- Generated Travel Plan ---")
    print(json.dumps(travel_plan.dict(), indent=2))