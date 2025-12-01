import os
from dotenv import load_dotenv
from typing import List, Optional
import json

from pydantic import BaseModel, Field
import gradio as gr

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# 1. Environment Management
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file.")

# 2. Pydantic Models for Structured Output

class Attraction(BaseModel):
    name: str = Field(description="Name of the attraction or activity")
    type: str = Field(description="Type of attraction (e.g., historical site, museum, park, shopping)")
    description: str = Field(description="A brief description of the attraction")
    estimated_time_minutes: int = Field(description="Estimated time needed for the attraction in minutes")

class Meal(BaseModel):
    name: str = Field(description="Name of the restaurant or type of meal")
    cuisine: str = Field(description="Cuisine type (e.g., Italian, French, local)")
    estimated_cost_usd: float = Field(description="Estimated cost of the meal in USD")
    time_of_day: str = Field(description="When the meal will be (e.g., Breakfast, Lunch, Dinner)")

class Accommodation(BaseModel):
    name: str = Field(description="Name of the hotel or accommodation")
    type: str = Field(description="Type of accommodation (e.g., Hotel, Hostel, Airbnb)")
    address: str = Field(description="Address of the accommodation")
    check_in_date: str = Field(description="Check-in date in YYYY-MM-DD format")
    check_out_date: str = Field(description="Check-out date in YYYY-MM-DD format")
    estimated_cost_usd_per_night: float = Field(description="Estimated cost per night in USD")

class Transportation(BaseModel):
    type: str = Field(description="Type of transportation (e.g., walk, subway, bus, taxi, rental car)")
    details: str = Field(description="Specific details about the transportation, e.g., route, duration, company")
    estimated_cost_usd: float = Field(description="Estimated cost of this transportation segment in USD")

class DayPlan(BaseModel):
    day_number: int = Field(description="The sequential number of the day in the itinerary")
    date: str = Field(description="The date for this day in YYYY-MM-DD format")
    theme: Optional[str] = Field(description="An optional theme for the day, e.g., 'Historical Exploration', 'Foodie Tour'")
    activities: List[Attraction] = Field(description="List of attractions and activities for the day")
    meals: List[Meal] = Field(description="List of meals for the day")
    daily_transportation: List[Transportation] = Field(description="List of transportation segments for the day")

class TravelItinerary(BaseModel):
    destination: str = Field(description="The travel destination")
    start_date: str = Field(description="The start date of the trip in YYYY-MM-DD format")
    end_date: str = Field(description="The end date of the trip in YYYY-MM-DD format")
    duration_days: int = Field(description="Total number of days for the trip")
    overall_budget_usd: str = Field(description="The overall budget for the trip (e.g., 'moderate', 'luxury', '2000 USD')")
    accommodation: Accommodation = Field(description="Details about the main accommodation for the trip")
    daily_plans: List[DayPlan] = Field(description="A day-by-day breakdown of the itinerary")

# 3. LLM Setup
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0, api_key=OPENAI_API_KEY)

# Create the parser
parser = PydanticOutputParser(pydantic_object=TravelItinerary)

# 4. Prompt Engineering
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI travel planner. Your task is to generate a detailed, structured travel itinerary based on user preferences. "
     "The output must be in JSON format, strictly adhering to the provided Pydantic schema.\n{format_instructions}"),
    ("human", "Generate a travel plan for: {user_preferences}"),
])

# Create the chain
chain = prompt | llm | parser

# 5. Core Logic Function
def generate_itinerary(user_preferences: str) -> str:
    try:
        # Invoke the chain to get a Pydantic object
        itinerary_object = chain.invoke({
            "user_preferences": user_preferences,
            "format_instructions": parser.get_format_instructions(),
        })
        # Convert the Pydantic object to a JSON string with indentation for readability
        return json.dumps(itinerary_object.dict(), indent=2)
    except Exception as e:
        return f"An error occurred: {e}"

# 6. Gradio User Interface
iface = gr.Interface(
    fn=generate_itinerary,
    inputs=gr.Textbox(
        lines=5,
        label="Enter your travel preferences",
        placeholder="e.g., I want to visit Rome for 3 days in October, focusing on historical sites and good food, with a moderate budget. I prefer a boutique hotel."
    ),
    outputs=gr.Textbox(
        lines=20,
        label="Generated Itinerary (JSON)",
        interactive=False
    ),
    title="AI-Powered Structured Travel Planner",
    description="Enter your travel preferences in natural language, and the AI will generate a structured, day-by-day itinerary in JSON format."
)

if __name__ == "__main__":
    iface.launch()