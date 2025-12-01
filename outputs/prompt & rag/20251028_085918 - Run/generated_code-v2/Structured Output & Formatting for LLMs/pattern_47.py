import os
import json
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

# Load environment variables from a .env file (if present)
load_dotenv()

# --- Pydantic Schemas for Structured Output ---

class Activity(BaseModel):
    """Represents a single activity or event within a travel plan."""
    name: str = Field(description="Name of the activity or event")
    type: str = Field(description="Type of activity (e.g., 'Attraction', 'Restaurant', 'Transportation', 'Accommodation')")
    time: Optional[str] = Field(None, description="Suggested time for the activity (e.g., '9:00 AM', 'Lunch', 'Evening')")
    description: Optional[str] = Field(None, description="Short description of the activity")
    location: Optional[str] = Field(None, description="Location or address of the activity")
    notes: Optional[str] = Field(None, description="Any additional notes or details for the activity")

class DailyPlan(BaseModel):
    """Represents the plan for a single day of the itinerary."""
    day: int = Field(description="Day number of the itinerary (e.g., 1, 2, 3)")
    date: str = Field(description="Date for this day's plan in YYYY-MM-DD format")
    theme: Optional[str] = Field(None, description="Optional theme or focus for the day (e.g., 'Historical Exploration', 'Food Tour')")
    activities: List[Activity] = Field(description="A list of activities planned for this specific day")

class TravelItinerary(BaseModel):
    """The main structured travel itinerary for the entire trip."""
    destination: str = Field(description="The travel destination (e.g., 'Paris, France')")
    start_date: str = Field(description="Start date of the trip in YYYY-MM-DD format")
    end_date: str = Field(description="End date of the trip in YYYY-MM-DD format")
    duration_days: int = Field(description="Total number of days for the trip")
    daily_plans: List[DailyPlan] = Field(description="A chronological list of daily plans for the entire itinerary")
    overall_notes: Optional[str] = Field(None, description="Any overall notes, tips, or important information for the trip")

# --- LLM Setup ---
# Ensure you have your OpenAI API key set as an environment variable (e.g., OPENAI_API_KEY)
# For better JSON adherence and complex plan generation, gpt-4o is recommended.
llm = ChatOpenAI(model="gpt-4o", temperature=0) 

# --- Structured Output Chain Definition ---
# Initialize the PydanticOutputParser with our TravelItinerary schema.
# This parser will attempt to parse the LLM's output into a TravelItinerary object.
parser = PydanticOutputParser(pydantic_object=TravelItinerary)

# Define the prompt template for the LLM.
# It explicitly instructs the LLM to generate JSON output conforming to the schema.
# The `format_instructions` are dynamically injected by the parser to guide the LLM.
prompt = PromptTemplate(
    template="""You are an expert travel planner. Your task is to generate a detailed, multi-day travel itinerary based on user preferences. 
    The output MUST be strictly in JSON format, adhering precisely to the provided Pydantic schema. Do not include any other text, 
    explanations, or comments outside the JSON.

    User Preferences:
    Destination: {destination}
    Start Date: {start_date}
    Number of Days: {num_days}
    Interests: {interests}
    Budget: {budget}
    Travelers: {travelers}

    Strictly follow the JSON schema below for the output:
    {format_instructions}

    Generate the JSON itinerary:
    """,
    input_variables=["destination", "start_date", "num_days", "interests", "budget", "travelers"],
    partial_variables={
        "format_instructions": parser.get_format_instructions() # Injects the JSON schema instructions
    },
)

# Construct the LangChain chain: Prompt -> LLM -> Parser
# The parser will automatically validate and convert the LLM's JSON string output
# into a Pydantic `TravelItinerary` object.
chain = prompt | llm | parser

def generate_itinerary(
    destination: str,
    start_date_str: str,
    num_days: int,
    interests: List[str],
    budget: str,
    travelers: str
) -> Optional[TravelItinerary]:
    """
    Generates a structured travel itinerary based on user preferences using an LLM.

    Args:
        destination: The desired travel destination (e.g., "Rome, Italy").
        start_date_str: The start date of the trip in "YYYY-MM-DD" format.
        num_days: The total number of days for the trip.
        interests: A list of user interests (e.g., ["history", "food", "museums"]).
        budget: The travel budget (e.g., "low", "mid-range", "luxury").
        travelers: Description of travelers (e.g., "2 adults", "family with 2 kids").

    Returns:
        A `TravelItinerary` Pydantic object if successful, None otherwise.
    """
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = start_date + timedelta(days=num_days - 1)
        end_date_str = end_date.strftime("%Y-%m-%d")

        print(f"Generating itinerary for {destination} starting {start_date_str} for {num_days} days...")

        # Invoke the LangChain chain with user inputs
        itinerary = chain.invoke({
            "destination": destination,
            "start_date": start_date_str,
            "num_days": num_days,
            "interests": ", ".join(interests), # Convert list to comma-separated string for prompt
            "budget": budget,
            "travelers": travelers
        })
        
        # Ensure the Pydantic model's date and duration fields are consistent with input
        itinerary.start_date = start_date_str
        itinerary.end_date = end_date_str
        itinerary.duration_days = num_days

        return itinerary
    except Exception as e:
        print(f"An error occurred during itinerary generation: {e}")
        print("Please ensure your OpenAI API key is correctly set and the LLM output conforms to the schema.")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    # !!! IMPORTANT: Make sure to set your OPENAI_API_KEY environment variable. !!!
    # You can do this by creating a .env file in the same directory as this script
    # with content like: OPENAI_API_KEY="your_api_key_here"

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run this example.")
    else:
        # Example User Preferences
        user_destination = "Kyoto, Japan"
        user_start_date = "2024-10-01" # YYYY-MM-DD
        user_num_days = 4
        user_interests = ["temples", "gardens", "traditional food", "cherry blossoms", "local culture"]
        user_budget = "mid-range to high"
        user_travelers = "2 adults"

        # Generate the structured itinerary
        structured_itinerary = generate_itinerary(
            destination=user_destination,
            start_date_str=user_start_date,
            num_days=user_num_days,
            interests=user_interests,
            budget=user_budget,
            travelers=user_travelers
        )

        if structured_itinerary:
            print("\n--- Successfully Generated Structured Itinerary ---")
            # Convert the Pydantic object to a dictionary and then pretty-print as JSON
            print(json.dumps(structured_itinerary.dict(), indent=2, ensure_ascii=False))

            # Demonstrating easy access to structured components
            print(f"\nDestination: {structured_itinerary.destination}")
            print(f"Trip Duration: {structured_itinerary.duration_days} days")
            if structured_itinerary.daily_plans:
                print(f"\nFirst day's date: {structured_itinerary.daily_plans[0].date}")
                print(f"First day's activities:")
                for activity in structured_itinerary.daily_plans[0].activities:
                    print(f"  - {activity.name} ({activity.type}, Time: {activity.time or 'Flexible'})")
        else:
            print("Failed to generate itinerary. Check error messages above.")
