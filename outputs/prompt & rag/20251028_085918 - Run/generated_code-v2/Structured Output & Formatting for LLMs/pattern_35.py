import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Ensure you have your OpenAI API key set as an environment variable
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

app = FastAPI(
    title="AI-Powered Personalized Travel Itinerary Generator",
    description="Generates personalized multi-day travel itineraries in a structured JSON format."
)

# --- Pydantic Models for Structured Output ---

class Activity(BaseModel):
    time: str = Field(..., description="Time of the activity (e.g., '9:00 AM', 'Lunchtime')")
    description: str = Field(..., description="Detailed description of the activity")
    location: Optional[str] = Field(None, description="Location of the activity")
    category: Optional[str] = Field(None, description="Category of the activity (e.g., 'Sightseeing', 'Meal', 'Transportation')")

class Day(BaseModel):
    day_number: int = Field(..., description="The day number in the itinerary")
    theme: Optional[str] = Field(None, description="Overall theme or focus for the day")
    activities: List[Activity] = Field(..., description="List of activities for the day")

class Itinerary(BaseModel):
    destination: str = Field(..., description="The main travel destination")
    start_date: str = Field(..., description="Start date of the trip (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date of the trip (YYYY-MM-DD)")
    total_days: int = Field(..., description="Total number of days for the trip")
    summary: str = Field(..., description="A brief summary of the entire trip itinerary")
    days: List[Day] = Field(..., description="Detailed breakdown of activities for each day")

# --- LLM Setup ---

# Initialize the LLM (make sure OPENAI_API_KEY is set in your environment)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Setup the parser for the Pydantic model
parser = PydanticOutputParser(pydantic_object=Itinerary)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that generates detailed and personalized travel itineraries. Your output MUST be in the following JSON format:\n{format_instructions}\nStrictly adhere to the schema, including all fields. If a field is optional and you don't have information, omit it or set to null, but ensure required fields are present and correctly formatted."),
    ("human", "Generate a {total_days}-day travel itinerary for {destination} starting on {start_date} for {travel_companions}. Interests include: {interests}. Budget: {budget}.")
])

# Create the LLM chain
itinerary_chain = prompt | llm | parser

# --- FastAPI Endpoint ---

class ItineraryRequest(BaseModel):
    destination: str = Field(..., example="Paris, France")
    start_date: str = Field(..., example="2024-09-15")
    end_date: str = Field(..., example="2024-09-20")
    interests: List[str] = Field(..., example=["history", "art museums", "food tours"])
    budget: str = Field(..., example="medium (mid-range hotels, nice restaurants)")
    travel_companions: str = Field(..., example="a couple with no kids")

@app.post("/generate-itinerary", response_model=Itinerary)
async def generate_travel_itinerary(request: ItineraryRequest):
    """
    Generates a personalized travel itinerary based on user preferences.
    The output is a structured JSON adhering to a predefined schema.
    """
    try:
        # Calculate total days
        from datetime import date
        d1 = date.fromisoformat(request.start_date)
        d2 = date.fromisoformat(request.end_date)
        total_days = (d2 - d1).days + 1
        if total_days <= 0:
            raise ValueError("End date must be after or on the start date.")

        response = await itinerary_chain.ainvoke({
            "destination": request.destination,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "total_days": total_days,
            "interests": ", ".join(request.interests),
            "budget": request.budget,
            "travel_companions": request.travel_companions
        })
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error generating itinerary: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate itinerary.")

# To run this application:
# 1. Save the code as main.py
# 2. Install dependencies: pip install fastapi uvicorn pydantic langchain-openai
# 3. Set your OpenAI API key as an environment variable: export OPENAI_API_KEY="your_key"
# 4. Run the app: uvicorn main:app --reload
# 5. Access the API at http://127.0.0.1:8000/docs for interactive documentation.