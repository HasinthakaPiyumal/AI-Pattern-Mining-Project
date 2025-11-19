from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv
import uuid
import datetime

# Load environment variables
load_dotenv()

# --- Pydantic Models ---
class TravelPreferences(BaseModel):
    budget: Optional[str] = None
    start_date: str = Field(..., example="2024-08-01")
    end_date: str = Field(..., example="2024-08-07")
    destinations: List[str] = Field(..., example=["Paris", "Rome"])
    interests: List[str] = Field(..., example=["museums", "food", "history"])
    travelers: int = Field(..., ge=1, example=2)
    travel_style: Optional[str] = None # e.g., "relaxed", "adventurous", "luxury"

class ItineraryItem(BaseModel):
    item_type: str = Field(..., example="accommodation")
    name: str = Field(..., example="Eiffel Tower")
    details: str = Field(..., example="Visit the iconic landmark.")
    start_time: Optional[str] = Field(None, example="2024-08-02T10:00:00")
    end_time: Optional[str] = Field(None, example="2024-08-02T12:00:00")
    cost: Optional[str] = Field(None, example="€20")

class ItineraryRequest(BaseModel):
    user_id: str = Field(..., example="user123")
    preferences: TravelPreferences

class ItineraryResponse(BaseModel):
    itinerary_id: str
    user_id: str
    plan: List[ItineraryItem]
    status: str = Field(..., example="planned") # e.g., "planned", "updated", "executed"

class FeedbackRequest(BaseModel):
    itinerary_id: str
    feedback_text: str = Field(..., example="The suggested restaurant was too expensive.")
    real_time_update: Optional[str] = Field(None, example="Flight LH456 delayed by 3 hours.")

# --- Mock Tools ---
# In a real application, these would call external APIs (e.g., flight booking, hotel APIs)
def search_flights(destination: str, start_date: str, end_date: str, travelers: int) -> str:
    """Simulates searching for flights."""
    return f"Found several flights to {destination} from {start_date} to {end_date} for {travelers} people. Prices starting from $500. Example: Flight BA286 (09:00-11:00) {start_date} from London to {destination}."

def book_hotel(destination: str, check_in: str, check_out: str, travelers: int, preferences: str = "") -> str:
    """Simulates booking a hotel."""
    return f"Booked a hotel in {destination} for {travelers} people from {check_in} to {check_out} with preferences: {preferences}. Example: Grand Hyatt {destination} for 3 nights."

def plan_activity(destination: str, activity_type: str, date: str, interests: str = "") -> str:
    """Simulates planning an activity."""
    return f"Planned a {activity_type} in {destination} on {date} based on interests: {interests}. Example: Guided tour of Louvre Museum on {date} at 10:00 AM."

def get_current_weather(location: str, date: str) -> str:
    """Simulates fetching current or forecasted weather for a location and date."""
    if datetime.datetime.strptime(date, "%Y-%m-%d").date() > datetime.date.today() + datetime.timedelta(days=7):
        return f"Weather forecast for {location} on {date}: Sunny with a high of 25°C. "
    return f"Current weather in {location}: Partly cloudy, 22°C."

# Define the tools for the agents
tools = [
    Tool(
        name="SearchFlights",
        func=search_flights,
        description="Use this tool to search for flight options given a destination, start date, end date, and number of travelers."
    ),
    Tool(
        name="BookHotel",
        func=book_hotel,
        description="Use this tool to book a hotel given a destination, check-in date, check-out date, number of travelers, and optional preferences."
    ),
    Tool(
        name="PlanActivity",
        func=plan_activity,
        description="Use this tool to plan an activity given a destination, activity type (e.g., 'sightseeing', 'dining', 'adventure'), date, and user interests."
    ),
    Tool(
        name="GetCurrentWeather",
        func=get_current_weather,
        description="Use this tool to get current or forecasted weather conditions for a location and date."
    )
]

# --- LLM Initialization ---
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# --- Custom Output Parser for JSON ---
class ItineraryOutputParser(JsonOutputParser):
    def parse(self, text: str) -> List[ItineraryItem]:
        try:
            parsed_data = super().parse(text)
            if not isinstance(parsed_data, list):
                # If the LLM returns a single dict instead of a list, wrap it
                if isinstance(parsed_data, dict):
                    return [ItineraryItem(**parsed_data)]
                raise ValueError("Expected a list of itinerary items or a single item dict.")
            return [ItineraryItem(**item) for item in parsed_data]
        except Exception as e:
            raise ValueError(f"Failed to parse itinerary: {e}\nOriginal text: {text}")

itinerary_output_parser = ItineraryOutputParser()

# --- LangChain Agents ---

# 1. Task Decomposition Agent
task_decomposition_prompt_template = PromptTemplate.from_template(
    """You are an expert travel planner. Given a user's travel preferences, decompose the overall trip planning into a list of specific sub-tasks.
    Focus on high-level categories like 'Accommodation', 'Flights', 'Activities', 'Transportation within destination'.
    
    User Preferences:
    {preferences}
    
    Output a comma-separated list of sub-tasks. Example: "Book Flights, Find Hotels, Plan Activities in Paris, Plan Activities in Rome, Arrange Local Transport".
    """
)
task_decomposition_chain = task_decomposition_prompt_template | llm.bind(stop=["\nOutcome:"]) | JsonOutputParser() # Not strictly JSON output, but will parse comma-separated string

# 2. Constraint Satisfaction Agent (integrated into Main Planner or as a step)
# For simplicity, we'll guide the main planner to satisfy constraints.

# 3. Main Planner Agent
main_planner_prompt_template = PromptTemplate.from_template(
    """You are a world-class AI travel planner. Your goal is to create a detailed, personalized travel itinerary based on the user's preferences, 
    addressing all sub-tasks and strictly adhering to constraints like budget, dates, and interests.
    
    You have access to the following tools: {tools}.
    
    Use the tools to generate each part of the itinerary.
    
    User Request: {request}
    Decomposed Sub-tasks: {sub_tasks}
    Current Itinerary (if any): {current_itinerary}
    
    Plan the trip step-by-step, using the tools as necessary. For each item in the itinerary, provide details including type, name, details, start_time, end_time, and cost. 
    Format the final output as a JSON list of ItineraryItem objects. If a field is not applicable, set it to null.
    Example ItineraryItem: {{"item_type": "flight", "name": "Flight BA286", "details": "London to Paris", "start_time": "2024-08-01T09:00:00", "end_time": "2024-08-01T11:00:00", "cost": "$300"}}
    Begin!"""
)

main_planner_agent = create_react_agent(llm, tools, main_planner_prompt_template)
main_planner_executor = AgentExecutor(agent=main_planner_agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 4. Adaptive Execution / Refinement Agent
adaptive_execution_prompt_template = PromptTemplate.from_template(
    """You are an adaptive AI travel agent. Given an existing travel itinerary and new feedback or a real-time update, your task is to intelligently adjust and refine the itinerary.
    Prioritize user satisfaction and practicality while adhering to any remaining constraints.
    
    You have access to the following tools: {tools}.
    
    Current Itinerary:
    {current_itinerary}
    
    New Feedback / Real-time Update: {feedback}
    
    Think step by step. Identify which parts of the itinerary need modification. Use the tools if necessary to find alternative options. 
    Return the updated itinerary as a JSON list of ItineraryItem objects. Ensure the output is a valid JSON list. If a field is not applicable, set it to null.
    """
)

adaptive_execution_agent = create_react_agent(llm, tools, adaptive_execution_prompt_template)
adaptive_execution_executor = AgentExecutor(agent=adaptive_execution_agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- FastAPI Application ---
app = FastAPI(title="Smart Travel Itinerary Planner")

# In-memory database for itineraries (for demonstration purposes)
itineraries_db: Dict[str, ItineraryResponse] = {}

@app.post("/plan_itinerary", response_model=ItineraryResponse)
async def plan_itinerary(request: ItineraryRequest):
    try:
        itinerary_id = str(uuid.uuid4())

        # Step 1: Task Decomposition
        preferences_str = request.preferences.model_dump_json()
        decomposition_result = task_decomposition_chain.invoke({"preferences": preferences_str})
        # Assuming decomposition_result is a list or string that can be processed
        if isinstance(decomposition_result, str):
            sub_tasks = decomposition_result.split(', ')
        elif isinstance(decomposition_result, dict) and 'answer' in decomposition_result:
            sub_tasks = decomposition_result['answer'].split(', ')
        else:
            sub_tasks = ["Plan Flights", "Plan Accommodation", "Plan Activities"]
        print(f"Decomposed Sub-tasks: {sub_tasks}")

        # Step 2: Main Planning and Constraint Satisfaction (handled by main_planner_executor)
        main_planner_input = {
            "request": f"Create a travel itinerary for user {request.user_id} with preferences: {preferences_str}",
            "sub_tasks": ", ".join(sub_tasks),
            "current_itinerary": ""
        }
        
        main_planning_output = await main_planner_executor.ainvoke(main_planner_input)
        
        # Extract the final answer which should be JSON string of itinerary items
        itinerary_plan_str = main_planning_output.get('output', '[]')

        # Parse the itinerary plan using the custom parser
        itinerary_items = itinerary_output_parser.parse(itinerary_plan_str)
        
        response = ItineraryResponse(
            itinerary_id=itinerary_id,
            user_id=request.user_id,
            plan=itinerary_items,
            status="planned"
        )
        itineraries_db[itinerary_id] = response
        return response
    except Exception as e:
        print(f"Error planning itinerary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to plan itinerary: {e}")

@app.post("/update_itinerary", response_model=ItineraryResponse)
async def update_itinerary(request: FeedbackRequest):
    if request.itinerary_id not in itineraries_db:
        raise HTTPException(status_code=404, detail="Itinerary not found.")

    current_itinerary = itineraries_db[request.itinerary_id]

    try:
        current_plan_json = [item.model_dump_json() for item in current_itinerary.plan]
        
        adaptive_input = {
            "current_itinerary": f"[{', '.join(current_plan_json)}]",
            "feedback": f"User feedback: {request.feedback_text}. Real-time update: {request.real_time_update if request.real_time_update else 'None'}"
        }

        adaptive_planning_output = await adaptive_execution_executor.ainvoke(adaptive_input)
        
        updated_plan_str = adaptive_planning_output.get('output', '[]')
        
        updated_itinerary_items = itinerary_output_parser.parse(updated_plan_str)

        current_itinerary.plan = updated_itinerary_items
        current_itinerary.status = "updated"
        itineraries_db[request.itinerary_id] = current_itinerary # Update in DB

        return current_itinerary
    except Exception as e:
        print(f"Error updating itinerary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update itinerary: {e}")

@app.get("/get_itinerary/{itinerary_id}", response_model=ItineraryResponse)
async def get_itinerary(itinerary_id: str):
    if itinerary_id not in itineraries_db:
        raise HTTPException(status_code=404, detail="Itinerary not found.")
    return itineraries_db[itinerary_id]

# To run the FastAPI app, save this file as main.py and run: uvicorn main:app --reload
# Make sure you have OPENAI_API_KEY set in your environment variables or in a .env file.
