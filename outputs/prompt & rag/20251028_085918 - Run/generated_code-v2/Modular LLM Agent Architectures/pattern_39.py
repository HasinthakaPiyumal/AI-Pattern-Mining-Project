
import os
import requests
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

# --- 1. LLM Core Setup ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# --- 2. Memory Module Setup ---
# Short-term Memory (Conversational Context)
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5, # Stores last 5 turns
    return_messages=True
)

# Long-term Memory (User Preferences & Travel History)
# In-memory Chroma for demonstration. For persistence, configure a directory.
vector_db = Chroma(embedding_function=embeddings, persist_directory=None) # No persistence for this example

def add_user_preference(user_id: str, preference: str):
    doc = {"user_id": user_id, "content": preference}
    vector_db.add_texts(texts=[preference], metadatas=[doc])
    return f"Preference added for user {user_id}."

def get_user_preferences(user_id: str, query: str = "") -> str:
    if query:
        # Retrieve relevant preferences based on query
        results = vector_db.similarity_search_with_score(query, k=3, filter={"user_id": user_id})
        if results:
            return "\n".join([r.page_content for r, _ in results])
    # Fallback to general preferences if query is empty or no specific match
    docs = vector_db.get(where={"user_id": user_id}) # This might not be ideal for general retrieval without specific query
    if docs and 'documents' in docs:
        return "\n".join(docs['documents'])
    return "No specific preferences found for this user or query."

class PreferenceInput(BaseModel):
    user_id: str = Field(description="ID of the user")
    preference: str = Field(description="The preference to store (e.g., 'prefers direct flights', 'likes historical sites').")

class PreferenceQueryInput(BaseModel):
    user_id: str = Field(description="ID of the user")
    query: str = Field(description="A query to retrieve specific preferences (e.g., 'flight preferences', 'food restrictions').")


# --- 3. Tool Use Module (Dummy Implementations) ---
# In a real application, these would make actual API calls.

def dummy_flight_search(destination: str, departure_date: str, return_date: str, passengers: int = 1) -> str:
    return json.dumps({
        "status": "success",
        "flights": [
            {"airline": "Example Air", "flight_number": "EA101", "price": 500, "departure_time": "08:00", "arrival_time": "14:00"},
            {"airline": "Global Fly", "flight_number": "GF202", "price": 550, "departure_time": "09:30", "arrival_time": "15:30"}
        ],
        "message": f"Found flights to {destination} from {departure_date} to {return_date} for {passengers} person(s)."
    })

def dummy_hotel_booking(location: str, check_in: str, check_out: str, guests: int = 1) -> str:
    return json.dumps({
        "status": "success",
        "hotels": [
            {"name": "Grand Hyatt Dummy", "stars": 5, "price_per_night": 200, "availability": True},
            {"name": "Budget Stay Inn", "stars": 3, "price_per_night": 80, "availability": True}
        ],
        "message": f"Found hotels in {location} from {check_in} to {check_out} for {guests} guest(s)."
    })

def dummy_activity_search(location: str, date: str, activity_type: str = "") -> str:
    return json.dumps({
        "status": "success",
        "activities": [
            {"name": "City Tour", "type": "sightseeing", "price": 50},
            {"name": "Cooking Class", "type": "culinary", "price": 75}
        ],
        "message": f"Found {activity_type} activities in {location} on {date}."
    })

def dummy_weather_forecast(location: str, date: str) -> str:
    return json.dumps({
        "status": "success",
        "location": location,
        "date": date,
        "forecast": "Sunny with a chance of clouds, 25°C",
        "message": f"Weather forecast for {location} on {date}."
    })

def dummy_calendar_integration(event_name: str, date: str, time: str, location: str = "") -> str:
    return json.dumps({
        "status": "success",
        "event_id": "cal_12345",
        "message": f"Added '{event_name}' to calendar on {date} at {time} in {location}."
    })

# Define Pydantic models for tool input validation
class FlightSearchInput(BaseModel):
    destination: str = Field(description="The destination city for the flight.")
    departure_date: str = Field(description="The departure date (YYYY-MM-DD).")
    return_date: str = Field(description="The return date (YYYY-MM-DD).")
    passengers: int = Field(default=1, description="Number of passengers.")

class HotelBookingInput(BaseModel):
    location: str = Field(description="The city or area for the hotel booking.")
    check_in: str = Field(description="The check-in date (YYYY-MM-DD).")
    check_out: str = Field(description="The check-out date (YYYY-MM-DD).")
    guests: int = Field(default=1, description="Number of guests.")

class ActivitySearchInput(BaseModel):
    location: str = Field(description="The location for the activity search.")
    date: str = Field(description="The date for the activity (YYYY-MM-DD).")
    activity_type: str = Field(default="", description="Optional type of activity (e.g., 'cultural', 'adventure').")

class WeatherForecastInput(BaseModel):
    location: str = Field(description="The city or region for the weather forecast.")
    date: str = Field(description="The date for the weather forecast (YYYY-MM-DD).")

class CalendarIntegrationInput(BaseModel):
    event_name: str = Field(description="Name of the event to add to the calendar.")
    date: str = Field(description="Date of the event (YYYY-MM-DD).")
    time: str = Field(description="Time of the event (HH:MM).")
    location: str = Field(default="", description="Optional location of the event.")


tools = [
    Tool(
        name="FlightSearch",
        func=dummy_flight_search,
        description="Searches for flights based on destination, dates, and number of passengers.",
        args_schema=FlightSearchInput
    ),
    Tool(
        name="HotelBooking",
        func=dummy_hotel_booking,
        description="Searches and simulates booking hotels based on location, dates, and number of guests.",
        args_schema=HotelBookingInput
    ),
    Tool(
        name="ActivitySearch",
        func=dummy_activity_search,
        description="Searches for activities in a given location and date, with an optional activity type.",
        args_schema=ActivitySearchInput
    ),
    Tool(
        name="WeatherForecast",
        func=dummy_weather_forecast,
        description="Gets the weather forecast for a specific location and date.",
        args_schema=WeatherForecastInput
    ),
    Tool(
        name="CalendarIntegration",
        func=dummy_calendar_integration,
        description="Adds an event to the user's calendar.",
        args_schema=CalendarIntegrationInput
    ),
    Tool(
        name="AddUserPreference",
        func=lambda user_id, preference: add_user_preference(user_id, preference),
        description="Adds a user's preference to their long-term memory for future reference.",
        args_schema=PreferenceInput
    ),
     Tool(
        name="GetUserPreferences",
        func=lambda user_id, query: get_user_preferences(user_id, query),
        description="Retrieves user preferences from long-term memory based on user ID and an optional query.",
        args_schema=PreferenceQueryInput
    )
]

# --- 4. Agent Setup ---
# The planning module is implicitly handled by the LLM's reasoning within the agent executor.

# Agent initialization
# We use AgentType.OPENAI_FUNCTIONS because our LLM is OpenAI and tools are defined with Pydantic schemas.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS, # Optimized for function calling models
    verbose=True,
    agent_kwargs={
        "memory_prompts": [conversational_memory.prompt],
        "input_keys": ["input", "chat_history"],
    },
    memory=conversational_memory,
    handle_parsing_errors=True
)

# --- Main Interaction Loop ---
if __name__ == "__main__":
    print("\nWelcome to the Intelligent Travel Assistant Agent (ITAA)!\n")
    print("I can help you plan, search, and manage your trips.\n")
    print("Type 'exit' or 'quit' to end the session.")

    # Example: Add an initial preference for a dummy user
    print(add_user_preference("user123", "prefers luxury hotels and cultural tours in Europe."))
    print(add_user_preference("user123", "dietary restriction: vegetarian."))
    print("\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("ITAA: Goodbye! Happy travels!")
            break

        try:
            # The agent will automatically use the conversational_memory and retrieve from vector_db if needed
            response = agent.invoke({"input": user_input, "chat_history": conversational_memory.load_memory_variables({})["chat_history"]})
            print(f"ITAA: {response['output']}")
        except Exception as e:
            print(f"ITAA: An error occurred: {e}")
            print("ITAA: I'm sorry, I encountered a problem. Could you please rephrase your request?")

