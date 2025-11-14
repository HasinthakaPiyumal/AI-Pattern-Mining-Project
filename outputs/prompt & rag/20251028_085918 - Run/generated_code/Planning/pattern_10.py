
import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import random
import chromadb
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

# --- 1. Pydantic Data Models --- #

class Activity(BaseModel):
    name: str
    description: str
    cost: float = 0.0
    duration_hours: float = 1.0
    type: str = "general"

class Flight(BaseModel):
    flight_number: str
    airline: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    price: float
    status: str = "On Time"

class Hotel(BaseModel):
    name: str
    location: str
    check_in: str
    check_out: str
    price_per_night: float
    rating: float

class Destination(BaseModel):
    city: str
    country: str
    start_date: str
    end_date: str
    activities: List[Activity] = []
    hotel: Optional[Hotel] = None
    transportation: str = ""

class Trip(BaseModel):
    trip_name: str
    destinations: List[Destination]
    total_budget: float
    current_cost: float = 0.0
    user_interests: List[str] = []
    accessibility_needs: List[str] = []
    status_messages: List[str] = []

class UserProfile(BaseModel):
    user_id: str
    preferences: Dict[str, Any]
    past_trips: List[Dict[str, Any]] = []

# --- 2. Simulated Tools/APIs --- #

# Placeholder for API key, ideally from environment variables
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

@tool
def search_flights(origin: str, destination: str, date: str) -> List[dict]:
    """Searches for available flights between origin and destination on a specific date."""
    st.session_state.status_messages.append(f"Searching flights from {origin} to {destination} on {date}...")
    time.sleep(1)
    flights = [
        Flight(flight_number="AA100", airline="American Airlines", origin=origin, destination=destination, departure_time="08:00", arrival_time="11:00", price=250.0, status="On Time").dict(),
        Flight(flight_number="DL201", airline="Delta", origin=origin, destination=destination, departure_time="09:30", arrival_time="12:30", price=280.0, status="On Time").dict(),
    ]
    st.session_state.status_messages.append(f"Found {len(flights)} flights.")
    return flights

@tool
def search_hotels(location: str, check_in: str, check_out: str, max_price_per_night: float = 500.0) -> List[dict]:
    """Searches for hotels in a given location for specified dates, with an optional maximum price per night."""
    st.session_state.status_messages.append(f"Searching hotels in {location} from {check_in} to {check_out}...")
    time.sleep(1)
    hotels = [
        Hotel(name="Grand Hyatt", location=location, check_in=check_in, check_out=check_out, price_per_night=200.0, rating=4.5).dict(),
        Hotel(name="Budget Inn", location=location, check_in=check_in, check_out=check_out, price_per_night=80.0, rating=3.0).dict()
    ]
    filtered_hotels = [h for h in hotels if h['price_per_night'] <= max_price_per_night]
    st.session_state.status_messages.append(f"Found {len(filtered_hotels)} hotels under ${max_price_per_night}.")
    return filtered_hotels

@tool
def get_activities(destination_city: str, interests: List[str] = [], accessibility: List[str] = []) -> List[dict]:
    """Retrieves popular activities and points of interest for a destination, optionally filtered by interests and accessibility needs."""
    st.session_state.status_messages.append(f"Getting activities for {destination_city} based on interests: {interests} and accessibility: {accessibility}...")
    time.sleep(1)
    all_activities = [
        Activity(name="Eiffel Tower", description="Iconic landmark", cost=25.0, duration_hours=2.0, type="sightseeing").dict(),
        Activity(name="Louvre Museum", description="World-famous art museum", cost=20.0, duration_hours=3.0, type="culture").dict(),
        Activity(name="Seine River Cruise", description="Scenic boat tour", cost=30.0, duration_hours=1.5, type="leisure").dict(),
        Activity(name="Wheelchair Accessible City Tour", description="Guided tour for mobility impaired", cost=50.0, duration_hours=3.0, type="accessibility").dict(),
        Activity(name="Food Tour", description="Taste local delicacies", cost=75.0, duration_hours=2.5, type="food").dict()
    ]

    filtered_activities = []
    for activity in all_activities:
        matches_interest = not interests or any(i.lower() in activity['type'].lower() for i in interests)
        matches_accessibility = not accessibility or any(a.lower() in activity['type'].lower() for a in accessibility)
        if matches_interest and matches_accessibility:
            filtered_activities.append(activity)

    st.session_state.status_messages.append(f"Found {len(filtered_activities)} activities.")
    return filtered_activities

@tool
def get_weather_forecast(city: str, date: str) -> str:
    """Fetches the weather forecast for a given city and date."""
    st.session_state.status_messages.append(f"Fetching weather for {city} on {date}...")
    time.sleep(0.5)
    weather_options = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
    forecast = random.choice(weather_options)
    st.session_state.status_messages.append(f"Weather in {city} on {date}: {forecast}.")
    return forecast

@tool
def update_flight_status(flight_number: str) -> str:
    """Simulates updating the status of a specific flight. Returns the new status."""
    st.session_state.status_messages.append(f"Checking status for flight {flight_number}...")
    time.sleep(0.5)
    statuses = ["On Time", "Delayed by 30 mins", "Cancelled"]
    new_status = random.choice(statuses)
    st.session_state.status_messages.append(f"Flight {flight_number} new status: {new_status}.")
    return new_status

# --- 3. Knowledge Base (ChromaDB Simulation) --- #

def initialize_knowledge_base():
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="travel_knowledge")
    
    # Add some dummy destination info
    if collection.count() == 0:
        collection.add(
            documents=[
                "Paris, France is known for its romantic atmosphere, the Eiffel Tower, and the Louvre Museum.",
                "Rome, Italy is famous for ancient ruins like the Colosseum and delicious pasta.",
                "Tokyo, Japan offers a blend of traditional culture and futuristic technology, with bustling markets and serene temples."
            ],
            metadatas=[
                {"city": "Paris", "country": "France"},
                {"city": "Rome", "country": "Italy"},
                {"city": "Tokyo", "country": "Japan"}
            ],
            ids=["doc1", "doc2", "doc3"]
        )
    return collection

# --- 4. LLM Agent Orchestrator (LangChain) --- #

def setup_llm_agent(openai_api_key: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
    tools = [
        search_flights,
        search_hotels,
        get_activities,
        get_weather_forecast,
        update_flight_status
    ]

    # Define the prompt for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an intelligent travel planner AI. Your goal is to create a detailed, optimized travel itinerary "
            "based on user input and constraints. You must decompose the task into sub-tasks (e.g., find flights, find hotels, "
            "plan activities) and use the provided tools to gather information. "
            "Always consider budget, interests, and accessibility. "
            "When re-planning, adapt the itinerary based on new real-time information. "
            "Be precise with dates and locations for tool calls. "
            "The user's current trip plan is stored in st.session_state.current_trip. Update it as you make decisions."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor

# --- 5. Real-time Event Monitor (Simulated) --- #

def simulate_realtime_event(trip: Trip) -> Optional[str]:
    """Simulates a real-time event that might trigger re-planning."""
    if random.random() < 0.3: # 30% chance of an event
        event_type = random.choice(["flight_delay", "bad_weather"])
        if event_type == "flight_delay" and trip.destinations:
            # Pick a random flight if available
            for dest in trip.destinations:
                # This is a simplification; in a real app, flights would be part of destinations
                # For now, let's just assume a flight associated with the first destination
                if dest.transportation and "Flight" in dest.transportation:
                    st.session_state.status_messages.append("SIMULATING: A flight delay occurred!")
                    return f"Flight associated with {dest.city} is delayed. Please re-evaluate the plan."
            return None # No flight to delay

        elif event_type == "bad_weather" and trip.destinations:
            target_city = random.choice(trip.destinations).city
            st.session_state.status_messages.append(f"SIMULATING: Bad weather detected in {target_city}!")
            return f"Bad weather predicted in {target_city}. Consider alternative indoor activities or changing plans."
    return None

# --- Streamlit UI --- #
st.set_page_config(layout="wide", page_title="Intelligent Travel Planner")
st.title("🌍 Intelligent Travel Planner")

# Initialize session state variables
if "current_trip" not in st.session_state:
    st.session_state.current_trip = None
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "status_messages" not in st.session_state:
    st.session_state.status_messages = []

# Sidebar for API Key and instructions
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key)
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        st.success("API Key set!")
    else:
        st.warning("Please enter your OpenAI API Key to proceed.")

    st.header("Instructions")
    st.markdown(
        "1. Enter your OpenAI API Key."
        "2. Provide your trip details and preferences."
        "3. Click 'Generate Plan' to get an initial itinerary."
        "4. Click 'Simulate Real-time Event & Re-plan' to see how the planner adapts."
    )

# Main content area
if not st.session_state.openai_api_key:
    st.info("Please enter your OpenAI API Key in the sidebar to start planning your trip.")
else:
    agent_executor = setup_llm_agent(st.session_state.openai_api_key)
    travel_kb_collection = initialize_knowledge_base()

    st.header("Plan Your Next Adventure")

    with st.form("trip_planning_form"):
        trip_name = st.text_input("Trip Name", "My Dream Vacation")
        destinations_input = st.text_area(
            "Destinations (e.g., Paris, France (2024-08-01 to 2024-08-05); Rome, Italy (2024-08-06 to 2024-08-10))"
        )
        budget = st.number_input("Total Budget ($)", min_value=100.0, value=2000.0, step=50.0)
        interests = st.multiselect("Interests", ["Sightseeing", "Culture", "Food", "Adventure", "Relaxation"], default=["Sightseeing", "Food"])
        accessibility = st.multiselect("Accessibility Needs", ["Wheelchair Accessible", "Hearing Impaired Friendly"], default=[]
        )
        submit_button = st.form_submit_button("Generate Plan")

        if submit_button:
            if not destinations_input:
                st.error("Please enter at least one destination.")
            else:
                st.session_state.status_messages = [] # Clear previous messages
                st.session_state.status_messages.append("Initiating plan generation...")
                
                # Parse destinations input
                parsed_destinations = []
                for dest_str in destinations_input.split(';'):
                    try:
                        parts = dest_str.strip().split('(')
                        city_country = parts[0].strip().replace(',', '') # Remove comma from city, country
                        dates_str = parts[1].replace(')', '').strip()
                        city_name, country_name = city_country.split(' ', 1) # Split city and country
                        start_date, end_date = dates_str.split(' to ')
                        parsed_destinations.append(
                            Destination(city=city_name, country=country_name, start_date=start_date, end_date=end_date)
                        )
                    except Exception as e:
                        st.error(f"Error parsing destination: {dest_str}. Format should be 'City, Country (YYYY-MM-DD to YYYY-MM-DD)'. Error: {e}")
                        parsed_destinations = []
                        break
                
                if parsed_destinations:
                    user_query = (
                        f"Plan a trip named '{trip_name}' for the following destinations: "
                        f"{'; '.join([f'{d.city}, {d.country} from {d.start_date} to {d.end_date}' for d in parsed_destinations])}. "
                        f"Total budget is ${budget}. User interests are {', '.join(interests)}. "
                        f"Accessibility needs: {', '.join(accessibility) if accessibility else 'None'}. "
                        "Please find flights, hotels, and activities for each destination. Optimize for budget and interests."
                    )
                    st.session_state.chat_history.append(("user", user_query))
                    
                    try:
                        with st.spinner("Generating initial trip plan..."):
                            # Store partial trip data in session state for agent to use
                            st.session_state.current_trip = Trip(
                                trip_name=trip_name,
                                destinations=parsed_destinations,
                                total_budget=budget,
                                user_interests=interests,
                                accessibility_needs=accessibility
                            )
                            
                            response = agent_executor.invoke({
                                "input": user_query,
                                "chat_history": st.session_state.chat_history
                            })
                            
                            # The agent's output is often a string. We need to parse it or have the agent structure it.
                            # For simplicity, let's assume the agent provides a coherent plan as text.
                            # A more robust solution would have the agent output a Pydantic object.
                            st.session_state.current_trip.status_messages.append(response["output"])
                            st.session_state.status_messages.append("Initial plan generated!")
                            st.session_state.chat_history.append(("ai", response["output"]))
                            st.toast("Trip plan generated successfully!")

                    except Exception as e:
                        st.error(f"An error occurred during plan generation: {e}")
                        st.session_state.status_messages.append(f"Error: {e}")

    if st.session_state.current_trip:
        st.subheader(f"Planned Trip: {st.session_state.current_trip.trip_name}")
        st.write(f"**Total Budget:** ${st.session_state.current_trip.total_budget:.2f}")
        st.write(f"**Current Estimated Cost:** ${st.session_state.current_trip.current_cost:.2f}")
        st.write(f"**Interests:** {', '.join(st.session_state.current_trip.user_interests)}")
        st.write(f"**Accessibility:** {', '.join(st.session_state.current_trip.accessibility_needs) if st.session_state.current_trip.accessibility_needs else 'None'}")

        st.markdown("### Itinerary Details")
        for i, dest in enumerate(st.session_state.current_trip.destinations):
            st.markdown(f"#### {dest.city}, {dest.country} ({dest.start_date} to {dest.end_date})")
            if dest.hotel:
                st.write(f"- **Hotel:** {dest.hotel.name} (Rating: {dest.hotel.rating}, ${dest.hotel.price_per_night}/night)")
            if dest.transportation:
                st.write(f"- **Transportation:** {dest.transportation}")
            if dest.activities:
                st.write("- **Activities:**")
                for activity in dest.activities:
                    st.write(f"  - {activity.name}: {activity.description} (Cost: ${activity.cost:.2f}, Duration: {activity.duration_hours}h)")
            else:
                st.write("- No detailed activities planned yet for this destination.")

        st.markdown("### Real-time Status and Actions")
        for msg in st.session_state.status_messages:
            st.info(msg)

        if st.button("Simulate Real-time Event & Re-plan"):
            event_message = simulate_realtime_event(st.session_state.current_trip)
            if event_message:
                st.session_state.status_messages.append(event_message)
                st.session_state.status_messages.append("Triggering re-planning...")
                replan_query = f"An event occurred: {event_message}. Please re-evaluate the current trip plan and suggest adjustments. "
                replan_query += f"Current trip details: {st.session_state.current_trip.json()}"
                
                st.session_state.chat_history.append(("user", replan_query))
                try:
                    with st.spinner("Re-planning due to real-time event..."):
                        response = agent_executor.invoke({
                            "input": replan_query,
                            "chat_history": st.session_state.chat_history
                        })
                        st.session_state.current_trip.status_messages.append(response["output"])
                        st.session_state.status_messages.append("Re-planning complete!")
                        st.session_state.chat_history.append(("ai", response["output"]))
                        st.toast("Trip re-planned successfully!")
                except Exception as e:
                    st.error(f"An error occurred during re-planning: {e}")
                    st.session_state.status_messages.append(f"Error during re-planning: {e}")
            else:
                st.session_state.status_messages.append("No significant real-time event simulated this time.")
                st.info("No event to re-plan for this time.")


