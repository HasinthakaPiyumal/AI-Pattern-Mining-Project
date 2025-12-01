import streamlit as st
import requests
import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Replace with your actual weather API key if using a real one
MOCK_WEATHER_API_KEY = "YOUR_MOCK_WEATHER_API_KEY"

# --- Pydantic Models for Tool Input Validation ---

class FlightSearchInput(BaseModel):
    departure: str = Field(description="The departure city.")
    destination: str = Field(description="The destination city.")
    start_date: str = Field(description="The start date of the trip (YYYY-MM-DD).")
    end_date: str = Field(description="The end date of the trip (YYYY-MM-DD).")
    preferences: str = Field(description="Any specific flight preferences like airline, class, etc.", default="any")

class AccommodationSearchInput(BaseModel):
    location: str = Field(description="The city or area for accommodation.")
    start_date: str = Field(description="The check-in date (YYYY-MM-DD).")
    end_date: str = Field(description="The check-out date (YYYY-MM-DD).")
    preferences: str = Field(description="Any specific accommodation preferences like hotel type, budget, amenities, etc.", default="any")

class WeatherInfoInput(BaseModel):
    location: str = Field(description="The city to get weather information for.")
    date: str = Field(description="The date for the weather forecast (YYYY-MM-DD).")

class ItineraryGenerationInput(BaseModel):
    flights: str = Field(description="Details of selected flights.")
    accommodations: str = Field(description="Details of selected accommodations.")
    points_of_interest: str = Field(description="Comma-separated list of points of interest or activities.")
    travel_dates: str = Field(description="The overall travel dates (e.g., '2024-08-01 to 2024-08-07').")
    preferences: str = Field(description="User's general travel preferences.")

# --- Mock External APIs ---

def mock_flight_search(departure, destination, start_date, end_date, preferences):
    print(f"Searching flights: {departure} to {destination}, {start_date} to {end_date}, preferences: {preferences}")
    # Simulate API call
    if "luxury" in preferences.lower():
        return f"Found luxury flights from {departure} to {destination} on {start_date} for $1500 (e.g., Business Class on Emirates)."
    return f"Found economy flights from {departure} to {destination} on {start_date} for $300 (e.g., United Airlines)."

def mock_accommodation_search(location, start_date, end_date, preferences):
    print(f"Searching accommodations: {location}, {start_date} to {end_date}, preferences: {preferences}")
    # Simulate API call
    if "budget" in preferences.lower():
        return f"Found budget hotel in {location} for $80/night (e.g., Hostel World)."
    return f"Found mid-range hotel in {location} for $150/night (e.g., Hilton Garden Inn)."

def mock_weather_info(location, date):
    print(f"Getting weather for: {location} on {date}")
    # Simulate API call (could integrate with a real weather API if desired)
    if MOCK_WEATHER_API_KEY == "YOUR_MOCK_WEATHER_API_KEY":
        return f"Weather in {location} on {date}: Sunny with a high of 25°C. (Mocked response)"
    # Example of a real API call (uncomment and configure if you have a real API key)
    # try:
    #     response = requests.get(f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={MOCK_WEATHER_API_KEY}")
    #     data = response.json()
    #     # Parse data to get relevant forecast for the date
    #     return f"Actual weather for {location} on {date}: {data['list'][0]['weather'][0]['description']}"
    # except Exception as e:
    #     return f"Could not retrieve real weather info: {e}. Returning mock."
    return f"Weather in {location} on {date}: Sunny with a high of 25°C. (Mocked response)"

def mock_itinerary_generation(flights, accommodations, points_of_interest, travel_dates, preferences):
    print(f"Generating itinerary with: flights={flights}, hotels={accommodations}, POIs={points_of_interest}")
    itinerary = f"-- Personalized Travel Itinerary --\n"
    itinerary += f"Travel Dates: {travel_dates}\n"
    itinerary += f"Preferences: {preferences}\n\n"
    itinerary += f"Flights:\n- {flights}\n\n"
    itinerary += f"Accommodations:\n- {accommodations}\n\n"
    if points_of_interest:
        itinerary += f"Activities/Points of Interest:\n- {points_of_interest.replace(',', '\n- ')}\n\n"
    itinerary += "Enjoy your trip!\n"
    return itinerary

# --- LangChain Tools ---

tools = [
    Tool(
        name="FlightSearch",
        func=lambda **kwargs: mock_flight_search(**kwargs),
        description="Useful for searching for flights based on departure, destination, dates, and preferences.",
        args_schema=FlightSearchInput,
    ),
    Tool(
        name="AccommodationSearch",
        func=lambda **kwargs: mock_accommodation_search(**kwargs),
        description="Useful for searching for accommodations based on location, dates, and preferences.",
        args_schema=AccommodationSearchInput,
    ),
    Tool(
        name="WeatherInfo",
        func=lambda **kwargs: mock_weather_info(**kwargs),
        description="Useful for getting weather information for a specific location and date.",
        args_schema=WeatherInfoInput,
    ),
    Tool(
        name="ItineraryGeneration",
        func=lambda **kwargs: mock_itinerary_generation(**kwargs),
        description="Useful for compiling flight, accommodation, and activity details into a structured travel itinerary.",
        args_schema=ItineraryGenerationInput,
    ),
]

# --- Memory Module ---

# Conversational Memory
conversational_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Long-term User Preference Memory (using in-memory Chroma for simplicity)
embeddings_model = OpenAIEmbeddings() if OPENAI_API_KEY else SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# In-memory ChromaDB for user preferences
preference_vectorstore = Chroma(embedding_function=embeddings_model, persist_directory=None)

def store_user_preference(user_id: str, preference: str):
    content = f"User {user_id} preference: {preference}"
    preference_vectorstore.add_texts([content], metadatas=[{"user_id": user_id, "type": "preference"}])

def get_user_preferences(user_id: str, k: int = 2):
    # Simulate retrieval of relevant preferences
    results = preference_vectorstore.similarity_search(f"User {user_id} previous travel preferences", k=k)
    preferences = [doc.page_content.replace(f"User {user_id} preference: ", "") for doc in results if doc.metadata.get("user_id") == user_id]
    return "; ".join(preferences) if preferences else "No specific preferences found."

# --- LLM and Agent Setup ---

llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

# Prompt for the LangChain agent
prompt = PromptTemplate.from_template("""You are a helpful AI Travel Planner. Your goal is to assist users in planning their trips by finding flights, accommodations, weather information, and creating detailed itineraries. 

Before making any suggestions, always try to understand the user's preferences. Store important user preferences for future reference. If you need more information to fulfill a request, ask clarifying questions.

Today's date is 2024-07-26.

TOOLS:
{tools}

FORMAT INSTRUCTIONS:
{format_instructions}

USER_PREFERENCES: {user_preferences}
CHAT HISTORY:
{chat_history}
Question: {input}
{agent_scratchpad}""")

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=conversational_memory, handle_parsing_errors=True)

# --- Streamlit UI --- 
st.set_page_config(page_title="AI Travel Planner", layout="centered")
st.title("✈️ AI Travel Planner")
st.markdown("I can help you plan your next trip! Tell me your travel plans and preferences.")

# Session state for chat history and user ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_123" # A simple fixed user ID for demonstration

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def handle_user_input(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        # Retrieve user-specific preferences
        current_user_preferences = get_user_preferences(st.session_state.user_id)
        
        # Add user preferences to the prompt for the current turn
        response = agent_executor.invoke({
            "input": user_input,
            "user_preferences": current_user_preferences # Inject preferences here
        })
        
        agent_response = response["output"]
        st.session_state.messages.append({"role": "assistant", "content": agent_response})
        with st.chat_message("assistant"):
            st.markdown(agent_response)
        
        # A simple way to try and capture/store preferences from the conversation
        # In a real app, this would be more sophisticated (e.g., entity extraction)
        if "my preference is" in user_input.lower() or "i prefer" in user_input.lower():
            preference_to_store = user_input.split("is", 1)[-1].strip() if "is" in user_input.lower() else user_input.split("prefer", 1)[-1].strip()
            if preference_to_store:
                store_user_preference(st.session_state.user_id, preference_to_store)
                st.info(f"Stored a new preference for {st.session_state.user_id}: {preference_to_store}")

# Chat input
if prompt := st.chat_input("What are your travel plans?"):
    handle_user_input(prompt)

st.sidebar.header("Travel Planner Settings")
st.sidebar.write(f"Current User ID: {st.session_state.user_id}")
st.sidebar.markdown("--- This is a demo. Mock APIs are used. --- ")

# Optional: Display current stored preferences (for debugging)
with st.sidebar.expander("Stored Preferences (Debug)"):
    st.write(get_user_preferences(st.session_state.user_id, k=5))
