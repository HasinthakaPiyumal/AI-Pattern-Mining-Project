import os
from dotenv import load_dotenv
from typing import Type

import streamlit as st
from pydantic import BaseModel, Field

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# --- 1. Pydantic Models for Tools ---

class FlightSearchInput(BaseModel):
    departure: str = Field(description="departure city")
    destination: str = Field(description="destination city")
    start_date: str = Field(description="start date of travel (YYYY-MM-DD)")
    end_date: str = Field(description="end date of travel (YYYY-MM-DD)")
    travelers: int = Field(description="number of travelers")

class HotelSearchInput(BaseModel):
    location: str = Field(description="city or region for the hotel search")
    check_in_date: str = Field(description="check-in date (YYYY-MM-DD)")
    check_out_date: str = Field(description="check-out date (YYYY-MM-DD)")
    guests: int = Field(description="number of guests")

class ActivitySearchInput(BaseModel):
    location: str = Field(description="city or region for activity search")
    date: str = Field(description="date for activity (YYYY-MM-DD)")
    interests: str = Field(description="comma-separated interests (e.g., museums, hiking)")

class WeatherCheckInput(BaseModel):
    location: str = Field(description="city or region to check weather")
    date: str = Field(description="date for weather forecast (YYYY-MM-DD)")

class BudgetEstimatorInput(BaseModel):
    destination: str = Field(description="travel destination")
    duration_days: int = Field(description="duration of stay in days")
    travelers: int = Field(description="number of travelers")
    preferences: str = Field(description="travel preferences (e.g., luxury, budget, mid-range)")

class VisaRequirementInput(BaseModel):
    nationality: str = Field(description="traveler's nationality")
    destination_country: str = Field(description="destination country")

# --- 2. Simulated External Tools ---

class FlightSearchTool(BaseTool):
    name = "flight_search"
    description = "Searches for flights based on departure, destination, dates, and number of travelers."
    args_schema: Type[BaseModel] = FlightSearchInput

    def _run(self, departure: str, destination: str, start_date: str, end_date: str, travelers: int) -> str:
        return f"Mock flight data for {travelers} people from {departure} to {destination} between {start_date} and {end_date}: Flight 123, Airline X, ${300*travelers} per person. Total: ${300*travelers*travelers}."

    async def _arun(self, departure: str, destination: str, start_date: str, end_date: str, travelers: int) -> str:
        raise NotImplementedError("asynchronous run not implemented")

class HotelSearchTool(BaseTool):
    name = "hotel_search"
    description = "Searches for hotels based on location, check-in, check-out dates, and number of guests."
    args_schema: Type[BaseModel] = HotelSearchInput

    def _run(self, location: str, check_in_date: str, check_out_date: str, guests: int) -> str:
        return f"Mock hotel data for {guests} guests in {location} from {check_in_date} to {check_out_date}: Hotel A, 4-star, ${150*guests} per night. Total for 3 nights: ${450*guests}."

    async def _arun(self, location: str, check_in_date: str, check_out_date: str, guests: int) -> str:
        raise NotImplementedError("asynchronous run not implemented")

class ActivitySearchTool(BaseTool):
    name = "activity_search"
    description = "Searches for activities in a given location for a specific date and interests."
    args_schema: Type[BaseModel] = ActivitySearchInput

    def _run(self, location: str, date: str, interests: str) -> str:
        return f"Mock activities in {location} on {date} with interests {interests}: City Tour, Museum Visit, Local Food Tasting."

    async def _arun(self, location: str, date: str, interests: str) -> str:
        raise NotImplementedError("asynchronous run not implemented")

class WeatherCheckTool(BaseTool):
    name = "weather_check"
    description = "Checks the weather forecast for a location on a specific date."
    args_schema: Type[BaseModel] = WeatherCheckInput

    def _run(self, location: str, date: str) -> str:
        return f"Mock weather forecast for {location} on {date}: Sunny, 25°C, light breeze."

    async def _arun(self, location: str, date: str) -> str:
        raise NotImplementedError("asynchronous run not implemented")

class BudgetEstimatorTool(BaseTool):
    name = "budget_estimator"
    description = "Estimates the budget for a trip based on destination, duration, travelers, and preferences."
    args_schema: Type[BaseModel] = BudgetEstimatorInput

    def _run(self, destination: str, duration_days: int, travelers: int, preferences: str) -> str:
        base_cost = 200 * duration_days * travelers
        if "luxury" in preferences.lower():
            estimated_budget = base_cost * 2
        elif "budget" in preferences.lower():
            estimated_budget = base_cost * 0.7
        else:
            estimated_budget = base_cost
        return f"Mock budget estimate for a {duration_days}-day trip to {destination} for {travelers} people with {preferences} preferences: ${estimated_budget}."

    async def _arun(self, destination: str, duration_days: int, travelers: int, preferences: str) -> str:
        raise NotImplementedError("asynchronous run not implemented")

class VisaRequirementTool(BaseTool):
    name = "visa_requirement_check"
    description = "Checks visa requirements for a given nationality traveling to a destination country."
    args_schema: Type[BaseModel] = VisaRequirementInput

    def _run(self, nationality: str, destination_country: str) -> str:
        if destination_country.lower() == "japan" and nationality.lower() == "usa":
            return "Mock visa requirement: US citizens do not require a visa for short stays in Japan."
        return f"Mock visa requirement for {nationality} to {destination_country}: Visa might be required. Please check official embassy website."

    async def _arun(self, nationality: str, destination_country: str) -> str:
        raise NotImplementedError("asynchronous run not implemented")

# List of all tools
tools = [
    FlightSearchTool(),
    HotelSearchTool(),
    ActivitySearchTool(),
    WeatherCheckTool(),
    BudgetEstimatorTool(),
    VisaRequirementTool()
]

# --- 3. LangChain Agent Setup ---
def create_travel_planner_agent():
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert travel planner AI. Your goal is to create a comprehensive and personalized travel itinerary based on user requests. Break down complex requests into logical steps and use the provided tools to gather information."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# --- 4. Streamlit UI ---
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️")
st.title("✈️ AI-Powered Personalized Travel Planner")

if "agent" not in st.session_state:
    st.session_state.agent = create_travel_planner_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat():
    st.session_state.messages = []
    st.session_state.agent = create_travel_planner_agent() # Reset agent if needed

st.sidebar.button("Clear Chat", on_click=clear_chat)

user_input = st.chat_input("Tell me about your travel plans...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Planning your trip..."):
        try:
            # LangChain Agent expects history as a list of dicts with 'role' and 'content'
            # For simple input, we can just pass the latest message and clear chat history for agent's own context
            # For a more advanced conversational agent, we would map the full st.session_state.messages to LangChain's ChatMessage format
            
            # For this example, we pass only the last human input and let the agent handle internal chat history
            # through `MessagesPlaceholder(variable_name="chat_history")` in the prompt, but for initial query
            # it's just the 'input' from user. More robust chat history passing for actual multi-turn conversations
            # would involve converting st.session_state.messages into LangChain's message objects and passing it
            # to `chat_history` variable in agent_executor.invoke().

            # Simple approach for demonstration:
            response = st.session_state.agent.invoke({"input": user_input, "chat_history": []})
            agent_response = response["output"]
        except Exception as e:
            agent_response = f"An error occurred during planning: {e}. Please try again."
            st.error(agent_response)

    st.session_state.messages.append({"role": "assistant", "content": agent_response})
    with st.chat_message("assistant"):
        st.markdown(agent_response)
