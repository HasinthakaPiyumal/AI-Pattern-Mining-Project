import streamlit as st
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

# Set OpenAI API key from environment variable
# Ensure you have OPENAI_API_KEY set in your environment

# 1. Pydantic Models for Structured Output
class Activity(BaseModel):
    name: str = Field(description="Name of the activity or attraction")
    time: str = Field(description="Suggested time for the activity (e.g., 'Morning', '2 PM - 4 PM')")
    description: Optional[str] = Field(None, description="Brief description of the activity")
    address: Optional[str] = Field(None, description="Address of the activity")
    estimated_duration: Optional[str] = Field(None, description="Estimated duration of the activity (e.g., '2 hours')")

class Meal(BaseModel):
    type: str = Field(description="Type of meal (e.g., 'Breakfast', 'Lunch', 'Dinner')")
    suggestion: str = Field(description="Meal suggestion or restaurant name")
    cuisine: Optional[str] = Field(None, description="Type of cuisine")
    address: Optional[str] = Field(None, description="Address of the restaurant")

class Accommodation(BaseModel):
    name: str = Field(description="Name of the hotel or accommodation")
    type: str = Field(description="Type of accommodation (e.g., 'Hotel', 'Airbnb', 'Boutique Hotel')")
    location: str = Field(description="General location or neighborhood")
    check_in_date: Optional[str] = Field(None, description="Check-in date (YYYY-MM-DD)")
    check_out_date: Optional[str] = Field(None, description="Check-out date (YYYY-MM-DD)")

class Transportation(BaseModel):
    method: str = Field(description="Method of transportation (e.g., 'Taxi', 'Metro', 'Walk', 'Bus')")
    details: Optional[str] = Field(None, description="Specific details about transportation")

class DayItinerary(BaseModel):
    day: int = Field(description="Day number of the trip")
    theme: Optional[str] = Field(None, description="Optional theme for the day (e.g., 'Historical Exploration')")
    activities: List[Activity] = Field(description="List of activities for the day")
    meals: List[Meal] = Field(description="List of meal suggestions for the day")
    transportation_tips: Optional[List[Transportation]] = Field(None, description="Optional transportation tips for the day")

class TravelPlan(BaseModel):
    destination: str = Field(description="The travel destination")
    duration_days: int = Field(description="Total number of days for the trip")
    travelers: str = Field(description="Number and type of travelers (e.g., 'family of four', 'solo traveler')")
    budget_level: str = Field(description="Budget level (e.g., 'moderate', 'luxury', 'budget-friendly')")
    overall_theme: Optional[str] = Field(None, description="Overall theme of the trip")
    accommodation: Optional[Accommodation] = Field(None, description="Suggested accommodation for the trip")
    itinerary: List[DayItinerary] = Field(description="Detailed daily itinerary")

# 2. LangChain Setup
llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo-preview")

# Prompt for initial natural language itinerary generation
nl_itinerary_prompt = PromptTemplate(
    template="""You are an AI travel agent. Generate a detailed {duration_days}-day travel itinerary for {travelers} in {destination}, focusing on {preferences} with a {budget_level} budget. 
Include daily activities, suggested meals, transportation tips, and accommodation type. Present this as a natural language narrative.
Ensure the itinerary is comprehensive and engaging. Start with a general overview and then detail each day.
""",
    input_variables=["duration_days", "travelers", "destination", "preferences", "budget_level"],
)

nl_itinerary_chain = LLMChain(llm=llm, prompt=nl_itinerary_prompt)

# Prompt for structured output generation using PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=TravelPlan)

structured_output_prompt = PromptTemplate(
    template="""Given the following natural language travel itinerary, extract the key components and structure them into a JSON object adhering to the following schema:

{format_instructions}

Natural Language Itinerary:
{itinerary_text}

Structured JSON Output:""",
    input_variables=["itinerary_text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

structured_output_chain = LLMChain(llm=llm, prompt=structured_output_prompt)

# Streamlit UI
st.set_page_config(layout="wide", page_title="AI Travel Planner")
st.title("🌍 AI Travel Planner")
st.markdown("Enter your travel preferences below and let AI generate a structured itinerary for you!")

with st.sidebar:
    st.header("Travel Preferences")
    destination = st.text_input("Destination", "Paris")
    duration_days = st.number_input("Duration (days)", min_value=1, max_value=30, value=5)
    travelers = st.text_input("Travelers (e.g., 'family of four', 'solo traveler')", "family of four")
    preferences = st.text_area("Key Interests/Preferences (e.g., 'historical sites, good food, art museums')", "historical sites, good food, art museums")
    budget_level = st.selectbox("Budget Level", ["budget-friendly", "moderate", "luxury"], index=1)

    generate_button = st.button("Generate Itinerary")

st.subheader("Generated Itinerary")

if generate_button:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY environment variable not set. Please set it to use the application.")
    else:
        with st.spinner("Generating natural language itinerary..."):
            try:
                # Generate natural language itinerary
                nl_itinerary_response = nl_itinerary_chain.run(
                    duration_days=duration_days,
                    travelers=travelers,
                    destination=destination,
                    preferences=preferences,
                    budget_level=budget_level
                )
                st.markdown("### Natural Language Itinerary")
                st.write(nl_itinerary_response)

                with st.spinner("Structuring the itinerary..."):
                    # Parse natural language to structured JSON
                    structured_output = structured_output_chain.run(itinerary_text=nl_itinerary_response)

                    # Attempt to parse the structured output string into the Pydantic model
                    try:
                        final_plan = parser.parse(structured_output)
                        st.markdown("### Structured Travel Plan (JSON)")
                        st.json(final_plan.model_dump())
                    except Exception as e:
                        st.error(f"Failed to parse structured output: {e}")
                        st.text_area("Raw Structured Output Attempt:", structured_output, height=300)

            except Exception as e:
                st.error(f"An error occurred during itinerary generation: {e}")
                st.info("Please ensure your OpenAI API key is valid and you have sufficient credits.")
else:
    st.info("Enter your travel details in the sidebar and click 'Generate Itinerary'.")
