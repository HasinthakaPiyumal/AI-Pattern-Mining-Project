from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
import os
import json

class Activity(BaseModel):
    name: str = Field(description="Name of the activity")
    description: str = Field(description="Short description of the activity")
    date: str = Field(description="Date of the activity in YYYY-MM-DD format")

class Accommodation(BaseModel):
    name: str = Field(description="Name of the accommodation (e.g., hotel, Airbnb)")
    type: str = Field(description="Type of accommodation (e.g., Hotel, Airbnb, Hostel)")
    check_in_date: str = Field(description="Check-in date in YYYY-MM-DD format")
    check_out_date: str = Field(description="Check-out date in YYYY-MM-DD format")

class Transportation(BaseModel):
    type: str = Field(description="Type of transportation (e.g., Flight, Train, Car Rental)")
    details: str = Field(description="Details of the transportation (e.g., flight number, car model, rental company)")
    travel_date: str = Field(description="Date of travel in YYYY-MM-DD format")

class Itinerary(BaseModel):
    destination: str = Field(description="The travel destination")
    start_date: str = Field(description="The start date of the trip in YYYY-MM-DD format")
    end_date: str = Field(description="The end date of the trip in YYYY-MM-DD format")
    activities: List[Activity] = Field(description="A list of planned activities")
    accommodation: Optional[Accommodation] = Field(None, description="Accommodation details for the trip")
    transportation: List[Transportation] = Field(description="A list of transportation segments for the trip")

def generate_travel_itinerary(user_request: str) -> str:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    parser = PydanticOutputParser(pydantic_object=Itinerary)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    itinerary_object = chain.invoke({"query": user_request})
    return json.dumps(itinerary_object.dict(), indent=2)

# Example Usage:
if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    user_input = "Plan a 3-day trip to Paris from July 1st to July 3rd, 2024. I want to see the Eiffel Tower, visit the Louvre, and have a nice dinner. I'll need a flight and a hotel."
    structured_plan = generate_travel_itinerary(user_input)
    print(structured_plan)