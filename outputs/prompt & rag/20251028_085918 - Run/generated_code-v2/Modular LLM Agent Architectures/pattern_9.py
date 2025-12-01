from dotenv import load_dotenv
import os
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime

load_dotenv()

# Tool Use Module
@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    print(f"Searching flights from {origin} to {destination} on {date}")
    if destination.lower() == "paris" and date == "2024-06-15":
        return "Flight AF123 from New York to Paris on 2024-06-15 at 10:00 AM, cost $800."
    return "No direct flights found for the given criteria."

@tool
def book_hotel(location: str, check_in: str, check_out: str, guests: int) -> str:
    print(f"Booking hotel in {location} from {check_in} to {check_out} for {guests} guests")
    if location.lower() == "paris" and guests == 2 and check_in == "2024-06-15":
        return "Hotel Le Grand Paris booked from 2024-06-15 to 2024-06-20 for 2 guests. Confirmation ID: HGP789."
    return "Could not book a hotel with the provided details."

@tool
def find_activities(location: str, date: str) -> str:
    print(f"Finding activities in {location} on {date}")
    if location.lower() == "paris" and date == "2024-06-16":
        return "Activities in Paris on 2024-06-16: Eiffel Tower visit, Louvre Museum tour, Seine River cruise."
    return "No specific activities found for the given location and date."

@tool
def get_weather(location: str, date: str) -> str:
    print(f"Getting weather for {location} on {date}")
    if location.lower() == "paris" and date == "2024-06-15":
        return "Weather in Paris on 2024-06-15: Sunny with a high of 25°C."
    return "Weather information not available."

tools = [search_flights, book_hotel, find_activities, get_weather]

# Memory Module
def get_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Main Agent

# LLM
llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI-powered travel agent. Your goal is to help users plan personalized trips by utilizing various tools. Remember to ask clarifying questions and confirm details before making any bookings."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=get_memory())

def run_travel_agent():
    print("Welcome to the AI Travel Agent! How can I help you plan your trip?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again or refine your request.")

if __name__ == "__main__":
    run_travel_agent()
