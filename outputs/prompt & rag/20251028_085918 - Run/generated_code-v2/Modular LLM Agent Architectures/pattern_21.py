from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --- LLM Integration ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# --- Custom In-Memory User Preferences Store ---
user_preferences = {}

# --- Tool Use Module (Simulated Functions) ---
def search_flights(origin: str, destination: str, date: str) -> str:
    return f"Simulated flight search from {origin} to {destination} on {date}: Flight AF123, price $500."

def book_hotel(destination: str, check_in: str, check_out: str, guests: int) -> str:
    return f"Simulated hotel booking in {destination} from {check_in} to {check_out} for {guests} guests: Hotel ABC, confirmation #XYZ."

def get_weather(location: str, date: str) -> str:
    return f"Simulated weather forecast for {location} on {date}: Sunny, 25 C."

def find_attractions(location: str, type: str = "any") -> str:
    return f"Simulated attractions in {location} ({type}): Eiffel Tower, Louvre Museum."

# --- Wrap functions as LangChain Tools ---
tools = [
    Tool(
        name="SearchFlights",
        func=search_flights,
        description="Useful for searching for flights between an origin and destination on a specific date."
    ),
    Tool(
        name="BookHotel",
        func=book_hotel,
        description="Useful for booking a hotel in a destination for specific dates and number of guests."
    ),
    Tool(
        name="GetWeather",
        func=get_weather,
        description="Useful for getting the weather forecast for a location on a specific date."
    ),
    Tool(
        name="FindAttractions",
        func=find_attractions,
        description="Useful for finding attractions in a given location, optionally filtered by type."
    ),
]

# --- Memory Module ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Planning Module (Agent Definition) ---
# The prompt template should guide the agent to use tools and memory effectively
# A simple ReAct style prompt can be used.
# The actual planning logic is embedded in the LLM's reasoning with this prompt.

prompt_template = PromptTemplate.from_template(
    """You are a helpful and autonomous travel planning agent. You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

User preferences: {user_preferences}
Chat history: {chat_history}
Question: {input}
Thought:{agent_scratchpad}"""
)

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt_template)

# Create the Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True # Robustness for initial development
)

def run_travel_agent():
    print("Hello! I'm your Smart Travel Planner Agent. How can I help you plan your trip today?")
    while True:
        user_input = input("\nYour request: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! Happy travels!")
            break
        try:
            # Pass user_preferences to the agent_executor dynamically
            response = agent_executor.invoke({
                "input": user_input,
                "user_preferences": user_preferences # Injecting preferences
            })
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again or refine your request.")

if __name__ == "__main__":
    # Example of setting user preferences (can be done via a separate function/UI interaction)
    user_preferences["budget"] = "moderate"
    user_preferences["travel_style"] = "adventure"
    user_preferences["preferred_airline"] = "Any"
    print(f"Initial user preferences set: {user_preferences}")
    run_travel_agent()
