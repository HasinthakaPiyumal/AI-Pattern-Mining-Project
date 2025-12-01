import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# --- Dummy Tool Implementations ---
def search_flights(origin: str, destination: str, departure_date: str, return_date: str = None) -> str:
    """Searches for flight options between an origin and destination on specified dates."""
    print(f"Searching flights from {origin} to {destination} on {departure_date}{f' returning on {return_date}' if return_date else ''}...")
    # In a real scenario, this would call a flight API
    if origin == "New York" and destination == "London" and departure_date == "2024-08-15":
        return "Found flight BA178 from New York to London on 2024-08-15, departing at 10:00 AM, arriving at 8:00 PM. Price: $750."
    return "No direct flights found for the specified criteria. Consider alternative dates or routes."

def book_hotel(location: str, check_in_date: str, check_out_date: str, preferences: str = None) -> str:
    """Books a hotel in the specified location for the given dates with optional preferences."""
    print(f"Booking hotel in {location} from {check_in_date} to {check_out_date} with preferences: {preferences}...")
    # In a real scenario, this would call a hotel booking API
    if location == "London" and check_in_date == "2024-08-15" and check_out_date == "2024-08-20":
        return "Hotel 'The Grand London' booked from 2024-08-15 to 2024-08-20. Confirmation ID: GL12345. Price: $200/night."
    return "Could not find or book a hotel matching your criteria."

def suggest_activities(location: str, interests: str = None) -> str:
    """Suggests activities and attractions in a given location based on interests."""
    print(f"Suggesting activities in {location} with interests: {interests}...")
    # In a real scenario, this would call an activity API or a local guide API
    if location == "London":
        if "history" in interests.lower():
            return "Activities in London: Visit the Tower of London, British Museum, Westminster Abbey."
        return "Activities in London: Explore Hyde Park, visit the London Eye, enjoy a show in West End."
    return "No specific activity suggestions for this location yet."

def get_weather(location: str, date: str) -> str:
    """Fetches the weather forecast for a specified location and date."""
    print(f"Getting weather for {location} on {date}...")
    # In a real scenario, this would call a weather API
    if location == "London" and date == "2024-08-15":
        return "Weather in London on 2024-08-15: Partly cloudy, high of 22°C, low of 14°C."
    return "Weather data not available for this location and date."

# --- Create Tools ---
tools = [
    Tool(
        name="Flight Search",
        func=lambda origin, destination, departure_date, return_date=None: search_flights(origin, destination, departure_date, return_date),
        description="Useful for searching for flight availability and prices. Input should be a comma-separated string of origin, destination, departure_date, and optionally return_date."
    ),
    Tool(
        name="Hotel Booking",
        func=lambda location, check_in_date, check_out_date, preferences=None: book_hotel(location, check_in_date, check_out_date, preferences),
        description="Useful for booking hotels. Input should be a comma-separated string of location, check_in_date, check_out_date, and optionally preferences."
    ),
    Tool(
        name="Activity Suggestion",
        func=lambda location, interests=None: suggest_activities(location, interests),
        description="Useful for getting suggestions for activities and attractions. Input should be a comma-separated string of location and optionally interests."
    ),
    Tool(
        name="Weather Forecast",
        func=lambda location, date: get_weather(location, date),
        description="Useful for getting the weather forecast. Input should be a comma-separated string of location and date."
    ),
]

# --- LLM Core ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Memory Module ---
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

# --- Agent Prompt ---
# The prompt should instruct the agent to use tools and manage conversation history.
# It's crucial for the agent to understand how to interact with the user and the tools.
prompt_template = """You are an intelligent travel planner agent. Your goal is to help users plan their trips by suggesting flights, hotels, and activities.
You have access to the following tools: {tools}

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

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""

# Create a PromptTemplate instance from the string
prompt = PromptTemplate.from_template(prompt_template)

# --- Planning Module (Agent Executor) ---
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15,
)

# --- Main Interaction Loop ---
if __name__ == "__main__":
    print("Hello! I am your Intelligent Travel Planner Agent. How can I help you plan your trip today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Agent: Goodbye! Happy travels!")
            break
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"Agent: An error occurred: {e}")
            print("Agent: Please try rephrasing your request.")
