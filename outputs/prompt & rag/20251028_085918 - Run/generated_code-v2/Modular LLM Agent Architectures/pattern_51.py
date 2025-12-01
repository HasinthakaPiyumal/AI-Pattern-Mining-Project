from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType

import os

# Set your OpenAI API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Placeholder functions for external API calls
def search_flights(origin: str, destination: str, date: str, preferences: str) -> str:
    return f"Searching flights from {origin} to {destination} on {date} with preferences: {preferences}... Found a direct flight for $300."

def search_accommodations(location: str, check_in_date: str, check_out_date: str, preferences: str) -> str:
    return f"Searching accommodations in {location} from {check_in_date} to {check_out_date} with preferences: {preferences}... Found a 4-star hotel for $150/night."

def search_activities(location: str, date: str, interests: str) -> str:
    return f"Searching activities in {location} on {date} for interests: {interests}... Found a city tour and a museum visit."

def create_itinerary(plan_details: str) -> str:
    return f"Creating itinerary with details: {plan_details}... Your personalized itinerary has been generated."

# Define the tools the agent can use
tools = [
    Tool(
        name="Search Flights",
        func=search_flights,
        description="useful for finding flight options between two locations on a specific date with user preferences."
    ),
    Tool(
        name="Search Accommodations",
        func=search_accommodations,
        description="useful for finding hotel or accommodation options in a location for specific dates with user preferences."
    ),
    Tool(
        name="Search Activities",
        func=search_activities,
        description="useful for finding local attractions and activities in a location on a specific date based on interests."
    ),
    Tool(
        name="Create Itinerary",
        func=create_itinerary,
        description="useful for formalizing and generating a travel itinerary based on gathered information."
    ),
]

# Initialize the LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

if __name__ == "__main__":
    print("Hello! I am your Smart Travel Planner agent. How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        try:
            response = agent_chain.invoke({"input": user_input})
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"Agent Error: {e}")
            print("Please try again or rephrase your request.")
