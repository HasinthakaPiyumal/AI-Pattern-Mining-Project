import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.7)

def book_flight(destination: str, date: str, passengers: int) -> str:
    return f"Flight booked to {destination} on {date} for {passengers} passengers (simulated)."

def find_hotel(location: str, check_in: str, check_out: str) -> str:
    return f"Searching for hotels in {location} from {check_in} to {check_out} (simulated)."

def plan_itinerary(destination: str, duration: str, interests: str) -> str:
    return f"Planning a {duration} trip to {destination} with interests in {interests} (simulated)."

tools = [
    DuckDuckGoSearchRun(name="Web_Search"),
    Tool(
        name="Book_Flight",
        func=book_flight,
        description="Useful for booking flights. Requires destination, date, and number of passengers.",
    ),
    Tool(
        name="Find_Hotel",
        func=find_hotel,
        description="Useful for finding hotels. Requires location, check-in date, and check-out date.",
    ),
    Tool(
        name="Plan_Itinerary",
        func=plan_itinerary,
        description="Useful for planning a travel itinerary. Requires destination, duration (e.g., '5 days'), and user interests (e.g., 'historical sites, food').",
    ),
]

memory = ConversationBufferWindowMemory(
    memory_key="chat_history", return_messages=True, k=5
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an intelligent travel assistant. Help users plan their trips by answering questions, booking flights/hotels (simulated), and planning itineraries."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

print("Welcome to your Intelligent Travel Assistant! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    try:
        response = agent_executor.invoke({"input": user_input})
        print(f"Agent: {response['output']}")
    except Exception as e:
        print(f"Agent Error: {e}")