from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_core.runnables import Runnable
from langchain_community.llms import OpenAI

class MemoryManager:
    def __init__(self):
        self.conversational_history = []
        self.user_preferences = {}
        self.trip_plan_details = {}

    def add_message(self, role, content):
        self.conversational_history.append({"role": role, "content": content})

    def get_history(self):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversational_history])

    def update_preferences(self, key, value):
        self.user_preferences[key] = value

    def get_preferences(self):
        return self.user_preferences

    def update_trip_plan(self, key, value):
        self.trip_plan_details[key] = value

    def get_trip_plan(self):
        return self.trip_plan_details

class TravelPlannerTools:
    def search_flights(self, query: str) -> str:
        return f"Searching for flights with query: {query}. Found flights from Example Airlines."

    def book_hotel(self, query: str) -> str:
        return f"Booking hotel with query: {query}. Booked Example Hotel."

    def create_itinerary(self, query: str) -> str:
        return f"Creating itinerary with query: {query}. Itinerary for Example City generated."

    def recommend_activities(self, query: str) -> str:
        return f"Recommending activities for query: {query}. Suggested sightseeing and dining."

class TravelPlannerAgent:
    def __init__(self, llm_api_key: str):
        self.memory = MemoryManager()
        self.tools_instance = TravelPlannerTools()
        self.llm = OpenAI(openai_api_key=llm_api_key)

        self.tools = [
            Tool(
                name="SearchFlights",
                func=self.tools_instance.search_flights,
                description="Searches for flight information based on origin, destination, dates, and other criteria."
            ),
            Tool(
                name="BookHotel",
                func=self.tools_instance.book_hotel,
                description="Books a hotel based on location, dates, and preferences."
            ),
            Tool(
                name="CreateItinerary",
                func=self.tools_instance.create_itinerary,
                description="Generates a detailed travel itinerary for a given destination and duration."
            ),
            Tool(
                name="RecommendActivities",
                func=self.tools_instance.recommend_activities,
                description="Recommends activities and attractions for a specific location."
            ),
        ]

        prompt_template = PromptTemplate.from_template(
            """You are an autonomous travel planner agent. Your goal is to help users plan multi-day trips.
            You have access to tools to search flights, book hotels, create itineraries, and recommend activities.
            Maintain a conversation and remember user preferences. Adapt plans dynamically.

            Current conversation history:
            {chat_history}

            User preferences:
            {user_preferences}

            Current trip plan details:
            {trip_plan}

            Question: {input}
            Thought:"""
        )

        self.agent = create_react_agent(self.llm, self.tools, prompt_template)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def run(self, user_input: str) -> str:
        self.memory.add_message("user", user_input)
        
        # Prepare the input for the agent, including memory context
        agent_input = {
            "input": user_input,
            "chat_history": self.memory.get_history(),
            "user_preferences": self.memory.get_preferences(),
            "trip_plan": self.memory.get_trip_plan()
        }

        response = self.agent_executor.invoke(agent_input)
        agent_output = response["output"]
        self.memory.add_message("agent", agent_output)

        # Example of how the agent might update memory based on its actions/responses
        # In a more advanced setup, parsing agent's natural language output or tool outputs
        # would drive memory updates.
        if "budget" in user_input.lower() and "$500" in user_input:
            self.memory.update_preferences("budget", "$500")
        if "london" in user_input.lower() and "trip" in user_input.lower():
            self.memory.update_trip_plan("destination", "London")

        return agent_output

if __name__ == "__main__":
    # Replace with your actual OpenAI API key
    # It's recommended to load this from environment variables
    import os
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable.")
    else:
        travel_agent = TravelPlannerAgent(llm_api_key=os.getenv("OPENAI_API_KEY"))

        print("Travel Agent initialized. Type 'exit' to quit.")
        while True:
            user_query = input("You: ")
            if user_query.lower() == 'exit':
                break
            agent_response = travel_agent.run(user_query)
            print(f"Agent: {agent_response}")
            print(f"Current Preferences: {travel_agent.memory.get_preferences()}")
            print(f"Current Trip Plan: {travel_agent.memory.get_trip_plan()}")