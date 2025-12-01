import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Simulate environment variable loading
class DotEnv:
    def load_dotenv(self):
        pass # In a real app, this would load .env file

dotenv_loader = DotEnv()
dotenv_loader.load_dotenv()

# --- Simulated LLM Core ---
class MockOpenAI:
    def __init__(self, api_key: str = "mock_key"):
        self.api_key = api_key

    def chat_completion(self, messages: List[Dict], max_tokens: int = 150) -> Dict:
        last_message = messages[-1]["content"]
        if "plan a trip to" in last_message.lower():
            return {"choices": [{"message": {"content": "Plan trip to destination. Find flights. Find accommodation. Suggest activities. Create itinerary."}}]}
        elif "book flight" in last_message.lower():
            return {"choices": [{"message": {"content": "Use FlightBookingTool to book flight."}}]}
        elif "book hotel" in last_message.lower():
            return {"choices": [{"message": {"content": "Use AccommodationBookingTool to book hotel."}}]}
        elif "suggest activities for" in last_message.lower():
            return {"choices": [{"message": {"content": "Use ActivitySuggestionTool to suggest activities."}}]}
        elif "check status" in last_message.lower():
            return {"choices": [{"message": {"content": "Use RealTimeMonitorTool to check status."}}]}
        elif "create itinerary" in last_message.lower():
            return {"choices": [{"message": {"content": "Use ItineraryManagementTool to create itinerary."}}]}
        return {"choices": [{"message": {"content": f"Understood: {last_message}"}}]}

class LLM:
    def __init__(self):
        self.client = MockOpenAI(api_key=os.getenv("OPENAI_API_KEY", "mock_openai_key"))

    def generate_response(self, prompt: str, history: List[Dict]) -> str:
        messages = [{"role": "system", "content": "You are a helpful travel planning agent."}]
        messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat_completion(messages=messages)
        return response["choices"][0]["message"]["content"]

# --- Memory Module ---
class ShortTermMemory:
    def __init__(self):
        self.conversation_history: List[Dict] = []

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict]:
        return self.conversation_history.copy()

    def clear(self):
        self.conversation_history = []

class LongTermMemory:
    def __init__(self):
        self.user_preferences: Dict[str, Any] = {}
        self.past_trips: List[Dict] = []

    def add_preference(self, key: str, value: Any):
        self.user_preferences[key] = value

    def get_preference(self, key: str, default: Any = None) -> Any:
        return self.user_preferences.get(key, default)

    def add_past_trip(self, trip_details: Dict):
        self.past_trips.append(trip_details)

    def retrieve_info(self, query: str) -> List[str]:
        # Simulate vector search for relevant info
        relevant_info = []
        if "beach" in query.lower() and self.get_preference("favorite_type", "") == "beach":
            relevant_info.append("User loves beach destinations.")
        return relevant_info

# --- Tool Use Module ---
class Tool:
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class FlightBookingParams(BaseModel):
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str] = None

def flight_booking_tool(params: FlightBookingParams) -> str:
    return f"Simulating flight booking from {params.origin} to {params.destination} on {params.departure_date}."

class AccommodationBookingParams(BaseModel):
    location: str
    check_in_date: str
    check_out_date: str
    guests: int

def accommodation_booking_tool(params: AccommodationBookingParams) -> str:
    return f"Simulating accommodation booking in {params.location} from {params.check_in_date} to {params.check_out_date} for {params.guests} guests."

class ActivitySuggestionParams(BaseModel):
    location: str
    date: Optional[str] = None
    type: Optional[str] = None

def activity_suggestion_tool(params: ActivitySuggestionParams) -> str:
    return f"Suggesting activities for {params.location}, focusing on {params.type or 'any type'}."

class RealTimeMonitorParams(BaseModel):
    item: str
    query: str

def real_time_monitor_tool(params: RealTimeMonitorParams) -> str:
    return f"Checking real-time status for {params.item} with query '{params.query}'. E.g., 'Flight AA123: On time. Weather London: Sunny.'"

class ItineraryManagementParams(BaseModel):
    action: str
    details: Dict

def itinerary_management_tool(params: ItineraryManagementParams) -> str:
    if params.action == "create":
        return f"Creating itinerary with details: {params.details}"
    elif params.action == "update":
        return f"Updating itinerary with details: {params.details}"
    elif params.action == "display":
        return f"Displaying itinerary: {params.details}"
    return "Unknown itinerary action."


# --- Planning Module ---
class PlanningModule:
    def __init__(self, llm: LLM, available_tools: Dict[str, Tool]):
        self.llm = llm
        self.available_tools = available_tools

    def decompose_task(self, query: str, history: List[Dict]) -> List[str]:
        prompt = f"Decompose the following travel request into simple, sequential steps: {query}"
        response = self.llm.generate_response(prompt, history)
        # Mock decomposition for this example
        if "plan a trip" in query.lower():
            return ["understand user preferences", "find flights", "find accommodation", "suggest activities", "create itinerary"]
        elif "book flight" in query.lower():
            return ["get flight details from user", "book flight"]
        return ["process_query"]

    def select_action(self, task: str, history: List[Dict]) -> Optional[Tool]:
        prompt = f"Given the task '{task}', which tool should be used? Available tools: {', '.join(self.available_tools.keys())}. Respond with just the tool name or 'None'."
        response = self.llm.generate_response(prompt, history).strip()
        for tool_name in self.available_tools.keys():
            if tool_name.lower() in response.lower():
                return self.available_tools[tool_name]
        return None

    def replan(self, current_plan: List[str], current_status: str, history: List[Dict]) -> List[str]:
        prompt = f"Given the current plan: {current_plan}, and the status: {current_status}, suggest a revised plan or confirm the current plan is good. Provide a new list of steps."
        response = self.llm.generate_response(prompt, history)
        # Mock replanning
        if "flight delay" in current_status.lower() and "find flights" in current_plan:
            return ["adjust flight booking", "notify user"] + current_plan[current_plan.index("find flights")+1:]
        return current_plan

# --- Main Travel Agent ---
class TravelAgent:
    def __init__(self):
        self.llm = LLM()
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self.tools = {
            "FlightBookingTool": Tool("FlightBookingTool", "Books flights", flight_booking_tool),
            "AccommodationBookingTool": Tool("AccommodationBookingTool", "Books accommodations", accommodation_booking_tool),
            "ActivitySuggestionTool": Tool("ActivitySuggestionTool", "Suggests activities", activity_suggestion_tool),
            "RealTimeMonitorTool": Tool("RealTimeMonitorTool", "Monitors real-time events", real_time_monitor_tool),
            "ItineraryManagementTool": Tool("ItineraryManagementTool", "Manages travel itineraries", itinerary_management_tool),
        }
        self.planning_module = PlanningModule(self.llm, self.tools)
        self.current_itinerary: Dict = {}

    def process_query(self, query: str) -> str:
        self.short_term_memory.add_message("user", query)
        history = self.short_term_memory.get_history()

        # Step 1: Decompose Task
        tasks = self.planning_module.decompose_task(query, history)
        response_messages = []

        for task in tasks:
            self.short_term_memory.add_message("system", f"Current task: {task}")
            tool_to_use = self.planning_module.select_action(task, self.short_term_memory.get_history())

            if tool_to_use:
                response_messages.append(f"Agent will use {tool_to_use.name} for task: {task}")
                
                # Simulate tool parameter extraction and execution
                if tool_to_use.name == "FlightBookingTool":
                    # In a real app, LLM would extract these params from query/memory
                    tool_output = tool_to_use(FlightBookingParams(origin="London", destination="Paris", departure_date="2024-07-20"))
                elif tool_to_use.name == "AccommodationBookingTool":
                    tool_output = tool_to_use(AccommodationBookingParams(location="Paris", check_in_date="2024-07-20", check_out_date="2024-07-25", guests=2))
                elif tool_to_use.name == "ActivitySuggestionTool":
                    tool_output = tool_to_use(ActivitySuggestionParams(location="Paris", type="sightseeing"))
                elif tool_to_use.name == "RealTimeMonitorTool":
                    tool_output = tool_to_use(RealTimeMonitorParams(item="weather", query="Paris"))
                elif tool_to_use.name == "ItineraryManagementTool":
                    if task == "create itinerary":
                        self.current_itinerary = {"trip_to": "Paris", "dates": "2024-07-20 to 2024-07-25"}
                        tool_output = tool_to_use(ItineraryManagementParams(action="create", details=self.current_itinerary))
                    elif task == "display itinerary":
                         tool_output = tool_to_use(ItineraryManagementParams(action="display", details=self.current_itinerary))
                    else:
                         tool_output = tool_to_use(ItineraryManagementParams(action="update", details={"note": "example update"}))
                else:
                    tool_output = f"Executed simulated tool: {tool_to_use.name}"
                
                response_messages.append(f"Tool Output: {tool_output}")
                self.short_term_memory.add_message("tool", tool_output)

                # Step 5: Re-plan based on tool output or real-time events (simulated)
                if "flight booking from London to Paris" in tool_output:
                    # Simulate a real-time event that requires replanning
                    tasks = self.planning_module.replan(tasks, "Flight AA123 delayed by 2 hours", self.short_term_memory.get_history())
                    response_messages.append("Agent detected a flight delay and re-planned.")

            else:
                response_messages.append(f"Agent couldn't find a specific tool for task: {task}. Using LLM for general response.")
                llm_response = self.llm.generate_response(f"Regarding '{task}': {query}", self.short_term_memory.get_history())
                response_messages.append(f"LLM Response: {llm_response}")
                self.short_term_memory.add_message("assistant", llm_response)
        
        final_response = "\n".join(response_messages)
        self.short_term_memory.add_message("assistant", final_response)
        return final_response

    def display_current_itinerary(self) -> str:
        if self.current_itinerary:
            return f"Current Itinerary: {self.current_itinerary}"
        return "No itinerary planned yet."


if __name__ == "__main__":
    agent = TravelAgent()
    
    # Simulate user preferences in long-term memory
    agent.long_term_memory.add_preference("favorite_type", "beach")
    agent.long_term_memory.add_preference("dietary_restrictions", "vegetarian")

    print("--- User Query 1: Plan a trip ---")
    response1 = agent.process_query("Plan me a 5-day trip to Paris, France. I like sightseeing and good food.")
    print(response1)
    print("\n" + "="*50 + "\n")

    print("--- User Query 2: Check current itinerary ---")
    response2 = agent.process_query("What's my current itinerary?")
    print(response2)
    print("\n" + "="*50 + "\n")

    print("--- User Query 3: Simulate real-time monitoring and replanning ---")
    # This query directly triggers a simulated replan scenario
    response3 = agent.process_query("My flight to Paris is delayed. What should I do?")
    print(response3)
    print("\n" + "="*50 + "\n")

    print("--- User Query 4: Check long-term memory ---")
    retrieved_info = agent.long_term_memory.retrieve_info("I'm looking for a beach vacation.")
    print(f"Retrieved from long-term memory: {retrieved_info}")

    print("\nFinal Itinerary Display:")
    print(agent.display_current_itinerary())
