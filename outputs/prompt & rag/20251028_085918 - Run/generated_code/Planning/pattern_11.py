import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

# --- 1. Pydantic Models for Structured Data ---

class FlightDetails(BaseModel):
    origin: str
    destination: str
    departure_date: datetime.date
    return_date: Optional[datetime.date] = None
    price: float
    airline: Optional[str] = None
    flight_number: Optional[str] = None

class AccommodationDetails(BaseModel):
    location: str
    check_in_date: datetime.date
    check_out_date: datetime.date
    type: str  # e.g., "hotel", "hostel", "apartment"
    name: Optional[str] = None
    price_per_night: float
    total_price: float

class ActivityDetails(BaseModel):
    location: str
    date: datetime.date
    description: str
    price: float
    booking_required: bool = False

class LocalTransportDetails(BaseModel):
    city: str
    type: str # e.g., "public transport pass", "rental car"
    duration_days: int
    price: float

class SubTask(BaseModel):
    name: str
    description: str
    status: str = "PENDING" # PENDING, IN_PROGRESS, COMPLETED, FAILED
    result: Optional[Union[FlightDetails, AccommodationDetails, ActivityDetails, LocalTransportDetails, str]] = None
    dependencies: List[str] = Field(default_factory=list) # Names of other subtasks it depends on

class TravelRequest(BaseModel):
    user_id: str
    destinations: List[str]
    start_date: datetime.date
    end_date: datetime.date
    budget: float
    interests: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)

class TravelPlan(BaseModel):
    plan_id: str
    user_request: TravelRequest
    sub_tasks: List[SubTask] = Field(default_factory=list)
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    status: str = "DRAFT" # DRAFT, IN_PROGRESS, COMPLETED, FAILED, OPTIMIZED
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)

# --- 2. Mock LLM and Tools (Conceptual) ---

# Mock Langchain-like components for demonstration without actual API calls
class MockChatModel:
    """A mock LLM that simulates responses for planning tasks."""
    def invoke(self, prompt: str) -> str:
        if "decompose" in prompt.lower():
            return """
            [
                {"name": "FlightBooking", "description": "Find and book flights for all destinations.", "dependencies": []},
                {"name": "AccommodationBooking", "description": "Find and book accommodation for all destinations.", "dependencies": ["FlightBooking"]},
                {"name": "DailyItineraryPlanning", "description": "Plan daily activities for each destination.", "dependencies": ["AccommodationBooking"]},
                {"name": "LocalTransportPlanning", "description": "Plan local transport for each destination.", "dependencies": ["AccommodationBooking"]}
            ]
            """
        elif "generate initial plan" in prompt.lower():
            if "FlightBooking" in prompt:
                return "Simulated flight details: From NYC to LON, 2024-08-01, $600."
            elif "AccommodationBooking" in prompt:
                return "Simulated hotel details: Hotel Central, London, 2024-08-01 to 2024-08-05, $150/night."
            elif "DailyItineraryPlanning" in prompt:
                return "Simulated activities: Day 1: British Museum, Day 2: Tower of London."
            elif "LocalTransportPlanning" in prompt:
                return "Simulated transport: London Oyster Card, 5 days, $50."
        elif "optimize" in prompt.lower():
            return "Optimized plan: Adjusted activity order for efficiency."
        return "Mock LLM response for: " + prompt[:50] + "..."

class MockPromptTemplate:
    """A mock prompt template for demonstration."""
    def format(self, **kwargs) -> str:
        return f"User request: {kwargs.get('request_str', '')}\nInstruction: {kwargs.get('instruction', '')}\nContext: {kwargs.get('context', '')}"

class MockTools:
    """A mock class to simulate external API interactions."""
    def search_flights(self, origin: str, destination: str, date: datetime.date, return_date: Optional[datetime.date] = None, budget: Optional[float] = None) -> Optional[FlightDetails]:
        print(f"Mocking flight search for {origin} to {destination} on {date}...")
        if "NYC" in origin and "LON" in destination and date == datetime.date(2024, 8, 1):
            return FlightDetails(origin=origin, destination=destination, departure_date=date, return_date=return_date, price=600.0, airline="MockAir", flight_number="MA101")
        return None

    def search_accommodation(self, location: str, check_in: datetime.date, check_out: datetime.date, budget: Optional[float] = None, type: Optional[str] = None) -> Optional[AccommodationDetails]:
        print(f"Mocking accommodation search for {location} from {check_in} to {check_out}...")
        if "London" in location and check_in == datetime.date(2024, 8, 1) and check_out == datetime.date(2024, 8, 5):
            return AccommodationDetails(location=location, check_in_date=check_in, check_out_date=check_out, type="hotel", name="Mock Hotel", price_per_night=150.0, total_price=600.0)
        return None

    def find_activities(self, location: str, date: datetime.date, interests: Optional[List[str]] = None) -> List[ActivityDetails]:
        print(f"Mocking activity search for {location} on {date}...")
        if "London" in location and date == datetime.date(2024, 8, 2):
            return [ActivityDetails(location="London", date=date, description="Visit British Museum", price=0.0, booking_required=False)]
        return []

    def plan_local_transport(self, city: str, duration_days: int, budget: Optional[float] = None) -> Optional[LocalTransportDetails]:
        print(f"Mocking local transport planning for {city} for {duration_days} days...")
        if "London" in city and duration_days == 5:
            return LocalTransportDetails(city=city, type="public transport pass", duration_days=duration_days, price=50.0)
        return None

# --- 3. TravelPlanner Class (Core Planning Engine) ---

class TravelPlanner:
    def __init__(self, llm_client: MockChatModel, tools: MockTools):
        self.llm_client = llm_client
        self.tools = tools
        self.decomposition_prompt = MockPromptTemplate()
        self.plan_generation_prompt = MockPromptTemplate()
        self.optimization_prompt = MockPromptTemplate()

    def plan_trip(self, request: TravelRequest) -> TravelPlan:
        print(f"Starting to plan trip for user {request.user_id} to {request.destinations}...")

        # 1. Intent Understanding & Task Decomposition
        print("Step 1: Decomposing user request into sub-tasks...")
        # In a real scenario, this would involve LLM parsing and Pydantic validation
        # For this mock, we'll use a predefined structure derived from mock LLM output
        sub_task_dicts = [
            {"name": "FlightBooking", "description": f"Find and book flights for {request.destinations[0]}", "dependencies": []},
            {"name": "AccommodationBooking", "description": f"Find and book accommodation in {request.destinations[0]}", "dependencies": ["FlightBooking"]},
            {"name": "DailyItineraryPlanning", "description": f"Plan daily activities in {request.destinations[0]}", "dependencies": ["AccommodationBooking"]},
            {"name": "LocalTransportPlanning", "description": f"Plan local transport in {request.destinations[0]}", "dependencies": ["AccommodationBooking"]}
        ]
        sub_tasks = [SubTask(**d) for d in sub_task_dicts]
        
        current_plan = TravelPlan(plan_id=f"plan_{request.user_id}_{datetime.datetime.now().timestamp()}",
                                  user_request=request,
                                  sub_tasks=sub_tasks,
                                  status="IN_PROGRESS")

        # Keep track of completed task names for dependency management
        completed_tasks = set()

        # 2. Task Orchestration & Execution (Iterative and adaptive)
        # Execute tasks when all dependencies are met
        while any(task.status == "PENDING" for task in current_plan.sub_tasks):
            tasks_to_execute_in_iteration = []
            for task in current_plan.sub_tasks:
                if task.status == "PENDING":
                    dependencies_met = all(dep in completed_tasks for dep in task.dependencies)
                    if dependencies_met:
                        tasks_to_execute_in_iteration.append(task)
            
            if not tasks_to_execute_in_iteration:
                print("No pending tasks with met dependencies. Potentially stuck or plan completed.")
                break

            for task in tasks_to_execute_in_iteration:
                print(f"Step 2: Executing sub-task: {task.name} ({task.description})...")
                task.status = "IN_PROGRESS"
                try:
                    task_result = self._execute_subtask(task, request)
                    task.result = task_result
                    task.status = "COMPLETED"
                    completed_tasks.add(task.name)
                    
                    # Update estimated cost based on task result
                    if isinstance(task_result, (FlightDetails, AccommodationDetails, ActivityDetails, LocalTransportDetails)):
                        if hasattr(task_result, 'price'):
                            current_plan.estimated_cost += task_result.price
                        if hasattr(task_result, 'total_price'): # For accommodation with a total price field
                            current_plan.estimated_cost += task_result.total_price
                    
                    print(f"Sub-task {task.name} completed with result: {task.result}")
                except Exception as e:
                    task.status = "FAILED"
                    task.result = f"Error: {str(e)}"
                    print(f"Sub-task {task.name} failed: {e}")
            
            current_plan.last_updated = datetime.datetime.now()

        # 3. Reasoning & Optimization (Post-execution or iterative)
        print("Step 3: Optimizing the travel plan...")
        optimization_feedback = self.llm_client.invoke(
            self.optimization_prompt.format(
                request_str=request.model_dump_json(),
                context=current_plan.model_dump_json(),
                instruction="Analyze the current plan for inefficiencies, cost savings, or better routing based on user constraints and suggest improvements."
            )
        )
        print(f"Optimization feedback (mock): {optimization_feedback}")
        current_plan.status = "OPTIMIZED"
        current_plan.last_updated = datetime.datetime.now()

        print(f"Trip planning completed for user {request.user_id}.")
        return current_plan

    def _execute_subtask(self, subtask: SubTask, request: TravelRequest) -> Any:
        """
        Simulates interaction with external tools/APIs based on the subtask.
        """
        if subtask.name == "FlightBooking":
            origin = "NYC" # Hardcoded for mock, normally determined by LLM or previous steps
            destination = request.destinations[0]
            flight_details = self.tools.search_flights(
                origin=origin,
                destination=destination,
                date=request.start_date,
                return_date=request.end_date,
                budget=request.budget * 0.3 # Example budget allocation
            )
            if not flight_details:
                raise ValueError("Could not find suitable flights.")
            return flight_details
        
        elif subtask.name == "AccommodationBooking":
            location = request.destinations[0]
            accommodation_details = self.tools.search_accommodation(
                location=location,
                check_in=request.start_date,
                check_out=request.end_date,
                budget=request.budget * 0.4 # Example budget allocation
            )
            if not accommodation_details:
                raise ValueError("Could not find suitable accommodation.")
            return accommodation_details

        elif subtask.name == "DailyItineraryPlanning":
            location = request.destinations[0]
            activities = self.tools.find_activities(
                location=location,
                date=request.start_date + datetime.timedelta(days=1), # Example for a day after arrival
                interests=request.interests
            )
            if not activities:
                return "No activities found for this day."
            return activities

        elif subtask.name == "LocalTransportPlanning":
            city = request.destinations[0]
            duration_days = (request.end_date - request.start_date).days + 1 # Include start and end day
            transport = self.tools.plan_local_transport(
                city=city,
                duration_days=duration_days,
                budget=request.budget * 0.1 # Example budget allocation
            )
            if not transport:
                return "Could not plan local transport."
            return transport

        # Fallback for unhandled subtasks (can use LLM to generate generic output)
        return self.llm_client.invoke(
            self.plan_generation_prompt.format(
                request_str=request.model_dump_json(),
                subtask_description=subtask.description,
                context=subtask.model_dump_json(),
                instruction=f"Generate an initial plan/result for the subtask: {subtask.name}."
            )
        )

# --- Example Usage ---
def main():
    # Initialize mock components
    mock_llm = MockChatModel()
    mock_tools = MockTools()
    planner = TravelPlanner(llm_client=mock_llm, tools=mock_tools)

    # Define a sample travel request
    sample_request = TravelRequest(
        user_id="user123",
        destinations=["London"],
        start_date=datetime.date(2024, 8, 1),
        end_date=datetime.date(2024, 8, 5),
        budget=2000.00,
        interests=["museums", "history", "food"],
        preferences={"accommodation_type": "hotel"}
    )

    # Plan the trip
    final_plan = planner.plan_trip(sample_request)

    print("\n--- Final Travel Plan ---")
    print(f"Plan ID: {final_plan.plan_id}")
    print(f"Status: {final_plan.status}")
    print(f"Estimated Total Cost: ${final_plan.estimated_cost:.2f}")
    print("Sub-tasks:")
    for task in final_plan.sub_tasks:
        print(f"  - {task.name}: {task.status}")
        if task.result:
            if isinstance(task.result, list):
                # Handle list of activities specifically
                activity_descriptions = [r.description if isinstance(r, ActivityDetails) else str(r) for r in task.result]
                print(f"    Result: {activity_descriptions}")
            elif isinstance(task.result, BaseModel):
                print(f"    Result: {task.result.model_dump_json(indent=2)}")
            else:
                print(f"    Result: {task.result}")

if __name__ == "__main__":
    main()