from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Tuple
import random

class Destination(BaseModel):
    name: str
    typical_travel_time: float
    cost_factor: float
    satisfaction_factor: float

class Route(BaseModel):
    origin: str
    destination: str
    base_duration: float
    base_cost: float

class UserProfile(BaseModel):
    budget: float
    preferred_travel_speed: float
    interests: List[str]

class TravelRequest(BaseModel):
    start_destination: str
    end_destination: str
    desired_date: str
    num_days: int
    budget: float

class ItinerarySegment(BaseModel):
    type: str
    origin: str
    destination: str
    duration: float
    cost: float
    description: str

class TravelItinerary(BaseModel):
    segments: List[ItinerarySegment]
    total_cost: float = 0.0
    total_duration: float = 0.0
    estimated_satisfaction: float = 0.0

class SimulationResult(BaseModel):
    total_cost: float
    total_duration: float
    estimated_satisfaction: float
    potential_issues: List[str]

class WorldModel:
    def __init__(self):
        self.destinations: Dict[str, Destination] = {
            "New York": Destination(name="New York", typical_travel_time=5.0, cost_factor=1.5, satisfaction_factor=0.8),
            "London": Destination(name="London", typical_travel_time=7.0, cost_factor=1.8, satisfaction_factor=0.9),
            "Paris": Destination(name="Paris", typical_travel_time=6.0, cost_factor=1.7, satisfaction_factor=0.95),
            "Tokyo": Destination(name="Tokyo", typical_travel_time=12.0, cost_factor=2.5, satisfaction_factor=0.92),
            "Rome": Destination(name="Rome", typical_travel_time=6.5, cost_factor=1.6, satisfaction_factor=0.88),
            "Berlin": Destination(name="Berlin", typical_travel_time=6.0, cost_factor=1.4, satisfaction_factor=0.85),
            "Madrid": Destination(name="Madrid", typical_travel_time=5.5, cost_factor=1.3, satisfaction_factor=0.87),
            "Amsterdam": Destination(name="Amsterdam", typical_travel_time=5.0, cost_factor=1.45, satisfaction_factor=0.89),
        }

        self.routes: Dict[Tuple[str, str], Route] = {
            ("New York", "London"): Route(origin="New York", destination="London", base_duration=7.0, base_cost=800.0),
            ("London", "New York"): Route(origin="London", destination="New York", base_duration=7.0, base_cost=750.0),
            ("London", "Paris"): Route(origin="London", destination="Paris", base_duration=2.0, base_cost=150.0),
            ("Paris", "London"): Route(origin="Paris", destination="London", base_duration=2.0, base_cost=140.0),
            ("Paris", "Rome"): Route(origin="Paris", destination="Rome", base_duration=1.5, base_cost=100.0),
            ("Rome", "Paris"): Route(origin="Rome", destination="Paris", base_duration=1.5, base_cost=90.0),
            ("New York", "Tokyo"): Route(origin="New York", destination="Tokyo", base_duration=14.0, base_cost=1500.0),
            ("Tokyo", "New York"): Route(origin="Tokyo", destination="New York", base_duration=14.0, base_cost=1450.0),
            ("London", "Berlin"): Route(origin="London", destination="Berlin", base_duration=1.5, base_cost=120.0),
            ("Berlin", "London"): Route(origin="Berlin", destination="London", base_duration=1.5, base_cost=110.0),
            ("Paris", "Madrid"): Route(origin="Paris", destination="Madrid", base_duration=2.0, base_cost=130.0),
            ("Madrid", "Paris"): Route(origin="Madrid", destination="Paris", base_duration=2.0, base_cost=125.0),
            ("Berlin", "Amsterdam"): Route(origin="Berlin", destination="Amsterdam", base_duration=1.0, base_cost=80.0),
            ("Amsterdam", "Berlin"): Route(origin="Amsterdam", destination="Berlin", base_duration=1.0, base_cost=75.0),
        }

    def get_destination(self, name: str) -> Destination or None:
        return self.destinations.get(name)

    def get_route(self, origin: str, destination: str) -> Route or None:
        return self.routes.get((origin, destination))

class TravelAgent:
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def generate_initial_plan(self, travel_request: TravelRequest, user_profile: UserProfile) -> TravelItinerary:
        segments: List[ItinerarySegment] = []
        current_location = travel_request.start_destination
        total_cost = 0.0
        total_duration = 0.0

        while current_location != travel_request.end_destination:
            route = self.world_model.get_route(current_location, travel_request.end_destination)
            if route:
                segment = ItinerarySegment(
                    type="Flight",
                    origin=current_location,
                    destination=travel_request.end_destination,
                    duration=route.base_duration,
                    cost=route.base_cost,
                    description=f"Direct flight from {current_location} to {travel_request.end_destination}"
                )
                segments.append(segment)
                total_cost += segment.cost
                total_duration += segment.duration
                current_location = travel_request.end_destination
            else:
                possible_next_destinations = [d for d in self.world_model.destinations.keys() if d != current_location and self.world_model.get_route(current_location, d)]
                if not possible_next_destinations:
                    break
                next_destination = random.choice(possible_next_destinations)
                route_to_next = self.world_model.get_route(current_location, next_destination)
                if route_to_next:
                    segment = ItinerarySegment(
                        type="Flight",
                        origin=current_location,
                        destination=next_destination,
                        duration=route_to_next.base_duration,
                        cost=route_to_next.base_cost,
                        description=f"Flight from {current_location} to {next_destination} (stopover)"
                    )
                    segments.append(segment)
                    total_cost += segment.cost
                    total_duration += segment.duration
                    current_location = next_destination
                else:
                    break

        return TravelItinerary(segments=segments, total_cost=total_cost, total_duration=total_duration, estimated_satisfaction=0.0)

    def simulate_itinerary(self, itinerary: TravelItinerary) -> SimulationResult:
        simulated_cost = itinerary.total_cost
        simulated_duration = itinerary.total_duration
        simulated_satisfaction = 0.0
        potential_issues: List[str] = []

        for segment in itinerary.segments:
            destination_info = self.world_model.get_destination(segment.destination)
            if destination_info:
                simulated_satisfaction += destination_info.satisfaction_factor * segment.duration
            
            if random.random() < 0.1: 
                simulated_duration += random.uniform(0.5, 2.0) 
                potential_issues.append(f"Delay on segment from {segment.origin} to {segment.destination}")

            if random.random() < 0.05: 
                simulated_cost += random.uniform(50.0, 200.0) 
                potential_issues.append(f"Unexpected cost increase on segment from {segment.origin} to {segment.destination}")

        if not itinerary.segments:
            simulated_satisfaction = 0.0
        else:
            simulated_satisfaction /= itinerary.total_duration if itinerary.total_duration > 0 else 1.0

        return SimulationResult(
            total_cost=simulated_cost,
            total_duration=simulated_duration,
            estimated_satisfaction=simulated_satisfaction,
            potential_issues=potential_issues
        )

    def refine_plan(self, itinerary: TravelItinerary, simulation_result: SimulationResult, user_profile: UserProfile) -> TravelItinerary:
        refined_itinerary = itinerary.copy(deep=True)

        if simulation_result.total_cost > user_profile.budget * 1.1: 
            cheaper_alternatives = []
            for i, segment in enumerate(refined_itinerary.segments):
                # A very simplistic attempt to find cheaper alternatives
                # In a real scenario, this would involve searching for other routes or even destinations
                if segment.cost > 200.0:
                    new_cost = segment.cost * 0.8 
                    cheaper_alternatives.append((i, new_cost))
            
            if cheaper_alternatives:
                idx, new_cost = random.choice(cheaper_alternatives)
                refined_itinerary.segments[idx].cost = new_cost
                print(f"Refining: Reduced cost for segment {idx}")

        # More complex refinement logic could go here based on duration, satisfaction, interests, etc.

        refined_itinerary.total_cost = sum(s.cost for s in refined_itinerary.segments)
        refined_itinerary.total_duration = sum(s.duration for s in refined_itinerary.segments)
        
        return refined_itinerary

    def plan_trip(self, travel_request: TravelRequest, user_profile: UserProfile, max_iterations: int = 5) -> TravelItinerary:
        current_itinerary = self.generate_initial_plan(travel_request, user_profile)

        for i in range(max_iterations):
            simulation_result = self.simulate_itinerary(current_itinerary)
            
            if simulation_result.total_cost <= user_profile.budget and not simulation_result.potential_issues: 
                current_itinerary.estimated_satisfaction = simulation_result.estimated_satisfaction
                current_itinerary.total_cost = simulation_result.total_cost
                current_itinerary.total_duration = simulation_result.total_duration
                return current_itinerary

            current_itinerary = self.refine_plan(current_itinerary, simulation_result, user_profile)
            
        final_simulation = self.simulate_itinerary(current_itinerary)
        current_itinerary.estimated_satisfaction = final_simulation.estimated_satisfaction
        current_itinerary.total_cost = final_simulation.total_cost
        current_itinerary.total_duration = final_simulation.total_duration
        return current_itinerary

app = FastAPI()

@app.post("/plan_trip", response_model=TravelItinerary)
async def plan_travel_trip(travel_request: TravelRequest, user_profile: UserProfile):
    world_model = WorldModel()
    travel_agent = TravelAgent(world_model)
    optimized_itinerary = travel_agent.plan_trip(travel_request, user_profile)
    return optimized_itinerary
