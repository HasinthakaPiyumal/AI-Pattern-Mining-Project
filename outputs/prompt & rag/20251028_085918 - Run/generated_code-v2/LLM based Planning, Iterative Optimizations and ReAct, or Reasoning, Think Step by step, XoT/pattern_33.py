import sys

# --- 2. Data Simulation Module ---
FLIGHT_DATA = {
    ("New York", "London"): [("NYC-LDN-001", 600, 7), ("NYC-LDN-002", 700, 8)],
    ("London", "Paris"): [("LDN-PRS-001", 150, 2), ("LDN-PRS-002", 180, 2.5)],
    ("Paris", "Rome"): [("PRS-ROM-001", 120, 1.5), ("PRS-ROM-002", 140, 1.8)],
    ("Rome", "New York"): [("ROM-NYC-001", 550, 9), ("ROM-NYC-002", 650, 10)],
    ("New York", "Paris"): [("NYC-PRS-001", 750, 8), ("NYC-PRS-002", 800, 8.5)],
    ("London", "Rome"): [("LDN-ROM-001", 200, 3), ("LDN-ROM-002", 230, 3.2)],
}

ACCOMMODATION_DATA = {
    ("New York", 0): [("NYCHotelA", 200), ("NYCHotelB", 250)], # Simplified: day 0 for initial stay planning
    ("New York", 1): [("NYCHotelA", 200), ("NYCHotelB", 250)],
    ("New York", 2): [("NYCHotelA", 200), ("NYCHotelB", 250)],
    ("New York", 3): [("NYCHotelA", 200), ("NYCHotelB", 250)],
    ("London", 0): [("LDNHotelA", 180), ("LDNHotelB", 220)],
    ("London", 1): [("LDNHotelA", 180), ("LDNHotelB", 220)],
    ("London", 2): [("LDNHotelA", 180), ("LDNHotelB", 220)],
    ("London", 3): [("LDNHotelA", 180), ("LDNHotelB", 220)],
    ("Paris", 0): [("PRSHotelA", 150), ("PRSHotelB", 190)],
    ("Paris", 1): [("PRSHotelA", 150), ("PRSHotelB", 190)],
    ("Paris", 2): [("PRSHotelA", 150), ("PRSHotelB", 190)],
    ("Paris", 3): [("PRSHotelA", 150), ("PRSHotelB", 190)],
    ("Rome", 0): [("ROMHotelA", 130), ("ROMHotelB", 170)],
    ("Rome", 1): [("ROMHotelA", 130), ("ROMHotelB", 170)],
    ("Rome", 2): [("ROMHotelA", 130), ("ROMHotelB", 170)],
    ("Rome", 3): [("ROMHotelA", 130), ("ROMHotelB", 170)],
}

class TravelSegment:
    def __init__(self, origin, destination, flight, accommodation, nights, departure_day, arrival_day, total_segment_cost):
        self.origin = origin
        self.destination = destination
        self.flight = flight
        self.accommodation = accommodation
        self.nights = nights
        self.departure_day = departure_day
        self.arrival_day = arrival_day
        self.total_segment_cost = total_segment_cost

    def __repr__(self):
        return (
            f"Segment(From: {self.origin}, To: {self.destination}, "
            f"Flight: {self.flight[0]} (${self.flight[1]}), "
            f"Hotel: {self.accommodation[0]} (${self.accommodation[1]}/night for {self.nights} nights), "
            f"Dep Day: {self.departure_day}, Arr Day: {self.arrival_day}, "
            f"Cost: ${self.total_segment_cost})")

class TravelPlan:
    def __init__(self, budget, min_night_stays=None):
        self.itinerary = []
        self.total_cost = 0
        self.remaining_budget = budget
        self.visited_destinations = set()
        self.min_night_stays = min_night_stays if min_night_stays is not None else {}
        self.current_day = 0

    def add_segment(self, segment):
        self.itinerary.append(segment)
        self.total_cost += segment.total_segment_cost
        self.remaining_budget -= segment.total_segment_cost
        self.visited_destinations.add(segment.destination)
        self.current_day = segment.arrival_day + segment.nights

    def remove_last_segment(self):
        if not self.itinerary:
            return None
        segment = self.itinerary.pop()
        self.total_cost -= segment.total_segment_cost
        self.remaining_budget += segment.total_segment_cost
        self.visited_destinations = set(s.origin for s in self.itinerary) | set(s.destination for s in self.itinerary)
        if self.itinerary:
            self.current_day = self.itinerary[-1].arrival_day + self.itinerary[-1].nights
        else:
            self.current_day = 0
        return segment

    def check_constraints(self):
        if self.remaining_budget < 0:
            return False

        destination_nights = {}
        for segment in self.itinerary:
            destination_nights[segment.destination] = destination_nights.get(segment.destination, 0) + segment.nights

        for dest, min_nights in self.min_night_stays.items():
            if dest in destination_nights and destination_nights[dest] < min_nights:
                return False
        return True

    def __str__(self):
        itinerary_str = "\n".join([str(s) for s in self.itinerary])
        return (
            f"--- Travel Plan ---\n"
            f"Total Cost: ${self.total_cost:.2f}\n"
            f"Remaining Budget: ${self.remaining_budget:.2f}\n"
            f"Visited: {list(self.visited_destinations)}\n"
            f"Itinerary:\n{itinerary_str}\n"
            f"-------------------\n")

    def __copy__(self):
        new_plan = TravelPlan(self.budget + self.total_cost, self.min_night_stays)
        new_plan.itinerary = [sys.modules[__name__].TravelSegment(s.origin, s.destination, s.flight, s.accommodation, s.nights, s.departure_day, s.arrival_day, s.total_segment_cost) for s in self.itinerary]
        new_plan.total_cost = self.total_cost
        new_plan.remaining_budget = self.remaining_budget
        new_plan.visited_destinations = self.visited_destinations.copy()
        new_plan.current_day = self.current_day
        return new_plan

class TravelAgent:
    def __init__(self, destinations, start_location, start_day, budget, constraints):
        self.destinations = destinations
        self.initial_start_location = start_location
        self.initial_start_day = start_day
        self.budget = budget
        self.min_night_stays = constraints.get("min_night_stays", {})
        self.max_trip_duration = constraints.get("max_trip_duration", float('inf'))

        self.simulated_flights = FLIGHT_DATA
        self.simulated_accommodations = ACCOMMODATION_DATA

        self.best_plan = None
        self.min_cost = float('inf')

    def _get_possible_options(self, current_location, current_day, destination):
        options = []
        flights = self.simulated_flights.get((current_location, destination), [])
        for flight_id, flight_cost, _ in flights:
            for nights in range(1, 4):
                if destination in self.min_night_stays and nights < self.min_night_stays[destination]:
                    continue

                arrival_day = current_day + 1
                total_accommodation_cost = 0
                accommodation_found = True
                representative_accommodation_detail = ("NoHotel", 0)
                
                for night_offset in range(nights):
                    acc_day = arrival_day + night_offset
                    accommodation_options = self.simulated_accommodations.get((destination, acc_day), [])
                    if not accommodation_options:
                        accommodation_found = False
                        break
                    cheapest_acc = min(accommodation_options, key=lambda x: x[1])
                    total_accommodation_cost += cheapest_acc[1]
                    if night_offset == 0: # Store info of the first night's hotel as representative
                        representative_accommodation_detail = cheapest_acc

                if accommodation_found:
                    # For display, show the first night's hotel and its cost per night
                    representative_accommodation = (f"{representative_accommodation_detail[0]}({nights} nights)", representative_accommodation_detail[1])
                    total_segment_cost = flight_cost + total_accommodation_cost
                    segment = TravelSegment(
                        origin=current_location,
                        destination=destination,
                        flight=(flight_id, flight_cost, 0),
                        accommodation=representative_accommodation,
                        nights=nights,
                        departure_day=current_day,
                        arrival_day=arrival_day,
                        total_segment_cost=total_segment_cost
                    )
                    options.append(segment)
        return options

    def _heuristic_score_segment(self, current_plan, segment, remaining_destinations):
        score = segment.total_segment_cost
        if current_plan.total_cost + segment.total_segment_cost > self.budget:
            score += 10000
        score += len(remaining_destinations) * 50
        return score

    def _find_best_plan_recursive(self, current_plan, current_location, remaining_destinations):
        if not remaining_destinations:
            if current_plan.total_cost < self.min_cost and current_plan.check_constraints():
                self.min_cost = current_plan.total_cost
                self.best_plan = current_plan.__copy__()
                return True
            return False

        if current_plan.total_cost >= self.min_cost:
            return False

        if current_plan.current_day > self.max_trip_duration:
            return False

        path_found_in_this_branch = False

        for next_destination in list(remaining_destinations):
            options = self._get_possible_options(current_location, current_plan.current_day, next_destination)
            options.sort(key=lambda seg: self._heuristic_score_segment(current_plan, seg, remaining_destinations))

            for segment in options:
                next_plan = current_plan.__copy__()
                next_plan.add_segment(segment)

                if not next_plan.check_constraints():
                    continue

                new_remaining = remaining_destinations - {next_destination}
                if self._find_best_plan_recursive(next_plan, next_destination, new_remaining):
                    path_found_in_this_branch = True
        
        return path_found_in_this_branch

    def plan_trip(self):
        self.best_plan = None
        self.min_cost = float('inf')

        target_destinations = set(self.destinations)
        
        for next_destination in list(target_destinations):
            options = self._get_possible_options(self.initial_start_location, self.initial_start_day, next_destination)
            options.sort(key=lambda seg: self._heuristic_score_segment(TravelPlan(self.budget, self.min_night_stays), seg, target_destinations - {next_destination}))

            for segment in options:
                temp_plan = TravelPlan(self.budget, self.min_night_stays)
                temp_plan.add_segment(segment)

                if not temp_plan.check_constraints():
                    continue

                new_remaining = target_destinations - {next_destination}
                self._find_best_plan_recursive(temp_plan, next_destination, new_remaining)
        
        return self.best_plan

def main():
    print("Welcome to the Smart Travel Planner AI!")
    print("Let's plan your multi-destination trip.")

    destinations_input = input("Enter your desired destinations, comma-separated (e.g., London, Paris, Rome): ")
    destinations = [d.strip() for d in destinations_input.split(',')]

    start_location = input("Enter your starting location (e.g., New York): ")

    start_day = 0
    try:
        budget = float(input("Enter your total travel budget: "))
    except ValueError:
        print("Invalid budget. Using a default of 2000.")
        budget = 2000.0

    min_night_stays_input = input("Enter minimum night stays for destinations (e.g., London:2, Paris:3) or leave blank: ")
    min_night_stays = {}
    if min_night_stays_input:
        try:
            for item in min_night_stays_input.split(','):
                city, nights = item.split(':')
                min_night_stays[city.strip()] = int(nights.strip())
        except ValueError:
            print("Invalid format for minimum night stays. Ignoring.")

    try:
        max_trip_duration = int(input("Enter maximum trip duration in days (e.g., 14) or leave blank for no limit): ") or "99999")
    except ValueError:
        print("Invalid trip duration. Using no limit.")
        max_trip_duration = float('inf')

    constraints = {
        "min_night_stays": min_night_stays,
        "max_trip_duration": max_trip_duration
    }

    print("\nPlanning your trip...")
    
    agent = TravelAgent(destinations, start_location, start_day, budget, constraints)

    best_plan = agent.plan_trip()

    if best_plan:
        print("\n--- Best Travel Plan Found ---")
        print(best_plan)
    else:
        print("\nCould not find a valid travel plan with the given constraints.")

if __name__ == "__main__":
    main()