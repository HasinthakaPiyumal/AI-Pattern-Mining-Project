class Event:
    def __init__(self, name: str, date: str, budget: float, guest_count: int, requirements: dict = None):
        self.name = name
        self.date = date
        self.budget = budget
        self.guest_count = guest_count
        self.requirements = requirements if requirements is not None else {}
        self.plan = {
            "venue": None,
            "catering": None,
            "entertainment": None,
            "vendors": [],
            "total_cost": 0.0
        }

class Venue:
    def __init__(self, name: str, capacity: int, cost_per_hour: float, available_dates: list, amenities: list = None):
        self.name = name
        self.capacity = capacity
        self.cost_per_hour = cost_per_hour
        self.available_dates = available_dates
        self.amenities = amenities if amenities is not None else []

class Vendor:
    def __init__(self, name: str, service_type: str, cost: float, available_dates: list, specialties: list = None):
        self.name = name
        self.service_type = service_type
        self.cost = cost
        self.available_dates = available_dates
        self.specialties = specialties if specialties is not None else []
