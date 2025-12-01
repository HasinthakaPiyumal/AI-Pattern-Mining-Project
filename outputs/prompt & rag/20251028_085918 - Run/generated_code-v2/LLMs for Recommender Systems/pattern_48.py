class ToolManager:
    def __init__(self):
        self.available_tools = {
            "search_flights": self._search_flights,
            "search_hotels": self._search_hotels,
            "search_attractions": self._search_attractions,
            "book_flight": self._book_flight, # Simulated booking
            "book_hotel": self._book_hotel   # Simulated booking
        }

    def _search_flights(self, origin, destination, date, passengers=1):
        """Simulates searching for flights."""
        print(f"[Tool Call] Searching flights from {origin} to {destination} on {date} for {passengers} passenger(s).")
        # In a real scenario, this would call an external flight API
        if "london" in destination.lower() and "paris" in origin.lower() and date == "2024-08-15":
            return {"status": "success", "flights": [{"id": "AF123", "airline": "Air France", "price": "$150", "departure": "09:00"}]}
        return {"status": "failure", "message": "No direct flights found or details unavailable."}

    def _search_hotels(self, location, check_in_date, check_out_date, adults=1):
        """Simulates searching for hotels."""
        print(f"[Tool Call] Searching hotels in {location} from {check_in_date} to {check_out_date} for {adults} adult(s).")
        # In a real scenario, this would call an external hotel API
        if "paris" in location.lower() and check_in_date == "2024-08-15":
            return {"status": "success", "hotels": [{"name": "Hotel de Paris", "price_per_night": "$200", "rating": "4.5"}]}
        return {"status": "failure", "message": "No hotels found or details unavailable."}

    def _search_attractions(self, location, category=None):
        """Simulates searching for attractions."""
        print(f"[Tool Call] Searching attractions in {location} with category: {category or 'any'}.")
        # In a real scenario, this would call an external attractions API
        if "paris" in location.lower():
            return {"status": "success", "attractions": ["Eiffel Tower", "Louvre Museum", "Notre Dame Cathedral"]}
        elif "rome" in location.lower():
            return {"status": "success", "attractions": ["Colosseum", "Roman Forum", "Vatican City"]}
        return {"status": "failure", "message": "No popular attractions found for this location."}

    def _book_flight(self, flight_id, passenger_name):
        """Simulates booking a flight."""
        print(f"[Tool Call] Attempting to book flight {flight_id} for {passenger_name}.")
        # This would involve a complex API interaction in reality
        return {"status": "success", "booking_id": f"FLIGHT-{flight_id}-{hash(passenger_name)}"}

    def _book_hotel(self, hotel_name, check_in, check_out, guest_name):
        """Simulates booking a hotel."""
        print(f"[Tool Call] Attempting to book hotel {hotel_name} from {check_in} to {check_out} for {guest_name}.")
        # This would involve a complex API interaction in reality
        return {"status": "success", "confirmation_id": f"HOTEL-{hotel_name}-{hash(guest_name)}"}

    def call_tool(self, tool_name, **kwargs):
        """Calls a registered tool with provided arguments."""
        tool_func = self.available_tools.get(tool_name)
        if tool_func:
            return tool_func(**kwargs)
        return {"status": "error", "message": f"Tool '{tool_name}' not found."}
