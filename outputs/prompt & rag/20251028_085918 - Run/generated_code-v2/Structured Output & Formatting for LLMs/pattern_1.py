import json
import re
from pydantic import BaseModel, Field
from typing import List, Optional


class Transportation(BaseModel):
    mode: str
    details: Optional[str] = None
    cost: Optional[float] = None


class Accommodation(BaseModel):
    name: str
    type: str
    check_in: str
    check_out: str
    cost_per_night: Optional[float] = None


class Activity(BaseModel):
    name: str
    time: str
    description: Optional[str] = None
    cost: Optional[float] = None


class Meal(BaseModel):
    type: str
    suggestion: str
    cost: Optional[float] = None


class DayPlan(BaseModel):
    date: str
    activities: List[Activity] = Field(default_factory=list)
    meals: List[Meal] = Field(default_factory=list)
    transportation: Optional[Transportation] = None


class TravelItinerary(BaseModel):
    destination: str
    start_date: str
    end_date: str
    budget: Optional[float] = None
    summary: Optional[str] = None
    accommodation: Optional[Accommodation] = None
    daily_plans: List[DayPlan] = Field(default_factory=list)


def generate_natural_language_itinerary(
    destination: str,
    start_date: str,
    end_date: str,
    budget: float,
    interests: List[str],
) -> str:
    return f"""
Here is a fantastic 3-day adventure to {destination} from {start_date} to {end_date}, tailored for your love of {', '.join(interests)} with a budget of ${budget:.2f}.

**Summary:** This itinerary focuses on exploring historical landmarks and local cuisine, ensuring a rich cultural experience.

**Accommodation:** Stay at The Grand Historical Hotel, a charming boutique hotel, from {start_date} to {end_date}. Check-in at 3:00 PM, Check-out at 11:00 AM. Estimated cost: $150.00 per night.

**Day 1 ({start_date}): Arrival & Ancient Wonders**
*   **Morning (9:00 AM):** Arrive at {destination} Airport. Take a taxi to The Grand Historical Hotel (approx. $30.00).
*   **Afternoon (1:00 PM):** Lunch at "Old Town Eatery" (local cuisine, est. $25.00).
*   **Afternoon (2:30 PM):** Explore the "Ancient Roman Forum" (history, cost: $20.00). Walk from the hotel.
*   **Evening (7:00 PM):** Dinner at "The Heritage Bistro" (fine dining, est. $60.00).

**Day 2 (2024-08-02): Cultural Immersion & Art**
*   **Morning (9:30 AM):** Visit the "National Art Museum" (art & culture, cost: $15.00). Use public bus (cost: $5.00).
*   **Afternoon (1:00 PM):** Lunch at "Museum Cafe" (light fare, est. $20.00).
*   **Afternoon (3:00 PM):** Wander through the "Historic District Market" (local crafts, no entry cost).
*   **Evening (7:30 PM):** Dinner at "Pasta Paradise" (Italian, est. $40.00).

**Day 3 ({end_date}): Panoramic Views & Departure**
*   **Morning (10:00 AM):** Ascend "Mount Observation" for panoramic city views (scenic, cost: $10.00 for cable car). Take a metro (cost: $7.00).
*   **Afternoon (12:30 PM):** Lunch at "Summit Cafe" (casual, est. $22.00).
*   **Afternoon (3:00 PM):** Return to hotel, collect luggage. Take a taxi to {destination} Airport (approx. $30.00).
*   **Evening (6:00 PM):** Depart from {destination}.

Have a wonderful trip!
"""

def parse_natural_language_itinerary(natural_language_itinerary: str) -> TravelItinerary:
    destination_match = re.search(r"adventure to (.*?)(?=\sfrom)", natural_language_itinerary)
    start_date_match = re.search(r"from (\d{4}-\d{2}-\d{2}) to", natural_language_itinerary)
    end_date_match = re.search(r"to (\d{4}-\d{2}-\d{2}),", natural_language_itinerary)
    budget_match = re.search(r"budget of \$(\d+\.\d{2})", natural_language_itinerary)
    summary_match = re.search(r"\*\*Summary:\*\*\s(.*?)\n\n", natural_language_itinerary, re.DOTALL)

    destination = destination_match.group(1) if destination_match else "Unknown"
    start_date = start_date_match.group(1) if start_date_match else "Unknown"
    end_date = end_date_match.group(1) if end_date_match else "Unknown"
    budget = float(budget_match.group(1)) if budget_match else None
    summary = summary_match.group(1).strip() if summary_match else None

    # Accommodation parsing
    accommodation_name_match = re.search(r"Accommodation: Stay at (.*?), a charming boutique hotel,", natural_language_itinerary)
    accommodation_dates_match = re.search(r"from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\. Check-in at (.*?)\, Check-out at (.*?)\.", natural_language_itinerary)
    accommodation_cost_match = re.search(r"Estimated cost: \$(\d+\.\d{2}) per night\.", natural_language_itinerary)

    accommodation = None
    if accommodation_name_match and accommodation_dates_match and accommodation_cost_match:
        accommodation = Accommodation(
            name=accommodation_name_match.group(1),
            type="boutique hotel",
            check_in=accommodation_dates_match.group(3),
            check_out=accommodation_dates_match.group(4),
            cost_per_night=float(accommodation_cost_match.group(1))
        )

    daily_plans = []
    day_sections = re.findall(r"\*\*Day (\d+) \((\d{4}-\d{2}-\d{2})\): (.*?)\*\*\n(.*?)(?=\n\*\*Day \d+ \(|Have a wonderful trip!)", natural_language_itinerary, re.DOTALL)

    for day_num, date, day_title, day_content in day_sections:
        activities = []
        meals = []
        day_transportation = None

        # Parse activities and meals
        items = re.findall(r"\*\s\*\*(.*?):\*\*\s(.*?)(?:\s\((.*?)\))?(?:,\s(?:cost: \$(\d+\.\d{2})|approx\. \$(\d+\.\d{2}))?)?\.", day_content)
        for item_time, item_name, item_desc_or_type, item_cost_str, item_cost_approx_str in items:
            cost = None
            if item_cost_str: cost = float(item_cost_str)
            elif item_cost_approx_str: cost = float(item_cost_approx_str)

            if "Lunch at" in item_name or "Dinner at" in item_name or "Breakfast at" in item_name:
                meal_type = re.search(r"(Lunch|Dinner|Breakfast)", item_name).group(1) if re.search(r"(Lunch|Dinner|Breakfast)", item_name) else "Unknown"
                meals.append(Meal(type=meal_type, suggestion=item_name, cost=cost))
            elif "Take a taxi" in item_name or "Use public bus" in item_name or "Take a metro" in item_name:
                mode = re.search(r"Take a (taxi|metro)|Use (public bus)", item_name).group(1) or re.search(r"Take a (taxi|metro)|Use (public bus)", item_name).group(2)
                day_transportation = Transportation(mode=mode, details=item_name, cost=cost)
            else:
                activities.append(Activity(name=item_name, time=item_time, description=item_desc_or_type, cost=cost))
        
        daily_plans.append(DayPlan(date=date, activities=activities, meals=meals, transportation=day_transportation))

    return TravelItinerary(
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        budget=budget,
        summary=summary,
        accommodation=accommodation,
        daily_plans=daily_plans,
    )


if __name__ == "__main__":
    # Simulated User Input
    user_destination = "Rome"
    user_start_date = "2024-08-01"
    user_end_date = "2024-08-03"
    user_budget = 1000.00
    user_interests = ["history", "art", "food"]

    # 1. Simulate LLM generating natural language itinerary
    print("\n--- Generating Natural Language Itinerary (Simulated LLM) ---")
    natural_itinerary = generate_natural_language_itinerary(
        user_destination,
        user_start_date,
        user_end_date,
        user_budget,
        user_interests,
    )
    print(natural_itinerary)

    # 2. Parse natural language itinerary into structured format
    print("\n--- Parsing Natural Language to Structured JSON ---")
    structured_itinerary = parse_natural_language_itinerary(natural_itinerary)
    
    # 3. Output structured JSON
    print(json.dumps(structured_itinerary.model_dump(), indent=2))

    # Example of accessing structured data
    print("\n--- Accessing Structured Data Example ---")
    if structured_itinerary.accommodation:
        print(f"Accommodation Name: {structured_itinerary.accommodation.name}")
    if structured_itinerary.daily_plans:
        print(f"First Day Activities: {[activity.name for activity in structured_itinerary.daily_plans[0].activities]}")
