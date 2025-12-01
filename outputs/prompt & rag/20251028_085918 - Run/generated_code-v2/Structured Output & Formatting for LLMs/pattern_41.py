import json

def parse_travel_plan_to_json(natural_language_plan: str) -> str:
    structured_plan = {
        "title": "Generated Travel Plan",
        "days": []
    }
    
    current_day = None
    lines = natural_language_plan.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Day "):
            if current_day is not None:
                structured_plan["days"].append(current_day)
            
            day_number = line.split(" ")[1]
            current_day = {
                "day_number": int(day_number),
                "accommodation": "N/A",
                "activities": [],
                "meals": [],
                "transportation": "N/A"
            }
        elif current_day is not None:
            if line.startswith("Accommodation:"):
                current_day["accommodation"] = line.replace("Accommodation:", "").strip()
            elif line.startswith("Activities:"): # Assuming activities are comma-separated on one line
                activities_str = line.replace("Activities:", "").strip()
                current_day["activities"] = [act.strip() for act in activities_str.split(',') if act.strip()]
            elif line.startswith("Meals:"): # Assuming meals are comma-separated on one line
                meals_str = line.replace("Meals:", "").strip()
                current_day["meals"] = [meal.strip() for meal in meals_str.split(',') if meal.strip()]
            elif line.startswith("Transportation:"):
                current_day["transportation"] = line.replace("Transportation:", "").strip()
    
    if current_day is not None:
        structured_plan["days"].append(current_day)

    return json.dumps(structured_plan, indent=4)

# Example Usage:
natural_language_plan_example = """
Day 1
Accommodation: Grand Hotel
Activities: Check-in, City tour, Dinner at local restaurant
Meals: Breakfast at hotel, Lunch at cafe, Dinner at local restaurant
Transportation: Airport taxi, Public bus

Day 2
Accommodation: Grand Hotel
Activities: Museum visit, Park stroll, Shopping
Meals: Breakfast at hotel, Lunch at street food, Dinner at fine dining
Transportation: Subway, Walking

Day 3
Accommodation: N/A
Activities: Check-out, Souvenir shopping, Departure
Meals: Breakfast at hotel
Transportation: Airport shuttle
"""

if __name__ == "__main__":
    json_output = parse_travel_plan_to_json(natural_language_plan_example)
    print(json_output)
