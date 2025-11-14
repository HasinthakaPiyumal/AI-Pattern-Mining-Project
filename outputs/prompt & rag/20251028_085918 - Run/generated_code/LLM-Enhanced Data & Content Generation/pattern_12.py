"""FitFlow AI: Personalized Workout and Nutrition Plan Generator"""

import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Pydantic Models ---

class UserProfile(BaseModel):
    user_id: str
    fitness_goals: List[str]
    dietary_restrictions: List[str]
    fitness_level: str  # e.g., 'beginner', 'intermediate', 'advanced'
    preferences: Optional[List[str]] = [] # e.g., 'home workouts', 'gym workouts', 'vegetarian', 'keto'

class Exercise(BaseModel):
    name: str
    sets: str
    reps: str
    instructions: str
    equipment: Optional[str] = None
    media_link: Optional[str] = None # Placeholder for enriching with actual data

class WorkoutPlan(BaseModel):
    day: str
    exercises: List[Exercise]

class Meal(BaseModel):
    name: str
    ingredients: List[str]
    instructions: str
    macros: Optional[Dict[str, str]] = None # e.g., {'calories': '500', 'protein': '30g'}

class NutritionPlan(BaseModel):
    day: str
    meals: List[Meal]

class GeneratedPlan(BaseModel):
    plan_id: str
    user_id: str
    workout_plans: List[WorkoutPlan]
    nutrition_plans: List[NutritionPlan]
    generated_at: str # In a real app, use datetime

class Feedback(BaseModel):
    feedback_id: str
    plan_id: str
    user_id: str
    rating: int # e.g., 1-5
    comments: Optional[str] = None
    timestamp: str # In a real app, use datetime

# --- In-Memory Data Storage (Placeholders for PostgreSQL) ---
# In a real application, these would interact with a database.

db_users: Dict[str, UserProfile] = {}
db_plans: Dict[str, GeneratedPlan] = {}
db_feedback: Dict[str, List[Feedback]] = {}

# --- FastAPI Application ---

app = FastAPI(
    title="FitFlow AI: Personalized Workout & Nutrition API",
    description="API for generating and managing personalized workout and nutrition plans using LLMs."
)

# --- LLM Core (Simulated Content Generation) ---
def generate_content_with_llm(
    user_profile: UserProfile,
    user_feedback: Optional[List[Feedback]] = None
) -> Dict[str, List]:
    """Simulates LLM interaction to generate workout and nutrition plans.
    In a real application, this would use LangChain to construct prompts
    and call an LLM API (e.g., OpenAI, Hugging Face).
    """

    # Example of how feedback could influence the prompt (simplified)
    feedback_summary = ""
    if user_feedback:
        positive_feedback = [f.comments for f in user_feedback if f.rating >= 4 and f.comments]
        negative_feedback = [f.comments for f in user_feedback if f.rating <= 2 and f.comments]
        if positive_feedback: feedback_summary += f"User generally liked: {', '.join(positive_feedback)}. "
        if negative_feedback: feedback_summary += f"User disliked: {', '.join(negative_feedback)}. "

    # This is where LangChain would typically be used to build a sophisticated prompt
    # Example placeholder for LLM generated content structure
    # In a real scenario, the LLM would return structured JSON or text parsed into this structure.
    workout_plan_data = [
        {
            "day": "Day 1: Full Body Strength",
            "exercises": [
                {"name": "Squats", "sets": "3", "reps": "8-12", "instructions": "Keep chest up, descend until thighs are parallel to floor.", "equipment": "Barbell"},
                {"name": "Push-ups", "sets": "3", "reps": "AMRAP", "instructions": "Keep body in a straight line from head to heels.", "equipment": "None"},
                {"name": "Dumbbell Rows", "sets": "3", "reps": "10-15 per arm", "instructions": "Pull dumbbell towards hip, squeeze shoulder blade.", "equipment": "Dumbbells"}
            ]
        },
        {
            "day": "Day 2: Cardio & Core",
            "exercises": [
                {"name": "Running", "sets": "1", "reps": "30 min", "instructions": "Maintain a steady pace.", "equipment": "Treadmill/Outdoors"},
                {"name": "Plank", "sets": "3", "reps": "60 sec hold", "instructions": "Keep body straight, engage core.", "equipment": "None"}
            ]
        }
    ]

    nutrition_plan_data = [
        {
            "day": "Day 1",
            "meals": [
                {"name": "Breakfast: Oatmeal with Berries", "ingredients": ["Oats", "Water/Milk", "Mixed Berries", "Protein Powder"], "instructions": "Cook oats, mix with berries and protein.", "macros": {"calories": "400", "protein": "25g"}},
                {"name": "Lunch: Chicken Salad", "ingredients": ["Grilled Chicken", "Mixed Greens", "Olive Oil Vinaigrette", "Avocado"], "instructions": "Combine ingredients.", "macros": {"calories": "550", "protein": "40g"}}
            ]
        },
        {
            "day": "Day 2",
            "meals": [
                {"name": "Breakfast: Scrambled Eggs with Spinach", "ingredients": ["Eggs", "Spinach", "Whole Wheat Toast"], "instructions": "Scramble eggs with spinach, serve with toast.", "macros": {"calories": "350", "protein": "20g"}}
            ]
        }
    ]

    # Simulate customization based on profile and feedback
    if "lose weight" in user_profile.fitness_goals:
        # Adjust plan data for weight loss (simplified)
        for meal_plan in nutrition_plan_data:
            for meal in meal_plan["meals"]:
                if meal["macros"] and "calories" in meal["macros"]:
                    current_cal = int(meal["macros"]["calories"].replace('cal', ''))
                    meal["macros"]["calories"] = f"{current_cal - 100}cal"
                    meal["name"] += " (lower calorie focus)"
    
    if "vegetarian" in user_profile.dietary_restrictions or "vegetarian" in user_profile.preferences:
        # Adjust plan data for vegetarian (simplified)
        for meal_plan in nutrition_plan_data:
            for meal in meal_plan["meals"]:
                if "Chicken" in meal["name"] or "Eggs" in meal["name"]:
                    meal["name"] = meal["name"].replace("Chicken", "Lentil").replace("Eggs", "Tofu Scramble")
                    meal["ingredients"] = [ing.replace("Chicken", "Lentils").replace("Eggs", "Tofu") for ing in meal["ingredients"]]
                    meal["instructions"] += " (vegetarian friendly)"

    return {
        "workout_plans": workout_plan_data,
        "nutrition_plans": nutrition_plan_data
    }

# --- API Endpoints ---

@app.post("/users/", response_model=UserProfile, summary="Create a new user profile")
async def create_user(user_profile: UserProfile):
    if user_profile.user_id in db_users:
        raise HTTPException(status_code=400, detail="User ID already exists.")
    db_users[user_profile.user_id] = user_profile
    return user_profile

@app.get("/users/{user_id}", response_model=UserProfile, summary="Retrieve a user profile")
async def get_user(user_id: str):
    if user_id not in db_users:
        raise HTTPException(status_code=404, detail="User not found.")
    return db_users[user_id]

@app.post("/generate_plan/{user_id}", response_model=GeneratedPlan, summary="Generate a personalized plan for a user")
async def generate_personalized_plan(user_id: str):
    if user_id not in db_users:
        raise HTTPException(status_code=404, detail="User not found.")
    
    user_profile = db_users[user_id]
    # Retrieve historical feedback for this user to influence new generation
    user_feedback = db_feedback.get(user_id, [])

    # Simulate LLM call
    llm_output = generate_content_with_llm(user_profile, user_feedback)
    
    workout_plans = [WorkoutPlan(**wp) for wp in llm_output["workout_plans"]]
    nutrition_plans = [NutritionPlan(**np) for np in llm_output["nutrition_plans"]]

    plan_id = str(uuid.uuid4())
    import datetime
    generated_at = datetime.datetime.now().isoformat()

    new_plan = GeneratedPlan(
        plan_id=plan_id,
        user_id=user_id,
        workout_plans=workout_plans,
        nutrition_plans=nutrition_plans,
        generated_at=generated_at
    )
    db_plans[plan_id] = new_plan
    return new_plan

@app.post("/feedback/", response_model=Feedback, summary="Submit feedback for a generated plan")
async def submit_feedback(feedback: Feedback):
    if feedback.plan_id not in db_plans:
        raise HTTPException(status_code=404, detail="Plan not found.")
    
    # Ensure the feedback user_id matches the plan's user_id for consistency
    if db_plans[feedback.plan_id].user_id != feedback.user_id:
        raise HTTPException(status_code=400, detail="User ID in feedback does not match user ID of the plan.")

    feedback_id = str(uuid.uuid4())
    import datetime
    feedback.timestamp = datetime.datetime.now().isoformat()
    feedback.feedback_id = feedback_id

    if feedback.user_id not in db_feedback:
        db_feedback[feedback.user_id] = []
    db_feedback[feedback.user_id].append(feedback)
    return feedback

@app.get("/plans/{user_id}", response_model=List[GeneratedPlan], summary="Retrieve all generated plans for a user")
async def get_user_plans(user_id: str):
    if user_id not in db_users:
        raise HTTPException(status_code=404, detail="User not found.")
    
    # Filter plans by user_id
    user_plans = [plan for plan in db_plans.values() if plan.user_id == user_id]
    return user_plans

# --- How to run the application ---
# To run this application, save it as `main.py` and execute:
# uvicorn main:app --reload
# Then open your browser to http://127.0.0.1:8000/docs for the API documentation.
