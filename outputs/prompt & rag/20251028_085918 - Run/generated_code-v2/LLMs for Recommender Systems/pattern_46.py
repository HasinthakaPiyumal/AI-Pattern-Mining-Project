import streamlit as st
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import random

# --- In-memory Data Store (Simulated) ---
ALL_ITEMS = [
    {'id': 'item_1', 'name': 'Eiffel Tower', 'category': 'landmark', 'description': 'Iconic iron lattice tower in Paris.'},
    {'id': 'item_2', 'name': 'Louvre Museum', 'category': 'museum', 'description': 'World-famous art museum in Paris, home to the Mona Lisa.'},
    {'id': 'item_3', 'name': 'Notre Dame Cathedral', 'category': 'landmark', 'description': 'Historic Catholic cathedral in Paris.'},
    {'id': 'item_4', 'name': 'Dune (Book)', 'category': 'book', 'description': 'Classic science fiction novel by Frank Herbert.'},
    {'id': 'item_5', 'name': 'Project Hail Mary (Book)', 'category': 'book', 'description': 'A science fiction novel by Andy Weir.'},
    {'id': 'item_6', 'name': 'Sainte-Chapelle', 'category': 'landmark', 'description': 'Royal chapel in the Gothic style in Paris.'},
    {'id': 'item_7', 'name': 'The Great Gatsby (Book)', 'category': 'book', 'description': 'A 1925 novel by F. Scott Fitzgerald.'},
    {'id': 'item_8', 'name': 'Where the Crawdads Sing (Book)', 'category': 'book', 'description': 'A 2018 novel by Delia Owens.'},
    {'id': 'item_9', 'name': 'The Matrix (Movie)', 'category': 'sci-fi action', 'description': 'A computer hacker learns about his reality.'},
    {'id': 'item_10', 'name': 'Blade Runner (Movie)', 'category': 'sci-fi neo-noir', 'description': 'A blade runner pursues replicants.'},
    {'id': 'item_11', 'name': 'Inception (Movie)', 'category': 'sci-fi action', 'description': 'A thief who steals information via dreams.'},
    {'id': 'item_12', 'name': 'Dune (Movie 2021)', 'category': 'sci-fi adventure', 'description': 'Film adaptation of Frank Herbert\'s novel.'},
    {'id': 'item_13', 'name': 'Mad Max: Fury Road (Movie)', 'category': 'action sci-fi', 'description': 'Post-apocalyptic action film.'},
    {'id': 'item_14', 'name': 'Pride and Prejudice (Movie)', 'category': 'period drama', 'description': 'Classic romantic drama.'},
    {'id': 'item_15', 'name': 'Lord of the Rings (Book)', 'category': 'fantasy', 'description': 'A young hobbit inherits a powerful ring.'},
    {'id': 'item_16', 'name': 'Harry Potter and the Sorcerer\'s Stone (Book)', 'category': 'fantasy', 'description': 'A young orphan discovers he is a wizard.'},
    {'id': 'item_17', 'name': 'Interstellar (Movie)', 'category': 'sci-fi', 'description': 'Explorers travel through a wormhole in space.'},
    {'id': 'item_18', 'name': 'Game of Thrones (TV Series)', 'category': 'fantasy drama', 'description': 'Noble families fight for control of Westeros.'},
]

# --- Mock LLM Call Function ---
def _mock_llm_call(prompt: str) -> str:
    """
    Simulates an LLM API call based on prompt content.
    In a real application, this would use an actual LLM client (e.g., openai.ChatCompletion.create).
    """
    if "zero-shot" in prompt.lower() and "paris" in prompt.lower() and "history" in prompt.lower():
        return "Recommended items: Louvre Museum, Notre Dame Cathedral, Sainte-Chapelle"
    elif "zero-shot" in prompt.lower() and "science fiction" in prompt.lower():
        return "Recommended items: Dune (Book), Project Hail Mary (Book)"
    elif "watched 'The Matrix' and 'Blade Runner'" in prompt and "action" in prompt.lower():
        return (
            "Thought Process:\n" +
            "1. User previously watched sci-fi action movies.\n" +
            "2. Current interest is action.\n" +
            "3. Recommend similar action-packed sci-fi.\n" +
            "Recommendation: Inception (Movie), Mad Max: Fury Road (Movie), Dune (Movie 2021)"
        )
    elif "watched 'Pride and Prejudice'" in prompt and "drama" in prompt.lower():
        return (
            "Thought Process:\n" +
            "1. User enjoys classic period dramas.\n" +
            "2. Current interest is drama.\n" +
            "3. Recommend more character-driven historical dramas.\n" +
            "Recommendation: Little Women (Movie), Atonement (Movie), Emma (Movie 2020)"
        )
    elif "rerank" in prompt.lower() and "action" in prompt.lower() and "sci-fi" in prompt.lower():
        # Simple reranking logic: prioritize sci-fi/action items
        candidates_str = prompt.split("Candidate Items: [")[1].split("]")[0]
        candidate_names = [item.strip().strip("'") for item in candidates_str.split(",")]
        
        ranked = []
        for item_name in candidate_names:
            if "matrix" in item_name.lower() or "inception" in item_name.lower() or "blade runner" in item_name.lower() or "dune (movie" in item_name.lower():
                ranked.insert(0, item_name) # High priority
            elif "interstellar" in item_name.lower() or "arrival" in item_name.lower():
                ranked.insert(1, item_name) # Medium priority
            else:
                ranked.append(item_name)
        return f"Reranked items: {', '.join(ranked)}"
    elif "rerank" in prompt.lower() and "fantasy" in prompt.lower():
        candidates_str = prompt.split("Candidate Items: [")[1].split("]")[0]
        candidate_names = [item.strip().strip("'") for item in candidates_str.split(",")]
        
        ranked = []
        for item_name in candidate_names:
            if "lord of the rings" in item_name.lower() or "harry potter" in item_name.lower() or "game of thrones" in item_name.lower():
                ranked.insert(0, item_name)
            else:
                ranked.append(item_name)
        return f"Reranked items: {', '.join(ranked)}"
    return "Recommended items: No specific recommendations based on input."

# --- Prompt Engineering Functions ---
def _create_zero_shot_prompt(user_interests: list[str], available_items: list[dict]) -> str:
    user_interests_str = ", ".join(user_interests)
    item_list_str = "\n".join([f"- {item['name']}: {item['description']}" for item in available_items])

    return f"""You are an expert recommender system. This is a zero-shot recommendation task.
Based on the user's interests, provide the top 3 recommendations from the list of available items.
Do not ask for more information. Just provide the recommendations.

User Interests: {user_interests_str}

Available Items:
{item_list_str}

Recommended items:"""

def _create_cot_prompt(user_id: str, user_history: list[str], current_interest: str, available_items: list[dict]) -> str:
    history_str = ", ".join(user_history)
    item_list_str = "\n".join([f"- {item['name']} ({item['category']}): {item['description']}" for item in available_items])

    return f"""You are an intelligent recommender system. Your task is to provide personalized recommendations.
To do this, you must first analyze the user's past interactions and current interest step-by-step.

User ID: {user_id}
User's Past Interactions: {history_str}
User's Current Interest: {current_interest}

Available Items:
{item_list_str}

Think step-by-step. First, analyze the user's history and current interest to infer their preferences.
Then, based on your analysis, provide a concise recommendation and list the recommended items.

Thought Process:
"""

def _create_rerank_prompt(user_interests: list[str], candidate_item_names: list[str]) -> str:
    user_interests_str = ", ".join(user_interests)
    candidate_list_str = ", ".join([f"'{item}'" for item in candidate_item_names])

    return f"""You are an expert at ranking items for users. This is a reranking task.
Given the user's interests and a list of candidate items, please rerank the items from most relevant to least relevant.
Provide only the comma-separated list of reranked item names.

User Interests: {user_interests_str}
Candidate Items: [{candidate_list_str}]

Reranked items:"""

# --- Candidate Generation Module ---
def _generate_candidates(user_id: str, all_items: list[dict], num_candidates: int = 7) -> list[dict]:
    # Simulate candidate generation: random selection + some popular items
    random.seed(hash(user_id) % 1000000) 
    candidates = random.sample(all_items, min(num_candidates, len(all_items)))

    # Ensure some popular items are always considered
    popular_item_ids = ['item_9', 'item_11', 'item_15'] # The Matrix, Inception, Lord of the Rings
    for item_id in popular_item_ids:
        popular_item = next((item for item in all_items if item['id'] == item_id), None)
        if popular_item and popular_item not in candidates:
            candidates.append(popular_item)
    return candidates[:num_candidates]

# --- Recommendation Functions ---
def zero_shot_recommendation(user_profile: dict) -> list[str]:
    prompt = _create_zero_shot_prompt(user_profile.get('interests', []), ALL_ITEMS)
    llm_response = _mock_llm_call(prompt)
    if "Recommended items:" in llm_response:
        recommendations_str = llm_response.split("Recommended items:")[1].strip()
        return [item.strip() for item in recommendations_str.split(',')] if recommendations_str else []
    return []

def cot_sequential_recommendation(user_id: str, user_history: list[str], current_interest: str) -> dict:
    prompt = _create_cot_prompt(user_id, user_history, current_interest, ALL_ITEMS)
    llm_response = _mock_llm_call(prompt)
    
    thought_process = ""
    recommended_items = []
    if "Thought Process:" in llm_response:
        parts = llm_response.split("Recommendation:")
        thought_process = parts[0].replace("Thought Process:\n", "").strip()
        if len(parts) > 1 and "Recommended items:" in parts[1]:
            recommendations_str = parts[1].split("Recommended items:")[1].strip()
            recommended_items = [item.strip() for item in recommendations_str.split(',')] if recommendations_str else []
    return {"recommendations": recommended_items, "thought_process": thought_process}

def rerank_with_candidate_generation(user_profile: dict) -> list[str]:
    candidate_items = _generate_candidates(user_profile['user_id'], ALL_ITEMS)
    candidate_item_names = [item['name'] for item in candidate_items]
    
    prompt = _create_rerank_prompt(user_profile.get('interests', []), candidate_item_names)
    llm_response = _mock_llm_call(prompt)
    
    if "Reranked items:" in llm_response:
        reranked_str = llm_response.split("Reranked items:")[1].strip()
        return [item.strip().strip("'") for item in reranked_str.split(',')] if reranked_str else []
    return []

# --- FastAPI Application ---
app = FastAPI(title="LLM-Powered Recommendation System")

class UserProfile(BaseModel):
    user_id: str
    interests: list[str]
    history: list[str] = [] # For sequential recommendations
    current_interest: str = "" # For sequential recommendations

class ZeroShotResponse(BaseModel):
    recommendations: list[str]

class CotResponse(BaseModel):
    recommendations: list[str]
    thought_process: str

class RerankResponse(BaseModel):
    reranked_items: list[str]

@app.post("/recommend/zero-shot", response_model=ZeroShotResponse)
async def get_zero_shot_recommendations(user_profile: UserProfile):
    recommendations = zero_shot_recommendation(user_profile.model_dump())
    return {"recommendations": recommendations}

@app.post("/recommend/cot", response_model=CotResponse)
async def get_cot_recommendations(user_profile: UserProfile):
    result = cot_sequential_recommendation(
        user_profile.user_id,
        user_profile.history,
        user_profile.current_interest
    )
    return result

@app.post("/recommend/rerank", response_model=RerankResponse)
async def get_reranked_recommendations(user_profile: UserProfile):
    reranked_items = rerank_with_candidate_generation(user_profile.model_dump())
    return {"reranked_items": reranked_items}

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="LLM Recommendation Demo")
st.title("🧠 LLM-Powered Recommendation System Demo")
st.write("This application demonstrates different LLM prompting techniques for recommendations.")

# FastAPI server URL (assuming it's running on localhost:8000)
FASTAPI_URL = "http://localhost:8000"

st.sidebar.header("User Input")
user_id = st.sidebar.text_input("User ID", "user_123")
user_interests_input = st.sidebar.text_area("Your Interests (comma-separated)", "history, travel, Paris")
user_history_input = st.sidebar.text_area("Your Recent Interactions (comma-separated, e.g., 'The Matrix (Movie)')", "The Matrix (Movie), Blade Runner (Movie)")
user_current_interest = st.sidebar.text_input("Current Specific Interest (for CoT)", "action movies")

user_interests_list = [i.strip() for i in user_interests_input.split(',') if i.strip()]
user_history_list = [h.strip() for h in user_history_input.split(',') if h.strip()]

user_data = {
    "user_id": user_id,
    "interests": user_interests_list,
    "history": user_history_list,
    "current_interest": user_current_interest
}

st.header("1. Zero-Shot Recommendation")
st.markdown("LLM generates recommendations directly from interests, without prior training.")
if st.button("Get Zero-Shot Recommendations"):    
    try:
        response = requests.post(f"{FASTAPI_URL}/recommend/zero-shot", json=user_data)
        if response.status_code == 200:
            result = response.json()
            st.success("Recommended Items:")
            for item in result.get("recommendations", []):
                st.write(f"- {item}")
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to FastAPI server at {FASTAPI_URL}. Please ensure it's running.")

st.header("2. Chain-of-Thought (CoT) Sequential Recommendation")
st.markdown("LLM reasons step-by-step based on your history and current interest.")
if st.button("Get CoT Recommendations"):    
    try:
        response = requests.post(f"{FASTAPI_URL}/recommend/cot", json=user_data)
        if response.status_code == 200:
            result = response.json()
            st.success("Recommended Items:")
            for item in result.get("recommendations", []):
                st.write(f"- {item}")
            st.info("LLM Thought Process:")
            st.code(result.get("thought_process", "No thought process provided."), language='text')
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to FastAPI server at {FASTAPI_URL}. Please ensure it's running.")

st.header("3. Candidate Generation & LLM Reranking")
st.markdown("First, candidates are generated, then LLM reranks them based on your interests.")
if st.button("Get Reranked Recommendations"):    
    try:
        response = requests.post(f"{FASTAPI_URL}/recommend/rerank", json=user_data)
        if response.status_code == 200:
            result = response.json()
            st.success("Reranked Items:")
            for item in result.get("reranked_items", []):
                st.write(f"- {item}")
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to FastAPI server at {FASTAPI_URL}. Please ensure it's running.")

st.markdown("""
--- 
To run this application:
1. Save the code as `llm_recommender_app.py`.
2. Install necessary libraries: `pip install fastapi uvicorn pydantic streamlit requests`
3. Start the FastAPI server: `uvicorn llm_recommender_app:app --reload` (in one terminal)
4. Start the Streamlit app: `streamlit run llm_recommender_app.py` (in another terminal)
""")
