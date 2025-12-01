import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Candidate Generation Module (Simulated) ---
def generate_candidates(destination, interests):
    # This is a simulated function. In a real application, this would query a database or API
    # For demonstration, we'll return a fixed set of candidates based on simple matching.
    
    all_pois = {
        "Paris": [
            {"name": "Eiffel Tower", "type": "landmark", "tags": ["sightseeing", "iconic"]},
            {"name": "Louvre Museum", "type": "museum", "tags": ["art", "culture"]},
            {"name": "Notre Dame Cathedral", "type": "landmark", "tags": ["history", "architecture"]},
            {"name": "Montmartre", "type": "district", "tags": ["art", "bohemian", "views"]},
            {"name": "Seine River Cruise", "type": "activity", "tags": ["sightseeing", "romantic"]},
            {"name": "Palace of Versailles", "type": "history", "tags": ["day trip", "history", "gardens"]},
        ],
        "Tokyo": [
            {"name": "Shibuya Crossing", "type": "landmark", "tags": ["iconic", "urban"]},
            {"name": "Senso-ji Temple", "type": "temple", "tags": ["culture", "history", "buddhist"]},
            {"name": "Tokyo Skytree", "type": "landmark", "tags": ["views", "modern"]},
            {"name": "Ghibli Museum", "type": "museum", "tags": ["art", "animation", "family"]},
            {"name": "Akihabara", "type": "district", "tags": ["electronics", "anime", "manga"]},
            {"name": "Imperial Palace", "type": "history", "tags": ["history", "gardens"]},
        ],
        "Rome": [
            {"name": "Colosseum", "type": "landmark", "tags": ["history", "ancient"]},
            {"name": "Roman Forum", "type": "landmark", "tags": ["history", "ancient"]},
            {"name": "Vatican City", "type": "religious", "tags": ["history", "art", "culture"]},
            {"name": "Trevi Fountain", "type": "landmark", "tags": ["iconic", "romantic"]},
            {"name": "Pantheon", "type": "landmark", "tags": ["architecture", "history"]},
            {"name": "Borghese Gallery and Museum", "type": "museum", "tags": ["art", "gardens"]},
        ]
    }

    selected_pois = []
    if destination in all_pois:
        for poi in all_pois[destination]:
            # Simple matching: if any interest tag matches, add it. Or if no specific interests are given.
            if not interests or any(i.lower() in t.lower() for i in interests.split(',') for t in poi["tags"]):
                selected_pois.append(poi)
    
    # Simulate some accommodations and activities (can be more sophisticated)
    accommodations = [
        {"name": f"{destination} Grand Hotel", "type": "luxury", "price_range": "$$$"},
        {"name": f"{destination} Boutique Stay", "type": "boutique", "price_range": "$$"},
        {"name": f"{destination} Hostel", "type": "budget", "price_range": "$"},
    ]
    activities = [
        {"name": "Local Food Tour", "type": "culinary"},
        {"name": "Cooking Class", "type": "experience"},
        {"name": "Walking City Tour", "type": "sightseeing"},
    ]
    
    return {"pois": selected_pois, "accommodations": accommodations, "activities": activities}

# --- LLM Recommendation Engine ---
def get_llm_chain():
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None

    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

    # Prompt template with Chain-of-Thought (CoT) instructions and placeholders
    # Few-shot examples can be added here if desired, but for brevity, we'll focus on CoT.
    template = """
You are an expert travel agent. Your task is to create a personalized travel itinerary based on user preferences and a list of available candidates.

Here are the user's travel details:
Request: {user_request}

Here are the candidate points of interest, accommodations, and activities for the destination:
Candidates: {candidates}

Follow these steps to generate the itinerary:
1.  **Understand User Needs:** Carefully read the user's request, identify key preferences, interests, budget, and number of days.
2.  **Select Best Candidates:** From the provided candidates, choose the most relevant POIs, accommodations, and activities that align with the user's preferences. Prioritize unique experiences and good flow between locations.
3.  **Draft Daily Plan:** Create a day-by-day itinerary, allocating activities logically. Suggest timings for activities, meals, and relaxation. Ensure the plan is realistic for the number of days.
4.  **Add Practical Tips:** Include practical advice like transportation suggestions, booking recommendations, or local etiquette.
5.  **Review and Refine:** Ensure the itinerary is coherent, personalized, and addresses all aspects of the user's request. Present it in a clear, engaging, and easy-to-read format.

Now, generate the personalized travel itinerary:
"""

    prompt = PromptTemplate(template=template, input_variables=["user_request", "candidates"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain

# --- Streamlit UI ---
st.set_page_config(page_title="AI Travel Itinerary Recommender")
st.title("🌍 Personalized AI Travel Itinerary Recommender")
st.markdown("Powered by LLMs, In-context Learning, and Chain-of-Thought Reasoning")

user_destination = st.text_input("Where do you want to go? (e.g., Paris, Tokyo, Rome)", "Paris")
user_interests = st.text_area("What are your interests? (e.g., art, history, food, outdoors, shopping)", "art, history, culture")
user_budget = st.selectbox("What's your budget like?", ["Budget-friendly ($", "Mid-range ($$", "Luxury ($$$"], index=1)
user_days = st.number_input("How many days will you be traveling?", min_value=1, max_value=30, value=5)
user_companions = st.text_input("Who are you traveling with? (e.g., solo, partner, family with kids)", "partner")

if st.button("Generate Itinerary"): 
    if not user_destination or not user_days:
        st.warning("Please enter a destination and number of days.")
    elif not OPENAI_API_KEY:
        st.error("OpenAI API key is not set. Please configure it in your .env file.")
    else:
        with st.spinner("Crafting your personalized itinerary..."): 
            # 1. Candidate Generation
            candidates_data = generate_candidates(user_destination, user_interests)
            candidates_str = f"POIs: {candidates_data['pois']}\nAccommodations: {candidates_data['accommodations']}\nActivities: {candidates_data['activities']}"

            # 2. Prepare user request for LLM
            user_full_request = f"I want a {user_days}-day trip to {user_destination}. My interests are {user_interests}. My budget is {user_budget}. I'm traveling with {user_companions}."

            # 3. LLM Recommendation Engine
            llm_chain = get_llm_chain()
            if llm_chain:
                try:
                    response = llm_chain.invoke({"user_request": user_full_request, "candidates": candidates_str})
                    st.subheader("✨ Your Personalized Travel Itinerary")
                    st.markdown(response['text'])
                except Exception as e:
                    st.error(f"An error occurred while generating the itinerary: {e}")
            else:
                st.error("Could not initialize the LLM chain. Please check API key.")

st.sidebar.markdown("## How it works")
st.sidebar.markdown("This app uses a Large Language Model (LLM) with **In-context Learning** and **Chain-of-Thought** reasoning to generate personalized travel itineraries. It simulates candidate generation and then prompts the LLM to act as an expert travel agent, breaking down the task into logical steps to provide a detailed plan.")
st.sidebar.markdown("**Note:** For this prototype, candidate POIs are hardcoded for Paris, Tokyo, and Rome. A real application would integrate with external travel APIs.")
