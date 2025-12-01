from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os

# --- 1. Configuration and Environment Variables ---
# In a real application, load from .env using python-dotenv
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# --- 2. Pydantic Models ---
class UserPreferences(BaseModel):
    genres: List[str] = Field(default_factory=list, description="List of preferred genres (e.g., 'Action', 'Comedy')")
    moods: List[str] = Field(default_factory=list, description="List of preferred moods (e.g., 'Exciting', 'Relaxing')")
    actors: List[str] = Field(default_factory=list, description="List of preferred actors/actresses")
    keywords: List[str] = Field(default_factory=list, description="List of plot keywords or themes")
    disliked_genres: List[str] = Field(default_factory=list, description="Genres to avoid")
    recently_watched: List[str] = Field(default_factory=list, description="Titles recently watched, to avoid recommending")

class CandidateItem(BaseModel):
    id: str
    title: str
    genres: List[str]
    plot_summary: str
    actors: List[str]
    director: str
    year: int

class Recommendation(BaseModel):
    title: str = Field(description="Title of the recommended movie/TV show")
    rationale: str = Field(description="Explanation of why this item is recommended")
    predicted_rating: float = Field(None, description="Optional predicted rating by the LLM")

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]
    message: str = "Recommendations generated successfully."

# --- 3. Dummy Data for Candidate Generation ---
# In a real scenario, this would come from a database or external API
dummy_movie_data = [
    {"id": "m001", "title": "The Grand Adventure", "genres": ["Action", "Adventure"], "plot_summary": "A group of explorers embarks on a perilous journey to find a lost treasure.", "actors": ["Hero McLead", "Action Woman"], "director": "Visionary Director", "year": 2020},
    {"id": "m002", "title": "Laugh Out Loud", "genres": ["Comedy", "Romance"], "plot_summary": "Two unlikely individuals find love amidst hilarious misunderstandings.", "actors": ["Funny Guy", "Charming Lady"], "director": "Comedy King", "year": 2019},
    {"id": "m003", "title": "Sci-Fi Odyssey", "genres": ["Sci-Fi", "Drama"], "plot_summary": "Humanity's last hope rests on a space mission to a distant galaxy.", "actors": ["Space Captain", "AI Companion"], "director": "Futuristic Mind", "year": 2023},
    {"id": "m004", "title": "Mystery of the Old House", "genres": ["Mystery", "Thriller"], "plot_summary": "A detective investigates strange occurrences in an abandoned mansion.", "actors": ["Sleuth Man", "Mysterious Woman"], "director": "Suspense Master", "year": 221},
    {"id": "m005", "title": "Historical Epic", "genres": ["Drama", "History"], "plot_summary": "The true story of a pivotal moment in ancient history, full of political intrigue.", "actors": ["King Arthur", "Queen Guinevere"], "director": "Epic Storyteller", "year": 2022},
    {"id": "m006", "title": "Animated Wonders", "genres": ["Animation", "Family"], "plot_summary": "A young hero discovers a magical world and must save it from an evil sorcerer.", "actors": ["Voice Actor A", "Voice Actor B"], "director": "Cartoon Genius", "year": 2021},
    {"id": "m007", "title": "Romantic Getaway", "genres": ["Romance", "Drama"], "plot_summary": "Two strangers meet on vacation and explore a blossoming relationship.", "actors": ["Love Interest A", "Love Interest B"], "director": "Heartfelt Director", "year": 2024}
]

# --- 4. Candidate Generation Module ---
class CandidateGenerationModule:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = [CandidateItem(**item) for item in data]

    def get_candidates(self, preferences: UserPreferences, limit: int = 10) -> List[CandidateItem]:
        candidates = []
        # Simple keyword/genre filtering for demonstration
        for item in self.data:
            # Filter by disliked genres
            if any(dg.lower() in [g.lower() for g in item.genres] for dg in preferences.disliked_genres):
                continue
            # Filter out recently watched
            if item.title in preferences.recently_watched:
                continue

            # Basic match for preferred genres/keywords
            genre_match = any(pg.lower() in [g.lower() for g in item.genres] for pg in preferences.genres)
            keyword_match = any(pk.lower() in item.plot_summary.lower() for pk in preferences.keywords)
            actor_match = any(pa.lower() in [a.lower() for a in item.actors] for pa in preferences.actors)

            if genre_match or keyword_match or actor_match or not (preferences.genres or preferences.keywords or preferences.actors):
                candidates.append(item)
                if len(candidates) >= limit:
                    break
        return candidates

# --- 5. LLM Orchestration Layer ---
class LLMOrchestrationLayer:
    def __init__(self, llm_model: Any = None): # llm_model would be an actual LLM client like OpenAI()
        self.llm_model = llm_model # Placeholder for actual LLM client

    def _construct_zero_shot_prompt(self, preferences: UserPreferences, candidates: List[CandidateItem]) -> str:
        pref_str = ", ".join(preferences.genres + preferences.moods + preferences.keywords + preferences.actors)
        candidate_str = "\n".join([f"- {c.title} ({c.year}, {', '.join(c.genres)}): {c.plot_summary}" for c in candidates])

        prompt = f"""You are a movie and TV show recommendation assistant. Recommend up to 3 items based on the user's preferences.

User Preferences: {pref_str}

Candidate Items to consider:
{candidate_str}

Provide your recommendations with a brief rationale for each.
Format your response as 'Title: [Title] - Rationale: [Rationale]'.

Recommendations:
"""
        return prompt

    def _construct_cot_prompt(self, preferences: UserPreferences, candidates: List[CandidateItem]) -> str:
        pref_str = f"Genres: {', '.join(preferences.genres)}. Moods: {', '.join(preferences.moods)}. Keywords: {', '.join(preferences.keywords)}. Actors: {', '.join(preferences.actors)}. Disliked Genres: {', '.join(preferences.disliked_genres)}. Recently Watched: {', '.join(preferences.recently_watched)}."
        candidate_str = "\n".join([f"- {c.title} ({c.year}, {', '.join(c.genres)}): {c.plot_summary}" for c in candidates])

        prompt = f"""You are an intelligent movie and TV show recommendation engine. Your task is to provide personalized recommendations by following a step-by-step reasoning process.

User Profile and Preferences:
{pref_str}

Available Candidate Items:
{candidate_str}

Follow these steps for your reasoning:
Step 1: Understand User Preferences. Summarize the user's explicit and implicit preferences, including what they like and dislike, and any items to avoid.
Step 2: Evaluate Candidates. For each candidate item, assess how well it aligns with the user's preferences, considering genres, plot, actors, and avoiding disliked elements. Explain your evaluation.
Step 3: Generate Justification/Rationale. For the top 3 most suitable candidates, provide a concise reason why each is a strong recommendation based on your evaluation.
Step 4: Select Top Recommendations. List the titles of the top 3 recommended items, formatted as 'Title: [Title] - Rationale: [Rationale]'.

Begin your step-by-step reasoning and then provide the recommendations.
"""
        return prompt

    async def get_llm_recommendations(self, preferences: UserPreferences, candidates: List[CandidateItem]) -> str:
        # For demonstration, we'll use a simulated LLM response.
        # In a real application, you would integrate with an actual LLM (e.g., OpenAI, Hugging Face).
        # Example: response = await self.llm_model.Completion.acreate(prompt=prompt, ...)

        # Choose a prompting strategy (CoT for complex tasks is generally better)
        prompt = self._construct_cot_prompt(preferences, candidates)

        # Simulate LLM response based on preferences and candidates
        # This is a very simplified simulation and would be replaced by an actual LLM call
        print("\n--- LLM PROMPT ---\n", prompt)

        simulated_response = ""
        if "action" in [g.lower() for g in preferences.genres] and any("adventure" in k.lower() for k in preferences.keywords):
            simulated_response += "Title: The Grand Adventure - Rationale: This film perfectly matches your preference for action and adventure, featuring explorers and a perilous journey.\n"
        if "comedy" in [g.lower() for g in preferences.genres] or "funny" in [m.lower() for m in preferences.moods]:
            simulated_response += "Title: Laugh Out Loud - Rationale: A lighthearted comedy with a romantic twist, ideal for a relaxing and humorous experience.\n"
        if "sci-fi" in [g.lower() for g in preferences.genres] and "futuristic" in [k.lower() for k in preferences.keywords]:
            simulated_response += "Title: Sci-Fi Odyssey - Rationale: Delves into a futuristic narrative with deep themes, aligning with your interest in science fiction."
        if not simulated_response:
            simulated_response = "Title: Animated Wonders - Rationale: A family-friendly animation, good for general viewing.\nTitle: Romantic Getaway - Rationale: A pleasant drama focused on romance.\nTitle: The Grand Adventure - Rationale: A classic action adventure."

        print("\n--- SIMULATED LLM RESPONSE ---\n", simulated_response)
        return simulated_response

# --- 6. Output Parsing & Post-processing ---
def parse_llm_output(llm_output: str) -> List[Recommendation]:
    recommendations = []
    lines = llm_output.strip().split('\n')
    for line in lines:
        if line.startswith("Title:"):
            parts = line.split(" - Rationale:", 1)
            if len(parts) == 2:
                title = parts[0].replace("Title:", "").strip()
                rationale = parts[1].strip()
                recommendations.append(Recommendation(title=title, rationale=rationale))
    return recommendations

# --- 7. FastAPI Application ---
app = FastAPI(
    title="CineSense: LLM-powered Movie & TV Show Recommender",
    description="An API for personalized movie and TV show recommendations using LLMs, In-context Learning, and Chain-of-Thought reasoning."
)

# Initialize modules
candidate_generator = CandidateGenerationModule(data=dummy_movie_data)
llm_orchestrator = LLMOrchestrationLayer() # No actual LLM client passed for simulation

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(preferences: UserPreferences):
    try:
        # 1. Candidate Generation
        candidates = candidate_generator.get_candidates(preferences, limit=5)
        if not candidates:
            return RecommendationResponse(recommendations=[], message="No suitable candidates found based on initial filtering. Try broader preferences.")

        # 2. LLM Orchestration & Recommendation
        llm_raw_output = await llm_orchestrator.get_llm_recommendations(preferences, candidates)

        # 3. Output Parsing
        parsed_recommendations = parse_llm_output(llm_raw_output)

        if not parsed_recommendations:
            return RecommendationResponse(recommendations=[], message="LLM could not generate structured recommendations. Try again or refine preferences.")

        return RecommendationResponse(recommendations=parsed_recommendations)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

