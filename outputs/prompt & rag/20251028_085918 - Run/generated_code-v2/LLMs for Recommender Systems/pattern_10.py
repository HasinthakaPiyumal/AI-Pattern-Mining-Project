# melody_mind_app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

from llm_handler import LLMHandler
from candidate_generation import CandidateGenerator

app = FastAPI(
    title="MelodyMind AI Music Recommender",
    description="An intuitive music curator leveraging LLMs for direct, personalized recommendations."
)

# Initialize the LLMHandler and CandidateGenerator
# In a real application, you might load configurations or models here.
llm_handler = LLMHandler()
candidate_generator = CandidateGenerator()

class RecommendationRequest(BaseModel):
    user_query: str
    strategy: str = "zero_shot"  # Can be "zero_shot", "few_shot", "cot"
    # For 'few_shot' strategy, examples could be provided, e.g., List[Dict] with 'input' and 'output'
    # For 'cot' strategy, the LLM handler will internally manage the multi-step reasoning

class Song(BaseModel):
    title: str
    artist: str
    genre: str
    mood: str

@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "Welcome to MelodyMind AI! Use the /recommend endpoint to get music suggestions."}

@app.post("/recommend", response_model=List[Song])
async def get_music_recommendations(request: RecommendationRequest):
    try:
        # Step 1: Candidate Generation
        # Generate an initial pool of relevant songs based on the user's query.
        # This prevents the LLM from having to 'recall' items from a vast catalog.
        initial_candidates = candidate_generator.generate_candidates(request.user_query)
        
        if not initial_candidates:
            # If no candidates are found by the initial filter, we can still pass the query
            # to the LLM to see if it can generate something creative, or return an empty list.
            # For simplicity, we'll let the LLM try with an empty list of candidates if none found
            # or handle a case where the LLM might struggle without relevant candidates.
            print(f"Warning: No initial candidates found for query: {request.user_query}. Proceeding with LLM.")

        # Step 2: LLM-powered Recommendation and Re-ranking
        # The LLM receives the user query and the candidates to apply ICL/CoT for personalized recommendations.
        recommendations = llm_handler.get_recommendations(
            user_query=request.user_query,
            candidates=initial_candidates,
            strategy=request.strategy
        )

        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found for your query.")

        return [Song(**song) for song in recommendations]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# To run this application:
# 1. Make sure you have uvicorn installed: pip install uvicorn
# 2. Save this file as melody_mind_app.py
# 3. Save llm_handler.py, candidate_generation.py, and data_simulator.py in the same directory.
# 4. Set your OpenAI API key as an environment variable (OPENAI_API_KEY).
# 5. Run from your terminal: uvicorn melody_mind_app:app --reload
# 6. Access the API at http://127.0.0.1:8000/docs for interactive documentation.