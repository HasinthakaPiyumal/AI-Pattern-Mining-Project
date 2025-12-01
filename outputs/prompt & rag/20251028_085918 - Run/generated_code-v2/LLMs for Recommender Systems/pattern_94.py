import json
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# --- Pydantic Models ---
class RecommendationRequest(BaseModel):
    user_id: str
    task_type: str  # e.g., "rating_prediction", "ranking_prediction"
    user_history: List[str]  # e.g., list of previously interacted items or keywords
    context: Optional[str] = None  # Additional context for recommendation
    num_recommendations: int = 5

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[str]
    explanation: str

# --- In-memory Data Store (for demonstration) ---
# A simple catalog of items
ITEM_CATALOG = [
    "The Hitchhiker's Guide to the Galaxy (Book)",
    "Dune (Book)",
    "Foundation (Book)",
    "The Lord of the Rings (Book)",
    "Blade Runner 2049 (Movie)",
    "Interstellar (Movie)",
    "Arrival (Movie)",
    "Inception (Movie)",
    "Cyberpunk 2077 (Game)",
    "The Witcher 3 (Game)",
    "Red Dead Redemption 2 (Game)",
    "Stardew Valley (Game)"
]

# --- Candidate Generation Module (Simplified In-memory) ---
def generate_candidates(user_history: List[str], context: Optional[str], num_candidates: int = 10) -> List[str]:
    """Simulates candidate generation based on simple keyword matching from context/history."""
    candidates = []
    if context:
        context_lower = context.lower()
        if "book" in context_lower:
            candidates.extend([item for item in ITEM_CATALOG if "book" in item.lower()])
        if "movie" in context_lower:
            candidates.extend([item for item in ITEM_CATALOG if "movie" in item.lower()])
        if "game" in context_lower:
            candidates.extend([item for item in ITEM_CATALOG if "game" in item.lower()])
    
    # Add some random items if not enough candidates or no specific context match
    if not candidates or len(candidates) < num_candidates:
        remaining_needed = num_candidates - len(candidates)
        import random
        random_items = random.sample(ITEM_CATALOG, k=min(remaining_needed, len(ITEM_CATALOG)))
        candidates.extend([item for item in random_items if item not in candidates])
    
    return candidates[:num_candidates]

# --- Prompt Engineering Module (Simplified Langchain simulation) ---
class PromptEngineer:
    def _construct_few_shot_examples(self, task_type: str) -> str:
        """Generates a few-shot example string for the prompt."""
        if task_type == "rating_prediction":
            return (
                "Example:\n" +
                "User History: ['The Martian']\n" +
                "Context: predict a rating for 'Project Hail Mary' from 1 to 5\n" +
                "Rating: 5\n"
            )
        elif task_type == "ranking_prediction":
            return (
                "Example:\n" +
                "User History: ['Dune']\n" +
                "Context: user likes sci-fi books. Rank these: ['Foundation', 'Neuromancer', 'The Lord of the Rings']\n" +
                "Reasoning: The user likes epic sci-fi. 'Foundation' is a classic epic sci-fi. 'Neuromancer' is cyberpunk sci-fi, less epic. 'The Lord of the Rings' is fantasy, not sci-fi.\n" +
                "Ranked List: ['Foundation', 'Neuromancer', 'The Lord of the Rings']\n"
            )
        return ""

    def _construct_cot_instructions(self, task_type: str) -> str:
        """Generates Chain-of-Thought instructions."""
        if task_type == "ranking_prediction":
            return (
                "Think step-by-step. First, analyze the user's history and stated preferences. " +
                "Second, evaluate each candidate item in the provided list against these preferences. " +
                "Third, explain your reasoning for the ranking, then provide the final ranked list.\n"
            )
        elif task_type == "rating_prediction":
            return (
                "Think step-by-step. First, analyze the user's history and the item to be rated. " +
                "Second, determine the most appropriate rating from 1 to 5. " +
                "Third, explain your reasoning for the rating, then provide the final rating.\n"
            )
        return ""

    def create_prompt(self, task_type: str, user_history: List[str], candidates: Optional[List[str]], context: Optional[str]) -> str:
        """Constructs a dynamic prompt for the LLM."""
        few_shot_examples = self._construct_few_shot_examples(task_type)
        cot_instructions = self._construct_cot_instructions(task_type)

        prompt_parts = [
            f"You are a recommendation system expert. Perform the following {task_type.replace('_', ' ')} task.\n",
            few_shot_examples,
            cot_instructions,
            f"User History: {user_history}\n"
        ]

        if candidates:
            prompt_parts.append(f"Candidates for ranking: {candidates}\n")
        
        if context:
            prompt_parts.append(f"Context/Instruction: {context}\n")

        if task_type == "rating_prediction":
            prompt_parts.append("Please provide your reasoning and the final rating (1-5).\n")
        elif task_type == "ranking_prediction":
            prompt_parts.append("Please provide your reasoning and the final ranked list of items.\n")
        
        return "\n".join(prompt_parts)

# --- LLM Interaction Layer (Simplified OpenAI simulation) ---
class LLMInteraction:
    def get_llm_response(self, prompt: str) -> str:
        """Simulates calling an LLM and returning a natural language response."
        In a real application, this would use the openai library:
        from openai import OpenAI
        client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
        """
        # Dummy responses for demonstration
        if "ranking_prediction" in prompt.lower() and "sci-fi" in prompt.lower():
            return (
                "Reasoning: The user history includes 'Dune', indicating a preference for epic sci-fi. " +
                "Among the candidates, 'Foundation' is a foundational epic sci-fi series. 'The Hitchhiker's Guide to the Galaxy' is sci-fi comedy, which might appeal but less directly to epic tastes. 'Blade Runner 2049' is a critically acclaimed sci-fi movie, but the user history points to books primarily.\n" +
                "Ranked List: ['Foundation (Book)', 'The Hitchhiker's Guide to the Galaxy (Book)', 'Blade Runner 2049 (Movie)']"
            )
        elif "rating_prediction" in prompt.lower() and "project hail mary" in prompt.lower():
             return (
                "Reasoning: The user liked 'The Martian', which is by the same author and shares a similar tone and scientific accuracy with 'Project Hail Mary'. Both are highly acclaimed sci-fi novels. " +
                "Given the strong similarity and positive prior experience, a high rating is warranted.\n" +
                "Rating: 5"
             )
        else:
            return (
                "Reasoning: Based on the provided history and task, I recommend the following.\n" +
                "Recommendations: ['Recommended Item A', 'Recommended Item B']"
            )

# --- Recommendation Output Processor ---
def parse_llm_response(llm_response: str, task_type: str) -> tuple[List[str], str]:
    """Parses the LLM's natural language response to extract recommendations and explanation."""
    explanation_start_tag = "Reasoning:"
    recommendations_tag_rating = "Rating:"
    recommendations_tag_ranking = "Ranked List:"

    explanation = "No explanation found."
    recommendations = []

    # Extract explanation
    if explanation_start_tag in llm_response:
        explanation_parts = llm_response.split(explanation_start_tag, 1)
        if len(explanation_parts) > 1:
            temp_explanation = explanation_parts[1].strip()
            # Try to cut the explanation before the next key tag if present
            if recommendations_tag_rating in temp_explanation:
                explanation = temp_explanation.split(recommendations_tag_rating, 1)[0].strip()
            elif recommendations_tag_ranking in temp_explanation:
                explanation = temp_explanation.split(recommendations_tag_ranking, 1)[0].strip()
            else:
                explanation = temp_explanation.strip()

    # Extract recommendations/rating
    if task_type == "rating_prediction" and recommendations_tag_rating in llm_response:
        rating_str = llm_response.split(recommendations_tag_rating, 1)[1].strip()
        recommendations = [f"Rating: {rating_str}"] # Treat rating as a single recommendation item
    elif task_type == "ranking_prediction" and recommendations_tag_ranking in llm_response:
        ranked_list_str = llm_response.split(recommendations_tag_ranking, 1)[1].strip()
        try:
            # Assuming the ranked list is a string representation of a Python list
            recommendations = json.loads(ranked_list_str.replace("'", '"'))
        except json.JSONDecodeError:
            print(f"Warning: Could not parse ranked list from LLM response: {ranked_list_str}")
            recommendations = [] # Fallback to empty list
    elif recommendations_tag_ranking in llm_response: # Generic fallback for recommendations if no task type match
        ranked_list_str = llm_response.split(recommendations_tag_ranking, 1)[1].strip()
        try:
            recommendations = json.loads(ranked_list_str.replace("'", '"'))
        except json.JSONDecodeError:
            print(f"Warning: Could not parse generic recommendations from LLM response: {ranked_list_str}")
            recommendations = []

    return recommendations, explanation

# --- FastAPI Application --- 
app = FastAPI("Prompt-based Recommendation System")

prompt_engineer = PromptEngineer()
llm_interaction = LLMInteraction()

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    candidates = None
    if request.task_type == "ranking_prediction":
        candidates = generate_candidates(request.user_history, request.context, request.num_recommendations * 2) # Generate more candidates than needed

    prompt = prompt_engineer.create_prompt(request.task_type, request.user_history, candidates, request.context)
    llm_response_content = llm_interaction.get_llm_response(prompt)
    
    recommendations, explanation = parse_llm_response(llm_response_content, request.task_type)

    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations[:request.num_recommendations], # Ensure correct number of recommendations
        explanation=explanation
    )

# To run this application:
# 1. Save the code as `main.py`
# 2. Run `pip install fastapi uvicorn pydantic`
# 3. Run `uvicorn main:app --reload`
# 4. Access the API at http://127.0.0.1:8000/docs