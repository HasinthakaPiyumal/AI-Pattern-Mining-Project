from fastapi import FastAPI
from pydantic import BaseModel
import re

# 1. Data Models
class CommentPairInput(BaseModel):
    comment_a: str
    comment_b: str

class ModerationResult(BaseModel):
    comment_a_score: int
    comment_b_score: int
    reason_for_scores: str
    more_severe_comment: str  # "A", "B", or "None"

# 2. LLM Handler (Mock for demonstration)
# In a real application, this would involve calling an actual LLM API or running a local model.
def perform_pairwise_evaluation(comment_a: str, comment_b: str) -> ModerationResult:
    community_guidelines = [
        "Hate Speech: Inciting hatred or violence against groups.",
        "Harassment: Targeting individuals with abuse or threats.",
        "Graphic Content: Explicit violence or sexually explicit material.",
        "Misinformation: Spreading demonstrably false information causing harm."
    ]
    guidelines_str = "\n- ".join(community_guidelines)

    # Simulate LLM's understanding and scoring
    # This is a simplified mock. A real LLM would produce more nuanced output.
    score_a = 0
    score_b = 0
    reason = []
    more_severe = "None"

    # Very basic keyword-based simulation for demonstration
    if "kill" in comment_a.lower() or "hate" in comment_a.lower():
        score_a = max(score_a, 7)
        reason.append(f"Comment A contains potentially hateful/violent language.")
    if "threat" in comment_a.lower() or "abuse" in comment_a.lower():
        score_a = max(score_a, 5)
        reason.append(f"Comment A contains potentially abusive/threatening language.")
    if "false" in comment_a.lower() and "claim" in comment_a.lower():
        score_a = max(score_a, 4)
        reason.append(f"Comment A contains potential misinformation.")

    if "kill" in comment_b.lower() or "hate" in comment_b.lower():
        score_b = max(score_b, 7)
        reason.append(f"Comment B contains potentially hateful/violent language.")
    if "threat" in comment_b.lower() or "abuse" in comment_b.lower():
        score_b = max(score_b, 5)
        reason.append(f"Comment B contains potentially abusive/threatening language.")
    if "false" in comment_b.lower() and "claim" in comment_b.lower():
        score_b = max(score_b, 4)
        reason.append(f"Comment B contains potential misinformation.")
    
    # Determine which is more severe
    if score_a > score_b:
        more_severe = "A"
    elif score_b > score_a:
        more_severe = "B"
    else:
        more_severe = "None"
    
    if not reason:
        reason.append("No significant violations detected in either comment.")

    return ModerationResult(
        comment_a_score=score_a,
        comment_b_score=score_b,
        reason_for_scores="; ".join(reason),
        more_severe_comment=more_severe
    )

# 3. API Endpoint
app = FastAPI()

@app.post("/moderate_comments", response_model=ModerationResult)
async def moderate_comments(comments: CommentPairInput):
    result = perform_pairwise_evaluation(comments.comment_a, comments.comment_b)
    return result

# To run this application:
# 1. Save the code as main.py
# 2. Install dependencies: pip install fastapi uvicorn pydantic
# 3. Run from your terminal: uvicorn main:app --reload
# 4. Access the API at http://127.0.0.1:8000/docs for Swagger UI.