from fastapi import FastAPI
from pydantic import BaseModel
from llm_evaluator import get_sentiment_score

app = FastAPI()

class ReviewInput(BaseModel):
    review_text: str

@app.post("/score_review")
async def score_review(review_input: ReviewInput):
    score = get_sentiment_score(review_input.review_text)
    return {"review_text": review_input.review_text, "score": score}