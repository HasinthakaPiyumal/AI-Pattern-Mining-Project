
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# 1. Load LLM Sentiment Predictor
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 2. Define Data Models (Pydantic)
class TicketInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    sentiment: str
    original_llm_prediction: str

# 3. Verbalizer Module
def verbalize_sentiment(llm_output: dict) -> str:
    label = llm_output.get("label", "UNKNOWN").upper()
    score = llm_output.get("score", 0.0)

    if label == "POSITIVE" and score > 0.8:
        return "Positive"
    elif label == "NEGATIVE" and score > 0.8:
        return "Negative"
    elif 0.5 < score <= 0.8 and (label == "POSITIVE" or label == "NEGATIVE"):
        return "Neutral"
    else:
        # For cases that might indicate urgency or other specific labels not directly from base LLM
        # This part would typically be more sophisticated, possibly involving keyword matching or another model layer
        if "urgent" in llm_output.get("sequence", "").lower() or \
           "critical" in llm_output.get("sequence", "").lower() or \
           "asap" in llm_output.get("sequence", "").lower():
            return "Urgent"
        return "Neutral"

# 4. API Layer (FastAPI)
@app.post("/analyze_sentiment", response_model=SentimentOutput)
async def analyze_sentiment(ticket: TicketInput):
    # LLM Sentiment Prediction
    llm_raw_prediction = sentiment_pipeline(ticket.text)[0]

    # Verbalizer Module to standardize output
    standardized_sentiment = verbalize_sentiment(llm_raw_prediction)

    return SentimentOutput(
        sentiment=standardized_sentiment,
        original_llm_prediction=llm_raw_prediction['label'] # Return the original label for transparency
    )

