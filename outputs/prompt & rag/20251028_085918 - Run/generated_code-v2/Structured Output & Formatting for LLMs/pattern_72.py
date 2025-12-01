from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import pipeline
import re

app = FastAPI()

# Load pre-trained summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load pre-trained sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

class ReviewSummary(BaseModel):
    product_id: str
    overall_sentiment: str
    pros: List[str]
    cons: List[str]
    key_features: List[str]
    summary_text: str

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

@app.post("/summarize_reviews", response_model=ReviewSummary)
async def summarize_reviews(product_id: str, reviews: List[str]):
    preprocessed_reviews = [preprocess_text(review) for review in reviews]
    combined_reviews = " ".join(preprocessed_reviews)

    # Generate summary
    summary_output = summarizer(combined_reviews, max_length=150, min_length=50, do_sample=False)
    summary_text = summary_output[0]['summary_text']

    # Perform sentiment analysis on individual reviews to extract pros/cons and overall sentiment
    pros = []
    cons = []
    sentiments = []

    for review in preprocessed_reviews:
        sentiment_result = sentiment_analyzer(review)[0]
        sentiments.append(sentiment_result['label'])
        if sentiment_result['label'] == 'POSITIVE' and sentiment_result['score'] > 0.8:
            # Simple extraction: consider the whole review or key phrases from positive reviews as pros
            pros.append(review[:100] + "...") # Take first 100 chars as a snippet
        elif sentiment_result['label'] == 'NEGATIVE' and sentiment_result['score'] > 0.8:
            # Simple extraction: consider the whole review or key phrases from negative reviews as cons
            cons.append(review[:100] + "...") # Take first 100 chars as a snippet
    
    # Determine overall sentiment
    positive_count = sentiments.count('POSITIVE')
    negative_count = sentiments.count('NEGATIVE')
    neutral_count = sentiments.count('NEUTRAL') # Sentiment models might output NEUTRAL

    if positive_count > negative_count and positive_count > neutral_count:
        overall_sentiment = "Positive"
    elif negative_count > positive_count and negative_count > neutral_count:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Mixed/Neutral"

    # For key features, a more advanced NLP technique would be needed (e.g., aspect-based sentiment analysis).
    # For this example, we'll use a simplified approach of extracting potential keywords from the summary.
    # In a real application, you might use NER or keyword extraction libraries.
    key_features = list(set(re.findall(r'\b\w{4,}\b', summary_text)))[:5] # Extract 5 longest words as features

    return ReviewSummary(
        product_id=product_id,
        overall_sentiment=overall_sentiment,
        pros=list(set(pros[:5])),
        cons=list(set(cons[:5])),
        key_features=key_features,
        summary_text=summary_text
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)