import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import spacy
from transformers import pipeline
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data (if not already present)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Pydantic Models for request and response
class ReviewInput(BaseModel):
    review_text: str

class ReviewSummary(BaseModel):
    original_review: str
    sentiment: str
    sentiment_score: float
    key_aspects: List[str]
    summary: str

class SummarizerResponse(BaseModel):
    summaries: List[ReviewSummary]

# Load Models and Pipelines
# Load a smaller spaCy model for efficiency
try:
    nlp_spacy = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'... This may take a moment.")
    spacy.cli.download("en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm")

# Sentiment analysis pipeline (using a pre-trained DistilBERT model)
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Summarization pipeline (using a pre-trained DistilBART model)
summarization_pipeline = pipeline(
    "summarization", 
    model="sshleifer/distilbart-cnn-12-6"
)

stop_words = set(stopwords.words('english'))

# Helper Functions
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(filtered_tokens)

def analyze_sentiment(text: str) -> Dict[str, Any]:
    result = sentiment_pipeline(text)[0]
    return {"sentiment": result["label"], "sentiment_score": result["score"]}

def extract_aspects(text: str) -> List[str]:
    doc = nlp_spacy(text)
    aspects = []
    # Extract noun chunks as potential aspects
    for chunk in doc.noun_chunks:
        # Filter for meaningful aspects: not just stopwords, or single very short words
        clean_chunk = " ".join([token.text for token in chunk if token.text.lower() not in stop_words and len(token.text) > 1])
        if clean_chunk and clean_chunk not in aspects: # Ensure uniqueness
            aspects.append(clean_chunk)
    # Limit to top N aspects for brevity
    return aspects[:5]

def generate_summary(text: str) -> str:
    # Adjust max_length and min_length for desired summary length
    summary = summarization_pipeline(text, max_length=50, min_length=10, do_sample=False)[0]["summary_text"]
    return summary

# FastAPI Application
app = FastAPI(
    title="E-commerce Product Review Summarizer",
    description="An AI service that summarizes product reviews and provides structured output including sentiment and key aspects."
)

@app.post("/summarize_reviews", response_model=SummarizerResponse, summary="Summarize a list of product reviews")
async def summarize_product_reviews(reviews: List[ReviewInput]):
    """
    Receives a list of product reviews, processes each, and returns structured summaries.
    Each summary includes the original review, sentiment, sentiment score, key aspects, and an overall summary.
    """
    all_summaries = []
    for review_input in reviews:
        review_text = review_input.review_text
        
        # Perform analysis using the original review text for context
        sentiment_info = analyze_sentiment(review_text)
        key_aspects = extract_aspects(review_text)
        summary_text = generate_summary(review_text)

        all_summaries.append(
            ReviewSummary(
                original_review=review_text,
                sentiment=sentiment_info["sentiment"],
                sentiment_score=sentiment_info["sentiment_score"],
                key_aspects=key_aspects,
                summary=summary_text,
            )
        )
    return SummarizerResponse(summaries=all_summaries)

# To run the application:
# 1. Save this code as `review_summarizer_api.py`
# 2. Install dependencies: 
#    pip install fastapi "uvicorn[standard]" pydantic spacy transformers nltk
# 3. Run from your terminal: 
#    uvicorn review_summarizer_api:app --reload
# 4. Access the API documentation at http://127.0.0.1:8000/docs