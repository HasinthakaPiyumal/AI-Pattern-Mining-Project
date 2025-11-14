from transformers import pipeline
from pydantic import BaseModel, Field
from typing import List, Literal

# Initialize sentiment analysis pipeline using a pre-trained model
# 'cardiffnlp/twitter-roberta-base-sentiment-latest' is a good general-purpose sentiment model.
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Pydantic model for Extracted Entities
class ExtractedEntities(BaseModel):
    product_features: List[str] = Field(default_factory=list, description="List of product features mentioned.")
    common_praise: List[str] = Field(default_factory=list, description="List of positive aspects/praise.")
    common_complaints: List[str] = Field(default_factory=list, description="List of negative aspects/complaints.")

# Pydantic model for the overall structured review output
class StructuredReviewOutput(BaseModel):
    review_text: str = Field(description="The original review text.")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall sentiment of the review.")
    sentiment_score: float = Field(description="Confidence score for the predicted sentiment.")
    extracted_entities: ExtractedEntities = Field(description="Key entities extracted from the review.")

def analyze_review_structured(review_text: str) -> StructuredReviewOutput:
    """
    Analyzes a product review for sentiment and extracts key entities,
    returning the results in a structured Pydantic model.
    """
    # 1. Sentiment Analysis
    sentiment_result = sentiment_pipeline(review_text)[0]
    sentiment_label = sentiment_result['label'].lower() # e.g., 'positive', 'negative', 'neutral'
    sentiment_score = round(sentiment_result['score'], 4)

    # The cardiffnlp model outputs 'LABEL_0' for negative, 'LABEL_1' for neutral, 'LABEL_2' for positive.
    # We need to map these to the Literal types defined in StructuredReviewOutput.
    if sentiment_label == 'label_0':
        mapped_sentiment = 'negative'
    elif sentiment_label == 'label_1':
        mapped_sentiment = 'neutral'
    elif sentiment_label == 'label_2':
        mapped_sentiment = 'positive'
    else:
        # Fallback for unexpected labels
        mapped_sentiment = 'neutral'

    # 2. Entity Extraction (Simplified demonstration using keyword matching)
    # In a production system, this would typically involve more advanced NLP techniques
    # like Named Entity Recognition (NER) models or rule-based systems for better accuracy.

    # Define lists of keywords related to features, praise, and complaints
    positive_keywords = ["great", "excellent", "love", "good", "perfect", "amazing", "satisfied", "recommend", "durable", "fast", "efficient", "comfortable", "sturdy", "bright", "clear", "responsive", "long-lasting"]
    negative_keywords = ["bad", "poor", "disappointed", "broken", "slow", "uncomfortable", "faulty", "issue", "problem", "expensive", "fragile", "short battery", "glitchy", "unresponsive", "cheap"]
    feature_keywords = ["battery life", "screen", "camera", "performance", "design", "ease of use", "sound quality", "material", "software", "support", "processor", "display", "speakers", "keyboard"]

    features_found = [kw for kw in feature_keywords if kw in review_text.lower()]
    praise_found = [kw for kw in positive_keywords if kw in review_text.lower()]
    complaints_found = [kw for kw in negative_keywords if kw in review_text.lower()]

    # Filter praise/complaints based on the overall sentiment to improve relevance.
    # For instance, if a review is overwhelmingly positive, minor 