import json
import re
from typing import List, Dict
from pydantic import BaseModel
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

class ReviewSummary(BaseModel):
    summary: str
    overall_sentiment_score: float
    pros: List[str]
    cons: List[str]

sentiment_pipeline_instance = None
summarization_pipeline_instance = None

def _load_nlp_pipelines():
    global sentiment_pipeline_instance, summarization_pipeline_instance
    if sentiment_pipeline_instance is None:
        sentiment_pipeline_instance = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    if summarization_pipeline_instance is None:
        summarization_pipeline_instance = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(texts: List[str]) -> List[Dict]:
    if not texts:
        return []

    _load_nlp_pipelines()
    sentiments = sentiment_pipeline_instance(texts)

    sentiment_details = []
    for i, text_item in enumerate(texts):
        label = sentiments[i]['label']
        score = sentiments[i]['score']
        numeric_score = 0
        if label == 'POSITIVE':
            numeric_score = score
        elif label == 'NEGATIVE':
            numeric_score = -score
        sentiment_details.append({"text": text_item, "label": label, "score": score, "numeric_score": numeric_score})

    return sentiment_details

def calculate_overall_sentiment_score(sentiment_details: List[Dict]) -> float:
    if not sentiment_details:
        return 0.0
    total_score = sum(d['numeric_score'] for d in sentiment_details)
    return total_score / len(sentiment_details)

def generate_summary(reviews: List[str]) -> str:
    if not reviews:
        return "No reviews to summarize."

    _load_nlp_pipelines()
    combined_reviews = " ".join(reviews)
    try:
        summary_result = summarization_pipeline_instance(combined_reviews, max_length=150, min_length=50, do_sample=False)
        return summary_result[0]['summary_text']
    except Exception as e:
        return (combined_reviews[:500] + "...") if len(combined_reviews) > 500 else combined_reviews

def extract_key_points(original_reviews: List[str], threshold: float = 0.9, max_points_per_category: int = 3) -> (List[str], List[str]):
    all_sentences = []
    for review in original_reviews:
        sentences = sent_tokenize(review)
        all_sentences.extend(sentences)

    all_sentences = [s.strip() for s in all_sentences if len(s.strip()) > 10]

    if not all_sentences:
        return [], []

    sentence_sentiment_details = analyze_sentiment(all_sentences)

    pros_candidates = []
    cons_candidates = []

    for detail in sentence_sentiment_details:
        sentence_text = detail["text"]
        label = detail["label"]
        score = detail["score"]

        if score > threshold:
            if label == 'POSITIVE':
                pros_candidates.append({"text": sentence_text, "score": score})
            elif label == 'NEGATIVE':
                cons_candidates.append({"text": sentence_text, "score": score})

    sorted_pros = sorted(pros_candidates, key=lambda x: x['score'], reverse=True)
    sorted_cons = sorted(cons_candidates, key=lambda x: x['score'], reverse=True)

    final_pros = list(set([p['text'] for p in sorted_pros]))[:max_points_per_category]
    final_cons = list(set([c['text'] for c in sorted_cons]))[:max_points_per_category]

    return final_pros, final_cons

def summarize_product_reviews(reviews: List[str]) -> str:
    preprocessed_reviews_for_summary_and_overall_sentiment = [preprocess_text(review) for review in reviews]
    non_empty_preprocessed_reviews = [r for r in preprocessed_reviews_for_summary_and_overall_sentiment if r]

    overall_sentiment_details = analyze_sentiment(non_empty_preprocessed_reviews)
    overall_sentiment_score = calculate_overall_sentiment_score(overall_sentiment_details)

    summary_text = generate_summary(non_empty_preprocessed_reviews)

    pros, cons = extract_key_points(reviews, threshold=0.9, max_points_per_category=3)

    review_summary = ReviewSummary(
        summary=summary_text,
        overall_sentiment_score=round(overall_sentiment_score, 4),
        pros=pros,
        cons=cons
    )

    return review_summary.json(indent=2)

