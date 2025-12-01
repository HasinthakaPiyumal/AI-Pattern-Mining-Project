import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException

try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

app = FastAPI()

class SummaryOutput(BaseModel):
    pros: list[str]
    cons: list[str]
    themes: list[str]

stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

summarizer_pipeline = pipeline("summarization", model="t5-small")

@app.post("/summarize_reviews", response_model=SummaryOutput)
async def summarize_reviews(reviews: list[str]):
    preprocessed_reviews = [preprocess_text(review) for review in reviews]
    combined_reviews = " ".join(preprocessed_reviews)

    prompt = f"""Summarize the following product reviews into key pros, cons, and common themes.
    Output the summary strictly in JSON format with three keys: "pros", "cons", and "themes".
    Each key should contain a list of strings.

    Reviews:
    {combined_reviews}

    JSON Summary:
    """

    summary = summarizer_pipeline(
        prompt,
        max_length=200,
        min_length=50,
        do_sample=False
    )[0]["summary_text"]

    try:
        summary_json = json.loads(summary)
        validated_summary = SummaryOutput(**summary_json)
        return validated_summary
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"JSON parsing or validation error: {e}")
        print(f"Raw summary text: {summary}")

        try:
            pros_match = re.search(r'"pros":\s*\[(.*?)\]', summary, re.DOTALL)
            cons_match = re.search(r'"cons":\s*\[(.*?)\]', summary, re.DOTALL)
            themes_match = re.search(r'"themes":\s*\[(.*?)\]', summary, re.DOTALL)

            extracted_pros = [item.strip().strip('"') for item in pros_match.group(1).split(',') if item.strip()] if pros_match else ["Could not extract pros"]
            extracted_cons = [item.strip().strip('"') for item in cons_match.group(1).split(',') if item.strip()] if cons_match else ["Could not extract cons"]
            extracted_themes = [item.strip().strip('"') for item in themes_match.group(1).split(',') if item.strip()] if themes_match else ["Could not extract themes"]

            return SummaryOutput(pros=extracted_pros, cons=extracted_cons, themes=extracted_themes)
        except Exception as fallback_e:
            raise HTTPException(status_code=500, detail=f"Failed to generate and parse structured summary: {fallback_e}. Raw model output: {summary}")
