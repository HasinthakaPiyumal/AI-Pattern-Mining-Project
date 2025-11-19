from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json

### filename: llm_summarizer.py

class ReviewSummary(BaseModel):
    overall_sentiment: str
    common_pros: List[str]
    common_cons: List[str]
    keywords: List[str]

class LLMSummarizer:
    def summarize_reviews(self, reviews: List[str]) -> ReviewSummary:
        # Construct a detailed prompt for the LLM.
        # This prompt explicitly instructs the LLM to output in JSON format
        # adhering to the ReviewSummary schema.
        prompt = f"""Analyze the following customer reviews and provide a structured summary in JSON format. The JSON should strictly follow this schema:
{{ "overall_sentiment": "[positive/negative/neutral]", "common_pros": ["pro1", "pro2"], "common_cons": ["con1", "con2"], "keywords": ["keyword1", "keyword2"] }}

Customer Reviews:
""" + "\n".join([f"- {review}" for review in reviews]) + """

Provide only the JSON output without any additional text or formatting.
"""

        # Simulate an LLM call. In a real application, this would involve
        # calling an actual LLM API (e.g., OpenAI, Ollama, Hugging Face).
        # For this example, we'll return a hardcoded, valid JSON string.
        simulated_llm_output = {
            "overall_sentiment": "positive",
            "common_pros": ["great battery life", "excellent camera", "fast performance"],
            "common_cons": ["a bit pricey", "gets warm sometimes"],
            "keywords": ["battery", "camera", "performance", "price", "warm"]
        }
        
        # In a real scenario, you'd parse the LLM's raw string output:
        # raw_llm_response_string = self._call_llm(prompt)
        # parsed_output = json.loads(raw_llm_response_string)
        
        # For simulation, we directly use the pre-defined dictionary
        parsed_output = simulated_llm_output

        # Validate the parsed dictionary against the ReviewSummary Pydantic model
        return ReviewSummary(**parsed_output)

### filename: main.py

app = FastAPI()

class ReviewRequest(BaseModel):
    reviews: List[str]

# Instantiate the LLMSummarizer service
llm_summarizer_service = LLMSummarizer()

@app.post("/summarize-reviews", response_model=ReviewSummary)
async def summarize_reviews_endpoint(request: ReviewRequest):
    summary = llm_summarizer_service.summarize_reviews(request.reviews)
    return summary
