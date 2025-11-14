import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Literal

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# 1. Define Pydantic models for structured output
class FeatureSentiment(BaseModel):
    name: str = Field(description="Name of the product feature or aspect")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Sentiment towards the extracted feature")
    excerpt: str = Field(description="A short quote from the review supporting the sentiment for this feature (optional, if available)", default="")

class ProductReviewAnalysis(BaseModel):
    summary: str = Field(description="A concise summary of the entire product review")
    features: List[FeatureSentiment] = Field(description="A list of identified product features with their sentiments")

# 2. Initialize the LLM
# Ensure OPENAI_API_KEY is set in your .env file or environment variables
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# 3. Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert e-commerce product review analyst. Your task is to summarize customer reviews and extract key product features along with their sentiment."),
    ("human", "Analyze the following product review and provide a summary, identify key features, and their sentiment. If possible, provide a short excerpt from the review that supports the sentiment for each feature.\n\nReview: {review}")
])

# 4. Create a Runnable sequence with the LLM and with_structured_output
# This ensures the LLM's output conforms to the ProductReviewAnalysis Pydantic model
structured_llm = llm.with_structured_output(schema=ProductReviewAnalysis)

# Chain the prompt and the structured LLM
review_analyzer_chain = prompt | structured_llm

def analyze_product_review(review_text: str) -> dict:
    """
    Analyzes a product review using an LLM to generate a summary,
    extract features, and determine sentiment in a structured JSON format.

    Args:
        review_text: The raw text of the customer review.

    Returns:
        A dictionary representing the structured JSON output.
    """
    try:
        # Invoke the chain with the review text
        analysis_result = review_analyzer_chain.invoke({"review": review_text})
        return analysis_result.model_dump() # Convert Pydantic model to a dictionary
    except Exception as e:
        print(f"An error occurred during review analysis: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Example Usage:
    sample_review_1 = (
        "I absolutely love this new smartphone! The battery life is incredible, easily lasting two days on a single charge. "
        "The camera takes stunning photos, especially in low light. However, the screen is a bit too reflective "
        "for outdoor use, which can be annoying. Customer service was excellent when I had a question about setup."
    )

    print("\n--- Analyzing Sample Review 1 ---")
    analysis_output_1 = analyze_product_review(sample_review_1)
    import json
    print(json.dumps(analysis_output_1, indent=2))

    sample_review_2 = (
        "This blender is okay. It works, but it's really noisy and not as powerful as I hoped. "
        "Cleaning is a bit of a hassle too. On the bright side, it's quite compact and fits well on my counter."
    )

    print("\n--- Analyzing Sample Review 2 ---")
    analysis_output_2 = analyze_product_review(sample_review_2)
    print(json.dumps(analysis_output_2, indent=2))

    sample_review_3 = "Mediocre product. Nothing special. Wouldn't recommend. Just a waste of money."

    print("\n--- Analyzing Sample Review 3 ---")
    analysis_output_3 = analyze_product_review(sample_review_3)
    print(json.dumps(analysis_output_3, indent=2))