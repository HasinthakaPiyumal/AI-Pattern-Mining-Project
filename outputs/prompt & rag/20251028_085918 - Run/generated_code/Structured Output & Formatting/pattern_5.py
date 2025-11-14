from typing import List, Literal, Optional
import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI # Or replace with your preferred LLM provider

# --- 1. Pydantic Model for Structured Output ---
class ProductReviewAnalysis(BaseModel):
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        ..., description="Overall sentiment of the product review."
    )
    features: List[str] = Field(
        ..., description="A list of key product features mentioned in the review."
    )
    severity: Optional[Literal["Low", "Medium", "High"]] = Field(
        None, description="Severity of negative feedback, applicable only if sentiment is 'Negative'."
    )
    reasoning: str = Field(
        ..., description="Brief explanation of why the sentiment and features were identified."
    )

# --- 2. LLM Setup ---
# IMPORTANT: Set your OpenAI API key as an environment variable (e.g., OPENAI_API_KEY)
# Or, if using Google Gemini, set GOOGLE_API_KEY and use ChatGoogleGenerativeAI
# For local models, you might use HuggingFaceHub or integrate a local server.
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0) # Consider a more capable model like gpt-4 for better results

# --- 3. LangChain Prompt and Parser ---
# Create an instance of the PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=ProductReviewAnalysis)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert e-commerce product review analyzer. "
            "Your task is to extract sentiment, key features, and "
            "optionally, the severity of negative feedback from customer reviews. "
            "Always output the results in a strict JSON format based on the following schema.\n{format_instructions}\n",
        ),
        ("human", "Analyze the following product review: {review_text}"),
    ]
)

# Construct the LangChain processing chain
review_analysis_chain: RunnableSequence = (
    prompt.partial(format_instructions=parser.get_format_instructions()) # Inject format instructions
    | llm
    | parser
)

# --- 4. FastAPI Application ---
app = FastAPI(
    title="E-commerce Product Review Analyzer",
    description="Analyzes product reviews to extract structured sentiment and feature data."
)

@app.post("/analyze-review", response_model=ProductReviewAnalysis)
async def analyze_product_review(review_text: str):
    """Analyzes a product review and returns structured sentiment, features, and severity."""
    try:
        # Invoke the LangChain review analysis chain
        structured_output = review_analysis_chain.invoke({"review_text": review_text})
        return structured_output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing review: {e}")

# --- Example Usage (for local testing) ---
if __name__ == "__main__":
    import uvicorn

    # Example of how to use the chain directly (without FastAPI)
    print("\n--- Direct Chain Invocation Examples ---")
    example_review_positive = "This laptop is amazing! The battery life is fantastic and the screen is super vibrant. Highly recommend!"
    example_review_negative = "Terrible product. The camera broke after a week and the customer service was unhelpful. Very disappointed."
    example_review_neutral = "The headphones are decent. Sound quality is okay, but the comfort could be better for long periods."

    print(f"\nAnalyzing (Positive): '{example_review_positive}'")
    try:
        result_positive = review_analysis_chain.invoke({"review_text": example_review_positive})
        print(json.dumps(result_positive.dict(), indent=2))
    except Exception as e:
        print(f"Error with positive review: {e}")

    print(f"\nAnalyzing (Negative): '{example_review_negative}'")
    try:
        result_negative = review_analysis_chain.invoke({"review_text": example_review_negative})
        print(json.dumps(result_negative.dict(), indent=2))
    except Exception as e:
        print(f"Error with negative review: {e}")

    print(f"\nAnalyzing (Neutral): '{example_review_neutral}'")
    try:
        result_neutral = review_analysis_chain.invoke({"review_text": example_review_neutral})
        print(json.dumps(result_neutral.dict(), indent=2))
    except Exception as e:
        print(f"Error with neutral review: {e}")


    print("\n--- Starting FastAPI Application ---")
    print("To run the FastAPI app, save this code as 'review_analyzer_app.py' and run:")
    print("uvicorn review_analyzer_app:app --reload")
    print("Then navigate to http://127.0.0.1:8000/docs for the API documentation.")
    # If you want to run uvicorn programmatically, uncomment the line below
    # uvicorn.run(app, host="0.0.0.0", port=8000)
