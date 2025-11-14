from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.openai_functions import create_structured_output_runnable
import os

# Set your OpenAI API key from environment variables
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# 1. Pydantic Models for Structured Output
class AspectSentiment(BaseModel):
    aspect: str = Field(description="The specific aspect of the product being reviewed (e.g., 'battery life', 'camera', 'comfort')")
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(description="The sentiment towards this specific aspect")

class ReviewSummary(BaseModel):
    product_id: str = Field(description="The ID of the product being reviewed")
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(description="Overall sentiment of all reviews for the product")
    key_pros: List[str] = Field(description="A list of key positive points or advantages mentioned in the reviews")
    key_cons: List[str] = Field(description="A list of key negative points or disadvantages mentioned in the reviews")
    summary: str = Field(description="A concise natural language summary of all the reviews")
    sentiment_by_aspect: List[AspectSentiment] = Field(description="A list of sentiments broken down by specific product aspects")

# 2. FastAPI Application Setup
app = FastAPI(
    title="E-commerce Product Review Summarizer",
    description="API to summarize product reviews and extract structured sentiment data using LLMs."
)

# 3. Langchain Integration
llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo-preview") # Use a powerful model for better structured output

# Prompt Template
# The prompt explicitly asks the LLM to provide output in JSON format
# matching the ReviewSummary schema. We provide instructions for each field.
review_summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at analyzing product reviews and extracting structured information. Your task is to summarize the provided customer reviews for a product, identify key pros and cons, determine overall sentiment, and break down sentiment by specific aspects. The output MUST be in the exact JSON format specified by the user schema."),
    ("human", "Product ID: {product_id}\n\nCustomer Reviews:\n{reviews}\n\nGenerate a structured JSON summary of these reviews according to the following schema:\n{schema_instructions}")
])

# Langchain structured output runnable using Pydantic
# This function automatically configures the LLM to generate output matching the Pydantic schema
# It leverages OpenAI's function calling capabilities for robust structured output.
structured_output_chain = create_structured_output_runnable(ReviewSummary, llm, review_summary_prompt)

# 4. FastAPI Endpoint
@app.post("/summarize-reviews", response_model=ReviewSummary)
async def summarize_product_reviews(
    product_id: str,
    reviews: List[str]
):
    """
    Summarize a list of customer reviews for a given product and return structured sentiment analysis.

    Args:
        product_id (str): The unique identifier for the product.
        reviews (List[str]): A list of raw customer review strings.

    Returns:
        ReviewSummary: A Pydantic model containing the structured summary and sentiment analysis.
    """
    if not reviews:
        raise HTTPException(status_code=400, detail="No reviews provided for summarization.")
    
    # Combine reviews into a single string for the LLM input
    combined_reviews = "\n---\n".join(reviews)
    
    try:
        # Invoke the Langchain structured output chain
        # The prompt template variables are passed here.
        # create_structured_output_runnable handles the schema instruction internally
        # using OpenAI function calling features.
        summary_data: ReviewSummary = await structured_output_chain.ainvoke({
            "product_id": product_id,
            "reviews": combined_reviews,
            "schema_instructions": ReviewSummary.schema_json(indent=2) # This is passed for clarity but mostly handled by create_structured_output_runnable
        })
        
        return summary_data
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error during review summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to summarize reviews: {e}")

# To run this application:
# 1. Save the code as `ecommerce_review_summarizer.py`
# 2. Install necessary libraries: `pip install fastapi uvicorn pydantic "langchain_openai>=0.0.5"`
# 3. Set your OpenAI API key as an environment variable (e.g., `export OPENAI_API_KEY="YOUR_KEY"`)
# 4. Run the Uvicorn server: `uvicorn ecommerce_review_summarizer:app --reload`
# 5. Access the API documentation at `http://127.0.0.1:8000/docs`
