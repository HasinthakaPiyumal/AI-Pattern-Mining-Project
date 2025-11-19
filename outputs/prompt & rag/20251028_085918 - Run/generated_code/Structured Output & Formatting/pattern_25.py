import os
from dotenv import load_dotenv
from typing import List, Dict

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from fastapi import FastAPI, HTTPException

# Load environment variables from .env file
load_dotenv()

# --- 1. models.py (Pydantic Schema Definition) ---

class ProductReviewSummary(BaseModel):
    product_id: str = Field(description="Identifier of the product")
    overall_sentiment: str = Field(description="Overall sentiment of the reviews (positive, neutral, or negative)")
    summary_headline: str = Field(description="Concise summary headline of the reviews")
    pros: List[str] = Field(description="List of identified pros from the reviews")
    cons: List[str] = Field(description="List of identified cons from the reviews")
    mentioned_features_sentiment: Dict[str, str] = Field(description="Dictionary mapping product features to their sentiment (positive, neutral, or negative)")

# --- 2. llm_service.py (LLM Interaction and Summarization Logic) ---

class ReviewSummarizer:
    def __init__(self, llm_model: ChatOpenAI):
        self.llm = llm_model
        self.parser = PydanticOutputParser(pydantic_object=ProductReviewSummary)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert product review analyzer. Your task is to summarize product reviews and extract key insights in a structured JSON format."),
            ("human", "Summarize the following product reviews for product ID: {product_id}.\n\nReviews: {reviews}\n\n{format_instructions}")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        self.chain = self.prompt | self.llm | self.parser

    def summarize_reviews(self, product_id: str, reviews: List[str]) -> ProductReviewSummary:
        try:
            reviews_text = "\n---\n".join(reviews)
            summary = self.chain.invoke({"product_id": product_id, "reviews": reviews_text})
            return summary
        except Exception as e:
            raise RuntimeError(f"Failed to summarize reviews: {e}")

# --- 3. main.py (FastAPI Application) ---

app = FastAPI(
    title="E-commerce Product Review Summarizer",
    description="API to summarize product reviews and output structured JSON."
)

# Initialize LLM and ReviewSummarizer
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-0125", temperature=0)
summarizer = ReviewSummarizer(llm_model=llm)


@app.post("/summarize-reviews", response_model=ProductReviewSummary)
async def summarize_product_reviews(
    product_id: str,
    reviews: List[str]
):
    try:
        summary = summarizer.summarize_reviews(product_id=product_id, reviews=reviews)
        return summary
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# Example of how to run this application:
# 1. Save this code as `main.py`
# 2. Create a `.env` file in the same directory with `OPENAI_API_KEY=your_openai_api_key_here`
# 3. Install necessary libraries: `pip install fastapi "uvicorn[standard]" pydantic langchain-openai python-dotenv`
# 4. Run the application: `uvicorn main:app --reload`
# 5. Access the API at http://127.0.0.1:8000/docs for interactive documentation.