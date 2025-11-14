import openai
import instructor
from pydantic import BaseModel, Field
import json
import os

# Ensure you have your OpenAI API key set as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 1. Define a Pydantic model for the structured output
class ProductReview(BaseModel):
    sentiment: str = Field(description="Overall sentiment of the review (e.g., 'positive', 'negative', 'neutral')")
    features_mentioned: list[str] = Field(description="List of specific product features mentioned in the review")
    pros: list[str] = Field(description="List of positive aspects or advantages mentioned by the reviewer")
    cons: list[str] = Field(description="List of negative aspects or disadvantages mentioned by the reviewer")
    rating_score: int = Field(description="Numerical rating given by the reviewer, if available (e.g., 1-5)", default=None)

# 2. Initialize the OpenAI client, wrapped with instructor
# This automatically enables structured output based on Pydantic models
client = instructor.patch(openai.OpenAI())

# 3. Create a function to analyze a product review
def analyze_review(review_text: str) -> ProductReview:
    """
    Analyzes a product review using an LLM to extract structured data.

    Args:
        review_text: The free-form text of the customer review.

    Returns:
        A ProductReview object containing the extracted structured data.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can use other models like "gpt-4"
            response_model=ProductReview, # This tells instructor to enforce the Pydantic schema
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert e-commerce product review analyzer. Extract key information from the provided review into a structured JSON format. Be precise and capture all relevant details for sentiment, features, pros, and cons. If a rating score is explicitly mentioned, extract it as an integer."
                },
                {"role": "user", "content": f"Analyze the following product review: \n\n\"\"\"{review_text}\"\"\""}
            ],
            max_retries=2, # instructor can retry if validation fails
        )
        return response
    except Exception as e:
        print(f"An error occurred during review analysis: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    sample_reviews = [
        "This blender is amazing! It crushes ice effortlessly and the motor is so powerful. A little noisy, but definitely worth the price. I give it a 5-star rating.",
        "The headphones have decent sound quality for the price, but the battery life is terrible, only lasting about 2 hours. Also, they feel very cheap. Not impressed, probably a 2 out of 5.",
        "I bought this smart speaker and it works as expected. Setup was a breeze, and the voice recognition is excellent. No major complaints, but the bass could be stronger. A solid 4 stars.",
        "This shirt is okay. The fabric feels nice, but it shrank a lot after the first wash. The color faded too. Disappointed, maybe a 2.5."
    ]

    print("\n--- Analyzing Sample Reviews ---")
    for i, review in enumerate(sample_reviews):
        print(f"\nReview {i+1}: {review}")
        structured_data = analyze_review(review)

        if structured_data:
            print("\nStructured Output (JSON):")
            print(json.dumps(structured_data.dict(), indent=2))
        else:
            print("Failed to generate structured data for this review.")
