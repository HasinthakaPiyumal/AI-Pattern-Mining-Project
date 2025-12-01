import os
import json
from typing import List, Literal
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# 1. Pydantic Model for Structured Output
class ReviewSummary(BaseModel):
    product_id: str
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"]
    key_features_mentioned: List[str]
    common_themes: List[str]
    positive_aspects: List[str]
    negative_aspects: List[str]
    summary_text: str

# 2. AI Processing Module
class ReviewSummarizer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"

    def summarize_reviews(self, product_id: str, reviews: List[str]) -> ReviewSummary:
        review_text = "\n".join([f"- {review}" for review in reviews])
        
        # Explicit instructions for desired output format in the prompt
        prompt = f"""You are an AI assistant tasked with summarizing customer reviews for an e-commerce product. 
        Your goal is to extract key information and present it in a structured JSON format.

        Product ID: {product_id}

        Customer Reviews:
        {review_text}

        Please provide a summary of these reviews, extracting the following information and ensuring the output strictly adheres to the JSON schema provided below. Do not include any other text or formatting outside the JSON object. All fields are mandatory.

        JSON Schema:
        {{
            "product_id": "<product_id>",
            "overall_sentiment": "<positive|negative|neutral|mixed>",
            "key_features_mentioned": ["<feature1>", "<feature2>"],
            "common_themes": ["<theme1>", "<theme2>"],
            "positive_aspects": ["<aspect1>", "<aspect2>"],
            "negative_aspects": ["<aspect1>", "<aspect2>"],
            "summary_text": "<a concise summary of all reviews>"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            json_output = response.choices[0].message.content
            
            # 3. Output Formatting and Validation Module
            parsed_data = json.loads(json_output)
            summary = ReviewSummary.model_validate(parsed_data)
            return summary
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from AI response: {e}\nRaw response: {json_output}")
        except ValidationError as e:
            raise ValueError(f"AI response does not match the expected schema: {e}\nRaw response: {json_output}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during AI processing: {e}")

# 4. Application Logic and Example Usage
if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your environment variables
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the script (e.g., export OPENAI_API_KEY=\'your_key_here\').")
    else:
        summarizer = ReviewSummarizer()

        # Example Product Reviews
        product_id = "PROD12345"
        customer_reviews = [
            "This smartphone has an amazing camera and the battery life is surprisingly good. Highly recommend!",
            "The screen is vibrant and the performance is smooth, but it's a bit too expensive for what it offers.",
            "I love the design and how lightweight it is. The software updates are frequent and useful.",
            "Battery drains too fast when gaming. Also, the camera is just okay, not as good as advertised. Disappointed.",
            "Great phone for daily use, excellent display. Could be cheaper though."
        ]

        print(f"\n--- Summarizing reviews for Product ID: {product_id} ---")
        try:
            summary = summarizer.summarize_reviews(product_id, customer_reviews)
            print("\n--- Structured Summary (Validated by Pydantic) ---")
            print(json.dumps(summary.model_dump(), indent=2))

            print("\n--- Accessing Specific Fields ---")
            print(f"Overall Sentiment: {summary.overall_sentiment}")
            print(f"Key Features: {', '.join(summary.key_features_mentioned)}")
            print(f"Summary: {summary.summary_text}")

        except (ValueError, RuntimeError) as e:
            print(f"Error: {e}")

        print("\n--- Another Example (Product with mostly negative reviews) ---")
        product_id_neg = "GADGET987"
        negative_reviews = [
            "This smart speaker constantly disconnects from Wi-Fi. It's incredibly frustrating to use.",
            "Sound quality is terrible, very tinny. My old speaker was much better. A complete waste of money.",
            "The voice assistant rarely understands commands. I have to repeat myself multiple times.",
            "Battery life is abysmal, dies after an hour of use. Also, the build quality feels cheap.",
            "Overpriced for its poor performance and unreliable connectivity. Do not buy."
        ]

        try:
            summary_neg = summarizer.summarize_reviews(product_id_neg, negative_reviews)
            print(f"\n--- Structured Summary for Product ID: {product_id_neg} ---")
            print(json.dumps(summary_neg.model_dump(), indent=2))

        except (ValueError, RuntimeError) as e:
            print(f"Error: {e}")