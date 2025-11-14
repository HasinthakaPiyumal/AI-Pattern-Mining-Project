import json
import re

# --- 1. Mock LLM Interaction (replace with actual LLM client) ---
class MockLLM:
    def __init__(self):
        pass

    def generate_structured_output(self, prompt: str) -> str:
        # Simulate an LLM generating a JSON string based on the prompt.
        # In a real application, this would involve calling an actual LLM API
        # and carefully crafting the prompt for direct JSON output.
        if "excellent battery life" in prompt.lower() and "stunning camera" in prompt.lower():
            return json.dumps({
                "product_id": "PROD123",
                "review_id": "REV001",
                "review_text": "This phone is amazing! The excellent battery life means I don't have to charge it constantly, and the stunning camera takes incredible photos. Highly recommend!",
                "sentiment": "positive",
                "key_features_mentioned": ["battery life", "camera"],
                "pros": ["excellent battery life", "stunning camera", "takes incredible photos"],
                "cons": [],
                "rating": 5
            })
        elif "slow performance" in prompt.lower() and "short battery" in prompt.lower():
             return json.dumps({
                "product_id": "PROD456",
                "review_id": "REV002",
                "review_text": "Very disappointed. The performance is incredibly slow, and the battery life is way too short. It constantly lags.",
                "sentiment": "negative",
                "key_features_mentioned": ["performance", "battery life"],
                "pros": [],
                "cons": ["slow performance", "short battery life", "lags constantly"],
                "rating": 1
            })
        else:
            # Fallback for other reviews or for demonstrating potential issues
            # A real LLM would attempt to follow the JSON structure defined in the prompt
            return json.dumps({
                "product_id": "UNKNOWN",
                "review_id": "UNKNOWN",
                "review_text": re.search(r'Review Text: "([^"]+)"', prompt).group(1) if re.search(r'Review Text: "([^"]+)"', prompt) else "No review text found in prompt.",
                "sentiment": "neutral",
                "key_features_mentioned": [],
                "pros": [],
                "cons": [],
                "rating": 3
            })

# --- 2. Prompt Engineering Module ---
def create_sentiment_prompt(review_text: str, product_id: str, review_id: str) -> str:
    return f"""
Extract the following information from the customer review and provide it in a JSON format.
Ensure the output is a valid JSON object with the specified keys and value types.

JSON Schema:
{{
    "product_id": "<string>",
    "review_id": "<string>",
    "review_text": "<string>",
    "sentiment": "<string>" (e.g., "positive", "negative", "neutral"),
    "key_features_mentioned": "<list_of_strings>",
    "pros": "<list_of_strings>",
    "cons": "<list_of_strings>",
    "rating": "<integer>" (1-5, or inferred if not explicit)
}}

Review Text: "{review_text}"
Product ID: "{product_id}"
Review ID: "{review_id}"

JSON Output:
"""

# --- 3. Output Processing Layer ---
def process_llm_output(llm_output: str, original_review_data: dict) -> dict:
    try:
        parsed_output = json.loads(llm_output)
        # Basic validation: ensure essential keys exist
        required_keys = ["product_id", "review_id", "review_text", "sentiment"]
        if not all(key in parsed_output for key in required_keys):
            raise ValueError("Missing required keys in LLM output.")

        # Further refinement/defaults if LLM misses something (Post-processing)
        parsed_output.setdefault("key_features_mentioned", [])
        parsed_output.setdefault("pros", [])
        parsed_output.setdefault("cons", [])
        parsed_output.setdefault("rating", None) # Can be inferred or default

        # Override product_id and review_id with original data to ensure consistency
        parsed_output["product_id"] = original_review_data.get("product_id", parsed_output.get("product_id"))
        parsed_output["review_id"] = original_review_data.get("review_id", parsed_output.get("review_id"))

        return parsed_output
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM: {e}")
        print(f"Raw LLM output: {llm_output}")
        # Fallback for malformed JSON: attempt regex extraction or simpler parsing
        # For demonstration, we'll return a basic error structure.
        return {
            "product_id": original_review_data.get("product_id", "ERROR"),
            "review_id": original_review_data.get("review_id", "ERROR"),
            "review_text": original_review_data.get("review_text", "ERROR"),
            "sentiment": "unknown",
            "key_features_mentioned": [],
            "pros": [],
            "cons": ["LLM output was malformed JSON"],
            "rating": None,
            "error": str(e)
        }
    except ValueError as e:
        print(f"Validation error in LLM output: {e}")
        return {
            "product_id": original_review_data.get("product_id", "ERROR"),
            "review_id": original_review_data.get("review_id", "ERROR"),
            "review_text": original_review_data.get("review_text", "ERROR"),
            "sentiment": "unknown",
            "key_features_mentioned": [],
            "pros": [],
            "cons": ["Validation failed on LLM output"],
            "rating": None,
            "error": str(e)
        }

# --- Main Application Logic ---
def analyze_review_structured(review_data: dict) -> dict:
    product_id = review_data["product_id"]
    review_id = review_data["review_id"]
    review_text = review_data["review_text"]

    print(f"\nProcessing Review ID: {review_id}")
    print(f"Review Text: {review_text[:100]}...")

    # 1. Prompt Engineering
    prompt = create_sentiment_prompt(review_text, product_id, review_id)
    # print(f"\n--- Generated Prompt ---\n{prompt}\n------------------------") # Uncomment to see the full prompt

    # 2. LLM Interaction
    llm = MockLLM()
    raw_llm_output = llm.generate_structured_output(prompt)
    print(f"\n--- Raw LLM Output (simulated) ---\n{raw_llm_output}\n------------------------------------")

    # 3. Output Processing
    structured_output = process_llm_output(raw_llm_output, review_data)
    print("--- Structured Output (processed) ---")
    print(json.dumps(structured_output, indent=2))
    print("-------------------------------------")
    return structured_output

# --- Sample Data ---
example_reviews = [
    {
        "product_id": "PROD123",
        "review_id": "REV001",
        "review_text": "This phone is amazing! The excellent battery life means I don't have to charge it constantly, and the stunning camera takes incredible photos. Highly recommend!"
    },
    {
        "product_id": "PROD456",
        "review_id": "REV002",
        "review_text": "Very disappointed. The performance is incredibly slow, and the battery life is way too short. It constantly lags. Not worth the price at all."
    },
    {
        "product_id": "PROD789",
        "review_id": "REV003",
        "review_text": "It's an okay laptop. The screen is bright, but the keyboard feels a bit flimsy. Decent for everyday tasks but not for heavy gaming."
    }
]

# --- Run the analysis for each example review ---
if __name__ == "__main__":
    all_structured_reviews = []
    for review in example_reviews:
        result = analyze_review_structured(review)
        all_structured_reviews.append(result)

    print("\n====== All Processed Reviews ======")
    print(json.dumps(all_structured_reviews, indent=2))
