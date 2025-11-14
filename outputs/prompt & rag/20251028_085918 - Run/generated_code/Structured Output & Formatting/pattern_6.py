import json
from transformers import pipeline
import spacy

# Load pre-trained sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load English tokenizer, tagger, parser, NER and word vectors
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'. This may take a moment...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def analyze_product_review(product_id: str, review_text: str) -> str:
    """
    Analyzes a product review for sentiment and extracts key features.
    Outputs the result in a structured JSON format.

    Args:
        product_id (str): The ID of the product.
        review_text (str): The raw text of the customer review.

    Returns:
        str: A JSON string containing the product ID, review text, sentiment,
             and a list of extracted keywords/features.
    """
    # 1. Sentiment Analysis
    sentiment_result = sentiment_analyzer(review_text)[0]
    sentiment_label = sentiment_result['label'].lower() # e.g., 'positive', 'negative'

    # 2. Keyword/Feature Extraction
    doc = nlp(review_text)
    # Extract noun phrases as potential keywords/features
    keywords = [chunk.text for chunk in doc.noun_chunks]

    # 3. Output Formatting
    structured_output = {
        "product_id": product_id,
        "review_text": review_text,
        "sentiment": sentiment_label,
        "extracted_keywords": keywords
    }

    return json.dumps(structured_output, indent=4)

if __name__ == "__main__":
    # Example Usage
    print("--- Example 1: Positive Review ---")
    product_id_1 = "PROD123"
    review_text_1 = "This laptop is absolutely fantastic! The battery life is amazing and the display is stunning. Highly recommend for productivity and entertainment."
    output_json_1 = analyze_product_review(product_id_1, review_text_1)
    print(output_json_1)

    print("\n--- Example 2: Negative Review ---")
    product_id_2 = "PROD456"
    review_text_2 = "The phone's camera is terrible in low light and the software is buggy. Very disappointed with my purchase, definitely not worth the price."
    output_json_2 = analyze_product_review(product_id_2, review_text_2)
    print(output_json_2)

    print("\n--- Example 3: Neutral/Mixed Review ---")
    product_id_3 = "PROD789"
    review_text_3 = "The headphones are comfortable to wear and sound quality is decent for the price. However, the noise cancellation isn't as good as advertised."
    output_json_3 = analyze_product_review(product_id_3, review_text_3)
    print(output_json_3)