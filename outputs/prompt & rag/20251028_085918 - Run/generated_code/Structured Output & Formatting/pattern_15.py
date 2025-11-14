import json
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

    def get_overall_sentiment(self, text):
        result = self.sentiment_pipeline(text)[0]
        # Map model output to common labels (POSITIVE, NEGATIVE, NEUTRAL if applicable)
        # For 'distilbert-base-uncased-finetuned-sst-2-english', it gives 'POSITIVE' or 'NEGATIVE'
        # For a more nuanced model, you might get 'NEUTRAL' too.
        label = result['label'].upper()
        # If the model only gives POSITIVE/NEGATIVE, we can't infer NEUTRAL accurately without more logic
        # For demonstration, we'll stick to what the model provides.
        return label

class AspectExtractor:
    def __init__(self):
        self.aspect_keywords = {
            "performance": ["performance", "fast", "slow", "speed", "lag", "responsive"],
            "battery life": ["battery", "charge", "life", "duration", "power"],
            "screen": ["screen", "display", "resolution", "brightness", "colors"],
            "price": ["price", "cost", "expensive", "cheap", "affordable", "value"],
            "delivery": ["delivery", "shipping", "shipped", "arrived"],
            "customer service": ["customer service", "support", "helped", "agent"],
            "camera": ["camera", "photos", "pictures", "video"],
            "sound": ["sound", "audio", "speakers", "headphone"],
            "design": ["design", "look", "style", "sleek", "bulky"]
        }

    def extract_aspects(self, review_text):
        extracted = []
        review_text_lower = review_text.lower()
        for aspect, keywords in self.aspect_keywords.items():
            if any(keyword in review_text_lower for keyword in keywords):
                extracted.append(aspect)
        return extracted

class ReviewProcessor:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.aspect_extractor = AspectExtractor()

    def process_review(self, review_id, review_text):
        overall_sentiment = self.sentiment_analyzer.get_overall_sentiment(review_text)
        identified_aspects = self.aspect_extractor.extract_aspects(review_text)

        aspects_with_sentiment = []
        for aspect in identified_aspects:
            # For simplicity, assign the overall sentiment to each identified aspect.
            # In a more advanced system, you'd perform fine-grained sentiment analysis per aspect.
            aspects_with_sentiment.append({"aspect": aspect, "sentiment": overall_sentiment})

        output_json = {
            "review_id": review_id,
            "overall_sentiment": overall_sentiment,
            "aspects": aspects_with_sentiment
        }
        return json.dumps(output_json, indent=4)

if __name__ == "__main__":
    processor = ReviewProcessor()

    # Sample reviews
    review1_id = "prod_123"
    review1_text = "This laptop has amazing performance and the battery life is excellent! Highly recommend it."
    
    review2_id = "prod_456"
    review2_text = "The screen is good, but the customer service was terrible. Very disappointed with the support."

    review3_id = "prod_789"
    review3_text = "It's okay for the price, nothing spectacular. The sound quality is average."

    print(f"Processing Review ID: {review1_id}")
    print(processor.process_review(review1_id, review1_text))
    print("\n" + "-" * 50 + "\n")

    print(f"Processing Review ID: {review2_id}")
    print(processor.process_review(review2_id, review2_text))
    print("\n" + "-" * 50 + "\n")

    print(f"Processing Review ID: {review3_id}")
    print(processor.process_review(review3_id, review3_text))
    print("\n" + "-" * 50 + "\n")

    # Example with a review that might be neutral or mixed
    review4_id = "prod_001"
    review4_text = "The camera takes decent pictures, but the design is a bit bulky. The price was reasonable."
    print(f"Processing Review ID: {review4_id}")
    print(processor.process_review(review4_id, review4_text))
    print("\n" + "-" * 50 + "\n")
