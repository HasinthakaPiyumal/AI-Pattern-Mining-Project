import pandas as pd
import random
from transformers import pipeline
import time # For simulating delays

class SyntheticReviewGenerator:
    def __init__(self):
        # A simple list of sentiment types
        self.sentiments = ["positive", "negative", "neutral"]
        # Predefined templates for generating reviews
        self.positive_templates = [
            "This product is absolutely amazing! Highly recommend it.",
            "I love this product, it exceeded all my expectations.",
            "Fantastic quality and great value for money.",
            "Couldn't be happier with my purchase, truly excellent.",
            "A must-buy! It works perfectly and looks great."
        ]
        self.negative_templates = [
            "Very disappointed with this product, it broke quickly.",
            "Not worth the money, had many issues with it.",
            "Expected more, but it failed to deliver.",
            "Poor quality and frustrating to use.",
            "I regret buying this, it's not good at all."
        ]
        self.neutral_templates = [
            "It's an okay product, does what it's supposed to.",
            "Nothing special, but it gets the job done.",
            "The product is fine, no major complaints or praises.",
            "Average quality, not bad but not great either.",
            "It functions as described, no surprises."
        ]

    def generate_reviews(self, product_description: str, num_reviews: int = 5) -> list:
        generated_data = []
        for _ in range(num_reviews):
            sentiment = random.choice(self.sentiments)
            review_text = ""
            if sentiment == "positive":
                review_text = random.choice(self.positive_templates)
            elif sentiment == "negative":
                review_text = random.choice(self.negative_templates)
            else: # neutral
                review_text = random.choice(self.neutral_templates)
            
            review_text = f"Regarding the {product_description}: {review_text}"
            
            generated_data.append({"review": review_text, "sentiment": sentiment})
        return generated_data

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def classify_sentiment(self, review_text: str) -> str:
        result = self.sentiment_pipeline(review_text)[0]
        label = result["label"] # e.g., "POSITIVE", "NEGATIVE"
        if label == "POSITIVE":
            return "positive"
        elif label == "NEGATIVE":
            return "negative"
        else:
            return "unknown" # sst-2 model doesn't output 'neutral'

class DataManager:
    def __init__(self):
        self.synthetic_data = pd.DataFrame(columns=["review", "sentiment", "source"])
        self.real_data = pd.DataFrame(columns=["review", "sentiment", "source"])

    def add_synthetic_data(self, data: list):
        if data:
            df = pd.DataFrame(data)
            df["source"] = "synthetic"
            self.synthetic_data = pd.concat([self.synthetic_data, df], ignore_index=True)

    def add_real_data(self, data: list):
        if data:
            df = pd.DataFrame(data)
            df["source"] = "real"
            self.real_data = pd.concat([self.real_data, df], ignore_index=True)

    def get_real_data_count(self) -> int:
        return len(self.real_data)

    def get_few_shot_examples(self, count: int = 3, from_synthetic: bool = True) -> list:
        source_data = self.synthetic_data if from_synthetic else self.real_data
        if len(source_data) < count:
            return source_data.to_dict(orient="records")
        return source_data.sample(n=count).to_dict(orient="records")

class ECommerceSentimentSystem:
    def __init__(self, product_description: str, real_data_threshold: int = 10):
        self.product_description = product_description
        self.real_data_threshold = real_data_threshold
        self.synthetic_generator = SyntheticReviewGenerator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.data_manager = DataManager()
        self.using_real_data = False

    def _prepare_few_shot_prompt(self, review: str, examples: list) -> str:
        # This function demonstrates the conceptual preparation of a few-shot prompt.
        # For a true LLM performing in-context learning, this full prompt would be sent.
        # For the pre-trained sentiment pipeline used here, only the 'review' itself is classified.
        prompt_parts = ["Analyze the sentiment of the following product review."]
        for ex in examples:
            prompt_parts.append(f"Example: Review: \"{ex['review']}\" Sentiment: {ex['sentiment']}")
        prompt_parts.append(f"Review to classify: \"{review}\" Sentiment:")
        return "\n".join(prompt_parts)

    def run_system(self, new_incoming_reviews: list):
        print(f"--- System Status for '{self.product_description}' ---")
        print(f"Current real data count: {self.data_manager.get_real_data_count()} / {self.real_data_threshold}")

        self.data_manager.add_real_data(new_incoming_reviews)
        print(f"Added {len(new_incoming_reviews)} new real reviews.")

        if self.data_manager.get_real_data_count() < self.real_data_threshold and not self.using_real_data:
            print("Insufficient real data. Using self-generated synthetic reviews for conceptual few-shot context.")
            if len(self.data_manager.synthetic_data) == 0:
                print("Generating initial batch of synthetic reviews...")
                synthetic_reviews = self.synthetic_generator.generate_reviews(self.product_description, num_reviews=10)
                self.data_manager.add_synthetic_data(synthetic_reviews)
                print(f"Generated {len(synthetic_reviews)} synthetic reviews.")
            
            few_shot_examples = self.data_manager.get_few_shot_examples(count=3, from_synthetic=True)
            print(f"Using {len(few_shot_examples)} synthetic examples as few-shot context.")

            for review_data in new_incoming_reviews:
                review_text = review_data["review"]
                # The full prompt is prepared for conceptual understanding,
                # but the sentiment_analyzer directly classifies the raw review.
                _ = self._prepare_few_shot_prompt(review_text, few_shot_examples)
                predicted_sentiment = self.sentiment_analyzer.classify_sentiment(review_text)
                print(f"Review: \"{review_text}\" | Predicted Sentiment (Early Stage with synthetic context): {predicted_sentiment}")

        else:
            if not self.using_real_data:
                print("Sufficient real data available. Transitioning to using real data for analysis.")
                self.using_real_data = True # Mark that we've transitioned

            few_shot_examples = self.data_manager.get_few_shot_examples(count=3, from_synthetic=False)
            print(f"Using {len(few_shot_examples)} real examples as few-shot context.")

            for review_data in new_incoming_reviews:
                review_text = review_data["review"]
                _ = self._prepare_few_shot_prompt(review_text, few_shot_examples)
                predicted_sentiment = self.sentiment_analyzer.classify_sentiment(review_text)
                print(f"Review: \"{review_text}\" | Predicted Sentiment (Real Data Stage with real context): {predicted_sentiment}")
        print("-" * 50)


# --- Simulation ---
if __name__ == "__main__":
    product_desc = "XYZ Smartwatch with advanced fitness tracking"
    sentiment_system = ECommerceSentimentSystem(product_desc, real_data_threshold=10)

    print("\n--- Initial Phase (Simulating New Product Launch) ---")
    
    # Week 1: A few early reviews
    week1_reviews = [
        {"review": "Just got my XYZ Smartwatch, looks sleek!", "sentiment": "neutral"},
        {"review": "Battery life is disappointing for the XYZ Smartwatch.", "sentiment": "negative"}
    ]
    sentiment_system.run_system(week1_reviews)
    time.sleep(1)

    # Week 2: More reviews
    week2_reviews = [
        {"review": "The fitness tracking on the XYZ Smartwatch is incredibly accurate.", "sentiment": "positive"},
        {"review": "Struggling with the app connection for my XYZ Smartwatch.", "sentiment": "negative"},
        {"review": "It's an okay smartwatch, nothing revolutionary.", "sentiment": "neutral"}
    ]
    sentiment_system.run_system(week2_reviews)
    time.sleep(1)

    # Week 3: Nearing threshold
    week3_reviews = [
        {"review": "Love my XYZ Smartwatch! Best purchase this year.", "sentiment": "positive"},
        {"review": "The screen scratches easily, very unhappy with my XYZ Smartwatch.", "sentiment": "negative"},
        {"review": "Good value for money, but the heart rate monitor is sometimes off.", "sentiment": "neutral"},
        {"review": "Highly recommend the XYZ Smartwatch for active people.", "sentiment": "positive"}
    ]
    sentiment_system.run_system(week3_reviews)
    time.sleep(1)

    # Week 4: Exceeding threshold, transition to real data for few-shot context
    print("\n--- Transition Phase (More Real Data Available) ---")
    week4_reviews = [
        {"review": "Finally, a smartwatch that lasts all day! XYZ Smartwatch is great.", "sentiment": "positive"},
        {"review": "Customer support for the XYZ Smartwatch is terrible.", "sentiment": "negative"},
        {"review": "It's functional, but the design is a bit bulky.", "sentiment": "neutral"}
    ]
    sentiment_system.run_system(week4_reviews)
    time.sleep(1)

    # Week 5: Continue with real data
    print("\n--- Post-Transition Phase (Relying on Real Data) ---")
    week5_reviews = [
        {"review": "The XYZ Smartwatch helped me achieve my fitness goals!", "sentiment": "positive"},
        {"review": "Software updates are too frequent and sometimes buggy.", "sentiment": "negative"},
        {"review": "Comfortable to wear, tracks steps well.", "sentiment": "positive"}
    ]
    sentiment_system.run_system(week5_reviews)