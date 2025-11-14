from llm_service import LLMReviewAnalyzer
from schemas import ReviewAnalysisOutput
import json

def main():
    analyzer = LLMReviewAnalyzer()

    # Example 1: Positive review with specific features
    review1_text = "This phone is amazing! The camera is absolutely excellent, takes stunning photos. The battery life, however, could be significantly better."
    review1_id = "review_001"
    product1_id = "prod_XYZ"
    
    print(f"\nAnalyzing Review 1 (ID: {review1_id}):\n'{review1_text}'")
    try:
        structured_output1: ReviewAnalysisOutput = analyzer.analyze_review(review1_text, review1_id, product1_id)
        print("Structured Output 1:")
        print(json.dumps(structured_output1.model_dump(), indent=2))
    except ValueError as e:
        print(f"Error processing review 1: {e}")

    # Example 2: Negative review
    review2_text = "Absolutely terrible product, complete waste of money. It broke within a week of light use. Very disappointed."
    review2_id = "review_002"
    product2_id = "prod_ABC"

    print(f"\nAnalyzing Review 2 (ID: {review2_id}):\n'{review2_text}'")
    try:
        structured_output2: ReviewAnalysisOutput = analyzer.analyze_review(review2_text, review2_id, product2_id)
        print("Structured Output 2:")
        print(json.dumps(structured_output2.model_dump(), indent=2))
    except ValueError as e:
        print(f"Error processing review 2: {e}")

    # Example 3: Neutral review
    review3_text = "It's an OK product. Does what it's supposed to, nothing really stands out either good or bad. Just average."
    review3_id = "review_003"
    product3_id = "prod_DEF"

    print(f"\nAnalyzing Review 3 (ID: {review3_id}):\n'{review3_text}'")
    try:
        structured_output3: ReviewAnalysisOutput = analyzer.analyze_review(review3_text, review3_id, product3_id)
        print("Structured Output 3:")
        print(json.dumps(structured_output3.model_dump(), indent=2))
    except ValueError as e:
        print(f"Error processing review 3: {e}")

if __name__ == "__main__":
    main()