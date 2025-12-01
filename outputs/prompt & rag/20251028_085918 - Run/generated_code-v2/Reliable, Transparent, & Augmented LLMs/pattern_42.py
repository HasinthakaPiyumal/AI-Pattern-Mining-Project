def moderate_review(review_text: str) -> str:
    inappropriate_keywords = ["hate speech", "scam", "fraud", "offensive", "illegal", "terrible quality", "broken on arrival"]
    
    review_text_lower = review_text.lower()
    
    for keyword in inappropriate_keywords:
        if keyword in review_text_lower:
            return "Inappropriate"
            
    return "Appropriate"

if __name__ == "__main__":
    # Example Usage
    review1 = "This product is amazing, I love it!"
    review2 = "Beware, this is a total scam and a waste of money."
    review3 = "The quality is good, highly recommend."
    review4 = "I received a broken on arrival item."
    
    print(f"Review 1: '{review1}' -> {moderate_review(review1)}")
    print(f"Review 2: '{review2}' -> {moderate_review(review2)}")
    print(f"Review 3: '{review3}' -> {moderate_review(review3)}")
    print(f"Review 4: '{review4}' -> {moderate_review(review4)}")