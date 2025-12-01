"""
Configuration settings for the e-commerce chatbot.
"""

# Placeholder for a fine-tuned model. In a real scenario, this would be a path
# to a model hosted locally or on a platform like Hugging Face.
FINE_TUNED_MODEL_NAME = "distilbert-base-uncased-finetuned-intent-ecommerce" # Example name

# Threshold for intent confidence to consider it ambiguous
CONFIDENCE_THRESHOLD = 0.7

# Example FAQs (for demonstration purposes)
FAQS = {
    "shipping": "Standard shipping takes 3-5 business days. Expedited shipping options are available at checkout.",
    "returns": "You can return most items within 30 days of purchase. Please visit our returns policy page for more details.",
    "payment_methods": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
    "order_status": "To check your order status, please provide your order number."
}