"""Configuration settings for the e-commerce chatbot."""

INTENTS = {
    "order_status": ["where is my order", "track my package", "delivery status"],
    "return_request": ["i want to return", "return an item", "how to return"],
    "product_inquiry": ["tell me about product", "product details", "specifications"],
    "billing_issue": ["payment problem", "incorrect charge", "bill issue"],
    "technical_support": ["website error", "app not working", "login issue"],
    "general_inquiry": ["hello", "hi", "question", "help"]
}

INTENT_RESPONSES = {
    "order_status": "Please provide your order number, and I will check the status for you.",
    "return_request": "To initiate a return, please visit our returns page or provide your order number and the item you wish to return.",
    "product_inquiry": "Could you please specify which product you're interested in?",
    "billing_issue": "Please provide your order number or account details so I can look into your billing issue.",
    "technical_support": "I understand you're experiencing a technical issue. Could you describe it in more detail?",
    "general_inquiry": "Hello! How can I assist you today?"
}

CONFIDENCE_THRESHOLD = 0.6 # Threshold to determine if intent is clear

# Mock personalization data
MOCK_USER_PROFILES = {
    "user123": {
        "name": "Alice",
        "preferred_language": "English",
        "past_orders": ["ORD789", "ORD456"],
        "recent_intent": "order_status"
    },
    "user456": {
        "name": "Bob",
        "preferred_language": "Spanish",
        "past_orders": ["ORD123"],
        "recent_intent": "billing_issue"
    }
}
