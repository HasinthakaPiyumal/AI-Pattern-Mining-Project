from transformers import pipeline

# 1. NLU Module Initialization
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels for intents
candidate_labels = [
    "order status",
    "returns and refunds",
    "product information",
    "technical support",
    "billing inquiry",
    "delivery issue",
    "account management",
    "general inquiry",
    "escalate to human agent"
]

# 2. Intent-to-Action Mapping and 3. Personalized Response Generation (Simulated)
def get_order_status(query):
    return f"To check your order status, please provide your order number. You can find it in your confirmation email."

def handle_returns_refunds(query):
    return f"For returns and refunds, please visit our 'Returns Policy' page or provide your order number to initiate a return."

def get_product_information(query):
    return f"I can help with product information. Could you please specify which product you are interested in?"

def provide_technical_support(query):
    return f"For technical support, please describe your issue in more detail, or visit our FAQ for common troubleshooting steps."

def handle_billing_inquiry(query):
    return f"Regarding billing inquiries, please provide your account details or the order number related to the charge."

def handle_delivery_issue(query):
    return f"I understand you're having a delivery issue. Please provide your order number so I can investigate."

def manage_account(query):
    return f"For account management, please log in to your account to update your details or reset your password."

def general_inquiry(query):
    return f"I'm here to help with general questions. What would you like to know?"

def escalate_to_human_agent(query):
    return f"I'm sorry I couldn't fully assist you. I'm escalating your query to a human agent who will contact you shortly."

# Map intents to functions
intent_actions = {
    "order status": get_order_status,
    "returns and refunds": handle_returns_refunds,
    "product information": get_product_information,
    "technical support": provide_technical_support,
    "billing inquiry": handle_billing_inquiry,
    "delivery issue": handle_delivery_issue,
    "account management": manage_account,
    "general inquiry": general_inquiry,
    "escalate to human agent": escalate_to_human_agent
}

# 4. User Interface (CLI)
def run_chatbot():
    print("Welcome to E-commerce Customer Support! How can I help you today?")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break

        if not user_input.strip():
            print("Chatbot: Please enter a query.")
            continue

        # NLU processing
        results = classifier(user_input, candidate_labels)
        predicted_intent = results["labels"][0]
        confidence_score = results["scores"][0]

        print(f"Chatbot (Intent Detected: '{predicted_intent}' with confidence {confidence_score:.2f}):")

        # Intent-to-Action Mapping
        action_function = intent_actions.get(predicted_intent, general_inquiry) # Default to general inquiry
        response = action_function(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    run_chatbot()