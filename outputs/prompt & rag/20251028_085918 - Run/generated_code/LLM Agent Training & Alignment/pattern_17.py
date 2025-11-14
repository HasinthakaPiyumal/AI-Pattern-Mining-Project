import random

def generate_human_demonstrations(num_demonstrations=10):
    """Simulates generation of human demonstration data for Behavior Cloning.
    Returns a list of (query, resolution_steps) tuples.
    """
    demonstrations = [
        ("My internet is not working.", ["Check router lights.", "Restart router.", "Contact ISP if issue persists."]),
        ("How do I return a product?", ["Go to 'My Orders'.", "Select item to return.", "Follow return instructions.", "Print return label."]),
        ("My account is locked.", ["Try 'Forgot Password'.", "Verify identity with security questions.", "Contact support for manual unlock."]),
        ("What are your operating hours?", ["Our support is available 24/7."]),
        ("How to change my shipping address?", ["Go to account settings.", "Find 'Addresses'.", "Edit or add new address."]),
        ("I received a damaged item.", ["Take photos of damage.", "Contact us with order details and photos.", "We will arrange replacement or refund."]),
        ("Where is my order?", ["Check tracking number provided in order confirmation.", "Visit carrier website."]),
        ("How do I update my payment method?", ["Go to account settings.", "Select 'Payment Methods'.", "Add or update card details."]),
        ("Can I cancel my subscription?", ["Go to 'Subscriptions' in account.", "Select subscription to cancel.", "Confirm cancellation."]),
        ("My software is crashing.", ["Check system requirements.", "Update software to latest version.", "Reinstall software if issue continues.", "Check for driver updates."])
    ]
    return random.sample(demonstrations, min(num_demonstrations, len(demonstrations)))

def generate_human_preferences(agent_outputs_per_query=3, num_queries=5):
    """Simulates generation of human preference data for training a Reward Model.
    Returns a list of (query, [output_A, output_B, ...], preferred_index) tuples.
    """
    preference_data = []
    sample_queries = [
        "My internet is slow.",
        "I want to change my flight.",
        "How do I reset my smart home device?",
        "What's the warranty on this product?",
        "I can't log into my email."
    ]

    for query in random.sample(sample_queries, min(num_queries, len(sample_queries))):
        possible_outputs = []
        # Simulate different agent-generated responses
        if "internet is slow" in query:
            possible_outputs = [
                "Try restarting your router and modem. If that doesn't work, contact your ISP.",
                "Check if other devices are using a lot of bandwidth. Also, ensure your Wi-Fi signal is strong.",
                "Have you considered upgrading your internet plan? Sometimes that helps."
            ]
        elif "change my flight" in query:
            possible_outputs = [
                "Go to 'Manage Booking' on our website, enter your booking reference, and follow the steps to modify your flight.",
                "You can change your flight through our mobile app or by calling our customer service line. Fees may apply.",
                "Flight changes are usually done online. Make sure to check the change fees for your ticket type."
            ]
        elif "reset my smart home device" in query:
            possible_outputs = [
                "Locate the reset button on your device, usually a small pinhole. Press and hold it for 10-15 seconds until the indicator light flashes.",
                "Refer to your device's manual for specific reset instructions, as it varies by model.",
                "Unplug the device, wait 30 seconds, then plug it back in. This often resolves minor issues."
            ]
        elif "warranty on this product" in query:
            possible_outputs = [
                "The warranty period for this product is typically one year from the date of purchase. Please check your purchase receipt for exact terms.",
                "All our products come with a standard 12-month warranty covering manufacturing defects. Extended warranties may be available for purchase.",
                "To find the warranty information, please provide the product model number or check the product page on our website."
            ]
        elif "can't log into my email" in query:
            possible_outputs = [
                "First, try resetting your password using the 'Forgot Password' link. If that fails, ensure your internet connection is stable.",
                "Check if Caps Lock is on. Also, try logging in from a different browser or device. If still no luck, contact your email provider directly.",
                "Have you recently changed your password? Sometimes it takes a moment to sync. Clear your browser cache and cookies."
            ]

        if possible_outputs:
            # Randomly select a preferred output for demonstration purposes
            preferred_idx = random.randint(0, len(possible_outputs) - 1)
            preference_data.append((query, possible_outputs, preferred_idx))

    return preference_data

# Example usage (for testing purposes, not part of the final code output)
# if __name__ == "__main__":
#     print("Human Demonstrations:")
#     demos = generate_human_demonstrations(2)
#     for q, res in demos:
#         print(f"  Query: {q}\n  Resolution: {res}")

#     print("\nHuman Preferences:")
#     prefs = generate_human_preferences(num_queries=2)
#     for q, outputs, pref_idx in prefs:
#         print(f"  Query: {q}\n  Outputs: {outputs}\n  Preferred: {outputs[pref_idx]}")