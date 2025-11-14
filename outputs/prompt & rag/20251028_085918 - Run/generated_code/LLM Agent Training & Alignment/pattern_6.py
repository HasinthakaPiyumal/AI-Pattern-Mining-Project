"""Simulates the collection of dual data (demonstrations and preferences) for training an AI customer support agent."""

import random

def generate_demonstration_data(num_samples: int = 100):
    """Generates synthetic human demonstration data."""
    demonstrations = []
    for i in range(num_samples):
        user_query = f"I want to return item {1000 + i}. It's damaged."
        agent_response = (
            f"Certainly, I can help you with that return for item {1000 + i}. "
            f"Could you please provide your order number and a photo of the damage?"
        )
        demonstrations.append({"query": user_query, "response": agent_response})
    return demonstrations

def generate_preference_data(num_samples: int = 50):
    """Generates synthetic human preference comparison data."""
    preferences = []
    for i in range(num_samples):
        user_query = f"My order {2000 + i} arrived late. I want a refund."
        candidate_a = (
            f"I apologize for the delay with order {2000 + i}. "
            f"Let me check the tracking and processing a partial refund for you."
        )
        candidate_b = (
            f"We're sorry your order {2000 + i} was late. "
            f"Please wait for 3-5 business days for us to investigate."
        )
        # Randomly decide which candidate is preferred
        preferred = random.choice(["A", "B"])
        preferences.append({
            "query": user_query,
            "response_a": candidate_a,
            "response_b": candidate_b,
            "preferred": preferred
        })
    return preferences

if __name__ == "__main__":
    print("Generating synthetic data...")
    demo_data = generate_demonstration_data()
    pref_data = generate_preference_data()

    print(f"Generated {len(demo_data)} demonstration samples.")
    print(f"Example demonstration: {demo_data[0]}")

    print(f"Generated {len(pref_data)} preference samples.")
    print(f"Example preference: {pref_data[0]}")

    # In a real scenario, you'd save this data to files or a database
    # e.g., import json; with open("demonstrations.json", "w") as f: json.dump(demo_data, f)
