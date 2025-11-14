def generate_demonstrations(num_interactions=100):
    """
    Simulates the generation of human demonstration data for behavior cloning.
    Each demonstration consists of a user query and an ideal agent response.
    """
    demonstrations = []
    for i in range(num_interactions):
        query = f"User query {i+1}: I have a problem with my order {1000 + i}."
        response = f"Agent response {i+1}: I understand you have an issue with order {1000 + i}. Let me check that for you."
        demonstrations.append({"query": query, "response": response})
    return demonstrations

def generate_preference_data(num_comparisons=50):
    """
    Simulates the generation of human preference data for reward modeling.
    Each entry consists of a query and two agent responses, with one marked as preferred.
    """
    preference_data = []
    for i in range(num_comparisons):
        query = f"User query {i+1}: My internet is not working."
        response_a = f"Agent response A {i+1}: Have you tried restarting your router?"
        response_b = f"Agent response B {i+1}: I'm sorry to hear that. Let's troubleshoot your connection step by step."
        # Simulate human preference: response B is generally better
        preference_data.append({"query": query, "chosen": response_b, "rejected": response_a})
    return preference_data

if __name__ == "__main__":
    print("Generating simulated demonstration data...")
    demos = generate_demonstrations()
    print(f"Generated {len(demos)} demonstrations. Example: {demos[0]}")

    print("\nGenerating simulated preference data...")
    prefs = generate_preference_data()
    print(f"Generated {len(prefs)} preference comparisons. Example: {prefs[0]}")