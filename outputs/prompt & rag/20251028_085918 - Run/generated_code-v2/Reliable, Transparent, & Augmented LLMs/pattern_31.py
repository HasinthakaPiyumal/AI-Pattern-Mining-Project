import random

def get_simulated_llm_response(query: str) -> (str, float):
    """
    Simulates an LLM's response for a medical query, including a diagnosis and a confidence score.
    In a real application, this would involve an actual LLM API call (e.g., OpenAI, Hugging Face).
    The confidence score here is simulated based on keywords and some randomness.
    """
    query_lower = query.lower()
    diagnosis = "Uncertain diagnosis. Please provide more information."
    confidence = random.uniform(0.4, 0.7) # Default lower confidence for general queries

    # Simulate diagnoses and confidence based on keywords
    if "headache" in query_lower and "severe" in query_lower and "nausea" in query_lower:
        diagnosis = "Migraine with aura"
        confidence = random.uniform(0.85, 0.98)
    elif "fever" in query_lower and "cough" in query_lower and "sore throat" in query_lower:
        diagnosis = "Common cold or Flu-like illness"
        confidence = random.uniform(0.75, 0.95)
    elif "chest pain" in query_lower and "shortness of breath" in query_lower:
        diagnosis = "Possible cardiac event or anxiety attack. Immediate medical attention recommended."
        confidence = random.uniform(0.60, 0.85) # Acknowledge seriousness but maintain some uncertainty
    elif "rash" in query_lower and "itchy" in query_lower:
        diagnosis = "Allergic reaction or Dermatitis"
        confidence = random.uniform(0.70, 0.90)
    elif "abdominal pain" in query_lower and "vomiting" in query_lower:
        diagnosis = "Gastroenteritis or Food poisoning"
        confidence = random.uniform(0.65, 0.88)
    elif "fatigue" in query_lower and "joint pain" in query_lower:
        diagnosis = "Potential autoimmune condition or chronic fatigue syndrome. Further tests needed."
        confidence = random.uniform(0.50, 0.75)

    # Simulate slightly lower confidence for less specific or complex queries
    if len(query.split()) < 5 or any(word in query_lower for word in ["complex", "rare", "unusual"]):
        confidence = max(0.4, confidence * random.uniform(0.7, 0.9))

    return diagnosis, round(confidence, 2)

def main():
    print("\n--- Medical Diagnosis Assistant with Confidence Scores ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("Enter patient symptoms or medical query: ")
        if user_query.lower() == 'exit':
            break

        if not user_query.strip():
            print("Please enter a valid query.\n")
            continue

        diagnosis, confidence = get_simulated_llm_response(user_query)

        print(f"\nAI Suggested Diagnosis: {diagnosis}")
        print(f"Confidence Score (0.0-1.0): {confidence:.2f}")

        # Abstention/Flagging Logic based on Confidence
        if confidence < 0.60:
            print("\n*** Low Confidence: This diagnosis requires significant human review or further investigation. ***")
        elif confidence < 0.75:
            print("\n* Moderate Confidence: Consider this diagnosis, but further verification is advisable. *")
        else:
            print("\nHigh Confidence: This diagnosis is likely correct, but clinical judgment is always paramount.")
        print("----------------------------------------------------------\n")

if __name__ == "__main__":
    main()
