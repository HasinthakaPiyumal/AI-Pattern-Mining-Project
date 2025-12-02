def get_user_input():
    return input("Customer: ")

def detect_emotion(query):
    query_lower = query.lower()
    if "urgent" in query_lower or "quickly" in query_lower or "asap" in query_lower:
        return "urgent"
    elif "important" in query_lower or "critical" in query_lower:
        return "important"
    elif "frustrated" in query_lower or "annoyed" in query_lower or "unhappy" in query_lower:
        return "frustrated"
    return "neutral"

def augment_prompt(query, emotion):
    emotional_phrase = ""
    if emotion == "urgent":
        emotional_phrase = "It is critical to my reputation to resolve this customer's urgent issue efficiently and with utmost care. "
    elif emotion == "important":
        emotional_phrase = "This customer's request is very important, treat it with high priority and diligence. "
    elif emotion == "frustrated":
        emotional_phrase = "The customer is expressing frustration, respond with extra empathy and understanding. "
    
    return f"{emotional_phrase}Customer query: {query}"

def simulate_llm_response(augmented_prompt):
    if "urgent" in augmented_prompt.lower():
        return "I understand this is an urgent matter. I will prioritize finding a quick and effective solution for you immediately."
    elif "important" in augmented_prompt.lower():
        return "I recognize the importance of this issue. I will carefully review your request and provide a thorough solution."
    elif "frustrated" in augmented_prompt.lower():
        return "I hear your frustration, and I apologize for the inconvenience. I'm here to help resolve this for you as smoothly as possible."
    else:
        return "Thank you for your query. I will assist you with this."

def display_response(response):
    print(f"Support AI: {response}")

def main():
    print("Empathetic Customer Support AI (Type 'exit' to quit)")
    while True:
        customer_query = get_user_input()
        if customer_query.lower() == 'exit':
            break
        
        detected_emotion = detect_emotion(customer_query)
        augmented_llm_prompt = augment_prompt(customer_query, detected_emotion)
        llm_response = simulate_llm_response(augmented_llm_prompt)
        display_response(llm_response)

if __name__ == "__main__":
    main()