ambiguous_demonstrations = [
    {
        "ambiguous_question": "Where is my order?",
        "clarifications": [
            "Are you looking for tracking information?",
            "Do you want to know the estimated delivery date?",
            "Are you trying to find the current status of your order?"
        ]
    },
    {
        "ambiguous_question": "I need help with my account.",
        "clarifications": [
            "Are you having trouble logging in?",
            "Do you need to update your personal information?",
            "Are you looking to change your password?"
        ]
    },
    {
        "ambiguous_question": "What about returns?",
        "clarifications": [
            "Are you asking about our return policy?",
            "Do you want to initiate a return for an item?",
            "Are you looking for information on how to package a return?"
        ]
    }
]

def generate_prompt(user_query, demonstrations):
    system_instruction = "You are an AI customer support chatbot for an e-commerce platform. When a question is ambiguous, provide a set of clarification options or follow-up questions instead of a direct answer, based on the provided examples. If the question is clear, provide a direct answer."
    demonstration_section = "\n\n--- Ambiguous Demonstration Examples ---\n"
    for demo in demonstrations:
        demonstration_section += f"User: {demo['ambiguous_question']}\n"
        demonstration_section += "Bot: " + " | ".join(demo['clarifications']) + "\n\n"
    demonstration_section += "--------------------------------------\n\n"
    final_prompt = f"{system_instruction}{demonstration_section}User: {user_query}\nBot:"
    return final_prompt

def simulate_llm_response(prompt, demonstrations):
    user_query = prompt.split("User: ")[-1].split("\nBot:")[0].strip()
    for demo in demonstrations:
        if user_query.lower() == demo["ambiguous_question"].lower():
            return " | ".join(demo["clarifications"])
    return f"Thank you for your query regarding '{user_query}'. I am processing your request and will provide a direct answer shortly."

def chatbot_response():
    print("Welcome to the E-commerce Customer Support Chatbot! (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        prompt = generate_prompt(user_input, ambiguous_demonstrations)
        llm_output = simulate_llm_response(prompt, ambiguous_demonstrations)
        print(f"Chatbot: {llm_output}")

if __name__ == "__main__":
    chatbot_response()