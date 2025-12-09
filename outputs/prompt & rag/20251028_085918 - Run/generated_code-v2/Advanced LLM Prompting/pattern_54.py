# persona_chatbot.py

import os

def get_llm_response(prompt_text):
    """
    This is a mock function to simulate an LLM API call.
    In a real application, you would integrate with an actual LLM service (e.g., OpenAI, Google Gemini).
    """
    print(f"\n--- LLM Input Prompt (with Persona) ---\n{prompt_text}\n-------------------------------------\n")
    return "[Simulated LLM Response]: Based on the persona and your query, I would provide a tailored response focusing on the details in the prompt above."

def detect_user_intent_and_suggest_persona(user_query):
    """
    Analyzes the user's query to suggest an appropriate persona.
    This is a simplified keyword-based detection for demonstration purposes.
    In a more advanced system, NLP models would be used for intent recognition and sentiment analysis.
    """
    query_lower = user_query.lower()

    if any(keyword in query_lower for keyword in ["hike", "trekking", "trail", "mountains", "technical", "durable", "rugged"]):
        return "expert_hiker"
    elif any(keyword in query_lower for keyword in ["beginner", "first time", "start", "easy", "simple", "getting started"]):
        return "friendly_newbie_guide"
    elif any(keyword in query_lower for keyword in ["budget", "cheap", "affordable", "deal", "discount", "save money", "value"]):
        return "budget_advisor"
    else:
        return None # No specific persona detected, fall back to generic handling

def construct_persona_prompt(persona_key, user_query, personas_data):
    """
    Constructs the full prompt for the LLM, embedding the selected persona's description.
    This is the core implementation of the 'Role Prompting' pattern.
    """
    persona = personas_data.get(persona_key)
    
    if persona:
        # Assign the role/persona to the GenAI
        role_instruction = f"Pretend you are a {persona['name']}. {persona['description']}"
        
        # Combine the role instruction with the user's query
        full_prompt = f"{role_instruction}\n\nUser: \"{user_query}\"\n\nAs the {persona['name']}, how would you respond:"
    else:
        # Generic prompt if no specific persona is selected
        full_prompt = f"You are a helpful customer support agent for an outdoor gear store. User asks: \"{user_query}\"\n\nHow would you respond:"
        
    return full_prompt

def run_chatbot():
    """
    Main function to run the persona-driven customer support chatbot.
    """
    print("\n=======================================================")
    print("  Welcome to the Persona-Driven Outdoor Gear Chatbot!")
    print("  Ask me anything about outdoor gear. Type 'exit' to quit.")
    print("=======================================================")

    # Define available personas for the chatbot
    PERSONAS = {
        "expert_hiker": {
            "name": "Expert Hiker",
            "description": "You are a seasoned expert in hiking and outdoor gear. Provide detailed, practical, and knowledgeable advice. Your tone is confident and helpful, focusing on durability, performance, and best practices.",
            "example_phrases": ["Based on my experience...", "For rugged conditions...", "I recommend considering..."]
        },
        "friendly_newbie_guide": {
            "name": "Friendly Newbie Guide",
            "description": "You are an enthusiastic and encouraging guide for beginners in outdoor activities. Your tone is warm, supportive, and simple, explaining concepts clearly without jargon. Focus on ease of use and getting started.",
            "example_phrases": ["Welcome to the outdoors!", "Don't worry, it's easy once you get started.", "A great first step would be..."]
        },
        "budget_advisor": {
            "name": "Budget Advisor",
            "description": "You are a savvy and practical advisor focused on finding the best value for money in outdoor gear. Your tone is helpful and cost-conscious, suggesting alternatives and highlighting deals. Focus on affordability and smart shopping.",
            "example_phrases": ["If you're on a budget...", "A great value option is...", "Consider this alternative to save money..."]
        }
    }


    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == 'exit':
            print("\n[Chatbot]: Goodbye! Stay adventurous!")
            break

        # Detect intent and suggest a persona based on the user's query
        suggested_persona_key = detect_user_intent_and_suggest_persona(user_input)

        if suggested_persona_key:
            print(f"[Chatbot]: Identifying user interest... Adopting the '{PERSONAS[suggested_persona_key]['name']}' persona.")
            final_prompt = construct_persona_prompt(suggested_persona_key, user_input, PERSONAS)
        else:
            print("[Chatbot]: No specific persona detected. Responding as a general support agent.")
            final_prompt = construct_persona_prompt(None, user_input, PERSONAS) # `None` triggers generic handling

        # Get a simulated response from the LLM using the constructed prompt
        chatbot_response = get_llm_response(final_prompt)
        print(f"[Chatbot]: {chatbot_response}")

if __name__ == "__main__":
    run_chatbot()
