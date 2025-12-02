from langdetect import detect
import openai
import os

# Set your OpenAI API key
# It's recommended to load this from an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def construct_icl_prompt(user_query: str, detected_lang: str) -> list:
    # Hardcoded In-Context Learning examples demonstrating cross-lingual transfer
    # Each example includes a problem in one language and a solution (or related text) in another,
    # and also a reverse example to stimulate bi-directional transfer.
    icl_examples = [
        {"role": "user", "content": "My internet is not working. (English)"},
        {"role": "assistant", "content": "Por favor, reinicie su router. (Spanish)"},
        {"role": "user", "content": "Mi internet no funciona. (Spanish)"},
        {"role": "assistant", "content": "Please restart your router. (English)"},
        {"role": "user", "content": "J'ai un problème avec ma facture. (French)"},
        {"role": "assistant", "content": "Your bill issue will be investigated. (English)"},
        {"role": "user", "content": "My bill has an error. (English)"},
        {"role": "assistant", "content": "Votre problème de facturation sera examiné. (French)"},
        {"role": "user", "content": "Ich kann mich nicht anmelden. (German)"},
        {"role": "assistant", "content": "Please check your login credentials. (English)"},
        {"role": "user", "content": "I cannot log in. (English)"},
        {"role": "assistant", "content": "Bitte überprüfen Sie Ihre Anmeldeinformationen. (German)"},
    ]

    # System message to guide the LLM
    system_message = {"role": "system", "content": f"You are a helpful multilingual customer support assistant. Respond to the user in their original query language if possible, otherwise use English. Your goal is to provide clear and concise solutions based on the context. Leverage the provided examples for cross-lingual understanding."}

    # Combine system message, ICL examples, and the user's current query
    messages = [system_message] + icl_examples + [
        {"role": "user", "content": user_query}
    ]
    return messages

def get_chatbot_response(prompt_messages: list) -> str:
    try:
        # Interact with the OpenAI LLM
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Or other suitable multilingual model
            messages=prompt_messages,
            temperature=0.7, # Controls randomness
            max_tokens=150   # Limits the length of the response
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error communicating with the chatbot: {e}"

def main():
    print("Multilingual Customer Support Chatbot (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        if not openai.api_key:
            print("Error: OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")
            continue

        try:
            detected_lang = detect(user_input)
            print(f"Detected Language: {detected_lang}")
        except Exception:
            detected_lang = "unknown"
            print("Could not reliably detect language, assuming English for prompting.")

        prompt = construct_icl_prompt(user_input, detected_lang)
        chatbot_response = get_chatbot_response(prompt)
        print(f"Chatbot: {chatbot_response}")

if __name__ == "__main__":
    main()