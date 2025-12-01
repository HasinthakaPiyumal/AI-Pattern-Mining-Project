import os
from clsp_reasoner import CLSPReasoner
from language_detector import LanguageDetector

class MultilingualChatbot:
    def __init__(self, api_key: str):
        self.language_detector = LanguageDetector()
        self.clsp_reasoner = CLSPReasoner(api_key=api_key)

    def get_response(self, query: str) -> str:
        detected_lang = self.language_detector.detect(query)
        print(f"Detected language: {detected_lang}")

        # Define a set of languages for CLSP ensembling
        # These could be dynamically determined or a fixed set based on support
        target_languages = ["en", "es", "fr", "de"] # English, Spanish, French, German

        # Ensure the detected language is included in the target languages for primary reasoning
        if detected_lang not in target_languages:
            target_languages.append(detected_lang)

        final_answer = self.clsp_reasoner.reason_with_consistency(query, target_languages)
        return final_answer

if __name__ == "__main__":
    # In a real application, get API key securely (e.g., from environment variables)
    # For this example, replace "YOUR_OPENAI_API_KEY" with your actual key
    openai_api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    if openai_api_key == "YOUR_OPENAI_API_KEY":
        print("WARNING: Please set the OPENAI_API_KEY environment variable or replace 'YOUR_OPENAI_API_KEY' in main.py.")

    chatbot = MultilingualChatbot(api_key=openai_api_key)

    print("\nMultilingual CLSP Chatbot (Type 'exit' to quit)")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            break

        response = chatbot.get_response(user_query)
        print(f"Bot: {response}")
