class MockMachineTranslationService:
    def __init__(self):
        self.lang_to_english = {
            "es": {
                "hola": "hello",
                "ayuda": "help",
                "problema": "problem",
                "gracias": "thank you",
                "envío": "shipping",
                "orden": "order"
            },
            "fr": {
                "bonjour": "hello",
                "aide": "help",
                "problème": "problem",
                "merci": "thank you",
                "expédition": "shipping",
                "commande": "order"
            }
        }
        self.english_to_lang = {
            "es": {
                "hello": "hola",
                "how can I help you": "¿cómo puedo ayudarte?",
                "we can help with your problem": "podemos ayudarte con tu problema",
                "thank you for contacting us": "gracias por contactarnos",
                "your shipping details are": "los detalles de tu envío son",
                "your order status is": "el estado de tu pedido es"
            },
            "fr": {
                "hello": "bonjour",
                "how can I help you": "comment puis-je vous aider ?",
                "we can help with your problem": "nous pouvons vous aider avec votre problème",
                "thank you for contacting us": "merci de nous avoir contactés",
                "your shipping details are": "les détails de votre expédition sont",
                "your order status is": "le statut de votre commande est"
            }
        }

    def translate_to_english(self, text, source_language):
        translated_words = [self.lang_to_english.get(source_language, {}).get(word.lower(), word) for word in text.split()]
        return " ".join(translated_words)

    def translate_from_english(self, text, target_language):
        translated_words = [self.english_to_lang.get(target_language, {}).get(word.lower(), word) for word in text.split()]
        return " ".join(translated_words)

class MockGenAIChatbot:
    def get_english_response(self, english_query):
        english_query_lower = english_query.lower()
        if "help" in english_query_lower or "problem" in english_query_lower:
            return "hello, how can I help you? we can help with your problem."
        elif "shipping" in english_query_lower:
            return "your shipping details are being processed."
        elif "order" in english_query_lower:
            return "your order status is pending."
        else:
            return "thank you for contacting us, how can I assist you further?"

def get_language_from_input(text):
    text_lower = text.lower()
    if "hola" in text_lower or "ayuda" in text_lower:
        return "es"
    elif "bonjour" in text_lower or "aide" in text_lower:
        return "fr"
    else:
        return "en"

def main():
    mt_service = MockMachineTranslationService()
    genai_chatbot = MockGenAIChatbot()

    print("Multilingual Customer Support Chatbot (type 'exit' to quit)")
    while True:
        customer_input = input("You: ")
        if customer_input.lower() == 'exit':
            break

        detected_language = get_language_from_input(customer_input)
        print(f"(Detected Language: {detected_language.upper()})")

        if detected_language != "en":
            english_query = mt_service.translate_to_english(customer_input, detected_language)
            print(f"(Translated to English for Chatbot: {english_query})")
        else:
            english_query = customer_input

        english_response = genai_chatbot.get_english_response(english_query)
        print(f"(Chatbot's English Response: {english_response})")

        if detected_language != "en":
            final_response = mt_service.translate_from_english(english_response, detected_language)
            print(f"Chatbot: {final_response}")
        else:
            print(f"Chatbot: {english_response}")

if __name__ == "__main__":
    main()