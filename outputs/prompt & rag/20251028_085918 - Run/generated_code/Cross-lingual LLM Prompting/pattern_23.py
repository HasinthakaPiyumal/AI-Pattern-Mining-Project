import os
import openai

class MultilingualChatbot:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.in_context_examples = [
            {
                "query_en": "What is your return policy for electronics?",
                "response_en": "Our return policy for electronics allows returns within 30 days with the original receipt and packaging.",
                "query_es": "¿Cuál es su política de devoluciones para productos electrónicos?",
                "response_es": "Nuestra política de devoluciones para productos electrónicos permite devoluciones dentro de los 30 días con el recibo original y el embalaje.",
            },
            {
                "query_en": "How can I track my order?",
                "response_en": "You can track your order by logging into your account and visiting the 'My Orders' section.",
                "query_es": "¿Cómo puedo rastrear mi pedido?",
                "response_es": "Puede rastrear su pedido iniciando sesión en su cuenta y visitando la sección 'Mis Pedidos'.",
            },
            {
                "query_fr": "Comment puis-je réinitialiser mon mot de passe?",
                "response_fr": "Pour réinitialiser votre mot de passe, veuillez cliquer sur le lien 'Mot de passe oublié' sur la page de connexion.",
                "query_en": "How can I reset my password?",
                "response_en": "To reset your password, please click on the 'Forgot Password' link on the login page.",
            }
        ]
        self.model_name = "gpt-3.5-turbo"

    def _generate_inclt_prompt(self, user_query, source_lang, target_lang):
        prompt_parts = []
        prompt_parts.append(f"You are a multilingual customer support assistant. Your task is to respond to customer queries in {target_lang}. To help you understand the nuances and generate accurate responses across languages, here are some examples incorporating both source and target languages for common queries:\n\n")

        for i, example in enumerate(self.in_context_examples):
            prompt_parts.append(f"--- Example {i+1} ---\n")
            
            if f"query_{source_lang}" in example and f"response_{source_lang}" in example:
                prompt_parts.append(f"Query ({source_lang}): {example[f'query_{source_lang}']}\n")
                prompt_parts.append(f"Response ({source_lang}): {example[f'response_{source_lang}']}\n")
            elif f"query_en" in example and f"response_en" in example and source_lang != "en":
                prompt_parts.append(f"Query (en - for cross-lingual context): {example['query_en']}\n")
                prompt_parts.append(f"Response (en - for cross-lingual context): {example['response_en']}\n")

            if f"query_{target_lang}" in example and f"response_{target_lang}" in example:
                prompt_parts.append(f"Query ({target_lang}): {example[f'query_{target_lang}']}\n")
                prompt_parts.append(f"Response ({target_lang}): {example[f'response_{target_lang}']}\n")
            elif f"query_en" in example and f"response_en" in example and target_lang != "en":
                prompt_parts.append(f"Query (en - for cross-lingual context): {example['query_en']}\n")
                prompt_parts.append(f"Response (en - for cross-lingual context): {example['response_en']}\n")

            prompt_parts.append("\n")

        prompt_parts.append(f"--- User's Current Query ---\n")
        prompt_parts.append(f"User Query ({source_lang}): {user_query}\n")
        prompt_parts.append(f"Assistant Response ({target_lang}): ")
        return "".join(prompt_parts)

    def get_response(self, user_query, source_lang="en", target_lang="en"):
        prompt = self._generate_inclt_prompt(user_query, source_lang, target_lang)
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set the environment variable with your OpenAI API key to run this example.")
    else:
        chatbot = MultilingualChatbot(api_key)

        print("Multilingual Customer Support Chatbot (Type 'exit' to quit)")

        while True:
            user_input = input("\nEnter an English query (type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break

            english_response = chatbot.get_response(user_input, source_lang="en", target_lang="en")
            print(f"Chatbot (English response): {english_response}")

            spanish_response = chatbot.get_response(user_input, source_lang="en", target_lang="es")
            print(f"Chatbot (Spanish response from English query): {spanish_response}")
            
            user_input_es = input("\nEnter a Spanish query (type 'exit' to quit): ")
            if user_input_es.lower() == 'exit':
                break
            english_response_from_es = chatbot.get_response(user_input_es, source_lang="es", target_lang="en")
            print(f"Chatbot (English response from Spanish query): {english_response_from_es}")

            french_response = chatbot.get_response("Comment puis-je suivre ma commande?", source_lang="fr", target_lang="fr")
            print(f"Chatbot (French response from French query): {french_response}")

            spanish_response_from_fr = chatbot.get_response("Comment puis-je suivre ma commande?", source_lang="fr", target_lang="es")
            print(f"Chatbot (Spanish response from French query): {spanish_response_from_fr}")