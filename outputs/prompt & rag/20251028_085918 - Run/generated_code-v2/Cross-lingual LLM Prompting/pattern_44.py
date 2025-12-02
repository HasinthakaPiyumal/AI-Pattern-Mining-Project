import langdetect
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def simulated_llm_response(prompt: str) -> str:
    response_lang_indicator = "Response ("
    if response_lang_indicator in prompt:
        start_index = prompt.rfind(response_lang_indicator) + len(response_lang_indicator)
        end_index = prompt.find("):", start_index)
        if start_index != -1 and end_index != -1:
            requested_lang = prompt[start_index:end_index].strip().lower()
        else:
            requested_lang = "en"
    else:
        requested_lang = "en"

    if any(keyword in prompt.lower() for keyword in ["refund policy", "política de reembolso", "politique de remboursement"]):
        if requested_lang == "es":
            return "Nuestra política de reembolso permite devoluciones dentro de los 30 días posteriores a la compra, siempre que el artículo esté sin usar y en su embalaje original."
        elif requested_lang == "fr":
            return "Notre politique de remboursement autorise les retours dans les 30 jours suivant l'achat, à condition que l'article soit inutilisé et dans son emballage d'origine."
        else:
            return "Our refund policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging."
    
    elif any(keyword in prompt.lower() for keyword in ["password reset", "restablecer contraseña", "réinitialiser mot de passe"]):
        if requested_lang == "es":
            return "Puede restablecer su contraseña visitando el enlace 'Olvidé mi contraseña' en la página de inicio de sesión e ingresando su correo electrónico."
        elif requested_lang == "fr":
            return "Vous pouvez réinitialiser votre mot de passe en visitant le lien 'Mot de passe oublié' sur la page de connexion et en entrant votre adresse e-mail."
        else:
            return "You can reset your password by visiting the 'Forgot Password' link on the login page and entering your email address."
            
    elif any(keyword in prompt.lower() for keyword in ["order issue", "problema con mi pedido", "problème avec ma commande"]):
        if requested_lang == "es":
            return "Para ayudarle con su pedido, por favor proporcione su número de pedido."
        elif requested_lang == "fr":
            return "Pour vous aider avec votre commande, veuillez fournir votre numéro de commande."
        else:
            return "Please provide your order number so we can assist you with your order issue."

    if requested_lang == "es":
        return "Lo siento, no tengo suficiente información para responder a eso en español. ¿Podría reformular su pregunta?"
    elif requested_lang == "fr":
        return "Désolé, je n'ai pas assez d'informations pour répondre à cela en français. Pourriez-vous reformuler votre question ?"
    else:
        return "I apologize, I don't have enough information to respond to that. Could you rephrase your question?"

class InCLT_Prompt_Generator:
    def __init__(self):
        self.cross_lingual_example_sets = [
            {
                "en": {"query": "How do I reset my password?", "answer": "You can reset your password by visiting the 'Forgot Password' link on the login page and entering your email address."},
                "es": {"query": "¿Cómo restablezco mi contraseña?", "answer": "Puede restablecer su contraseña visitando el enlace 'Olvidé mi contraseña' en la página de inicio de sesión e ingresando su correo electrónico."},
                "fr": {"query": "Comment réinitialiser mon mot de passe ?", "answer": "Vous pouvez réinitialiser votre mot de passe en visitant le lien 'Mot de passe oublié' sur la page de connexion et en entrant votre adresse e-mail."}
            },
            {
                "en": {"query": "What is your refund policy?", "answer": "Our refund policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging."},
                "es": {"query": "¿Cuál es su política de reembolso?", "answer": "Nuestra política de reembolso permite devoluciones dentro de los 30 días posteriores a la compra, siempre que el artículo esté sin usar y en su embalaje original."},
                "fr": {"query": "Quelle est votre politique de remboursement ?", "answer": "Notre politique de remboursement autorise les retours dans les 30 jours suivant l'achat, à condition que l'article soit inutilisé et dans son emballage d'origine."}
            },
            {
                "en": {"query": "I have an issue with my order.", "answer": "Please provide your order number so we can assist you with your order issue."},
                "es": {"query": "Tengo un problema con mi pedido.", "answer": "Para ayudarle con su pedido, por favor proporcione su número de pedido."},
                "fr": {"query": "J'ai un problème avec ma commande.", "answer": "Pour vous aider avec votre commande, veuillez fournir votre numéro de commande."}
            }
        ]

    def _get_relevant_example_sets(self, query_lang: str) -> list:
        return self.cross_lingual_example_sets

    def generate_prompt(self, user_query: str, query_lang: str, target_lang_for_response: str) -> str:
        prompt_parts = ["You are a multilingual customer support chatbot. Respond concisely and helpfully in the requested language."]
        prompt_parts.append("\nHere are some examples of customer interactions in various languages to help you understand and respond cross-lingually:")
        
        examples_to_include = self._get_relevant_example_sets(query_lang)

        for example_set in examples_to_include:
            prompt_parts.append("-" * 30)
            for lang, data in example_set.items():
                prompt_parts.append(f"Query ({lang}): {data['query']}")
                prompt_parts.append(f"Answer ({lang}): {data['answer']}")
        prompt_parts.append("-" * 30)

        prompt_parts.append(f"Customer Query ({query_lang}): {user_query}")
        prompt_parts.append(f"Response ({target_lang_for_response}):")

        return "\n".join(prompt_parts)

class Query_Language_Detector:
    def detect(self, text: str) -> str:
        try:
            detected_lang = langdetect.detect(text)
            if detected_lang == "en": return "en"
            if detected_lang == "es": return "es"
            if detected_lang == "fr": return "fr"
            return "en"
        except langdetect.lang_detect_exception.LangDetectException:
            return "en"

class CustomerSupportChatbot:
    def __init__(self):
        self.language_detector = Query_Language_Detector()
        self.prompt_generator = InCLT_Prompt_Generator()

    def get_response(self, user_query: str) -> str:
        query_lang = self.language_detector.detect(user_query)
        target_lang_for_response = query_lang

        prompt = self.prompt_generator.generate_prompt(user_query, query_lang, target_lang_for_response)
        
        llm_output = simulated_llm_response(prompt)
        
        return llm_output

def run_chatbot_interface():
    chatbot = CustomerSupportChatbot()
    print("Welcome to the Multilingual Customer Support Chatbot (InCLT Crosslingual Transfer Prompting)!")
    print("Languages supported for explicit examples: English (en), Spanish (es), French (fr).")
    print("Type 'exit' to quit.")
    print("-" * 60)

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")
        print("-" * 60)

if __name__ == "__main__":
    run_chatbot_interface()