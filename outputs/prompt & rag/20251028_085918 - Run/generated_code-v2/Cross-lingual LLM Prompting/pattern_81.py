class MultilingualPromptGenerator:
    def __init__(self, multilingual_examples):
        self.multilingual_examples = multilingual_examples

    def construct_inclt_prompt(self, user_query: str, source_language: str, target_transfer_language: str = "en") -> str:
        system_instruction = "[System Instruction: You are a helpful customer support assistant. Answer queries based on the provided context and examples.]\n\n"
        example_section = ""

        for example in self.multilingual_examples:
            # Ensure the example has the required languages
            if source_language in example and target_transfer_language in example:
                example_section += f"Example ({source_language}):\n"
                example_section += f"User: {example[source_language]['query']}\n"
                example_section += f"Assistant: {example[source_language]['response']}\n\n"

                example_section += f"Example ({target_transfer_language}):\n"
                example_section += f"User: {example[target_transfer_language]['query']}\n"
                example_section += f"Assistant: {example[target_transfer_language]['response']}\n\n"
            else:
                # Optionally handle cases where an example doesn't have all needed languages
                pass

        user_query_section = f"User: {user_query}\nAssistant:"

        full_prompt = system_instruction + example_section + user_query_section
        return full_prompt

# --- Simulation of Usage ---

# 1. Multilingual Knowledge Base/Examples Storage
multilingual_knowledge_base = [
    {
        "es": {"query": "¿Cómo puedo restablecer mi contraseña?", "response": "Puedes restablecer tu contraseña visitando la sección de \"Configuración de la cuenta\" y haciendo clic en \"Restablecer contraseña\"."},
        "en": {"query": "How can I reset my password?", "response": "You can reset your password by visiting the \"Account Settings\" section and clicking on \"Reset Password\"."},
        "fr": {"query": "Comment puis-je réinitialiser mon mot de passe ?", "response": "Vous pouvez réinitialiser votre mot de passe en visitant la section \"Paramètres du compte\" et en cliquant sur \"Réinitialiser le mot de passe\"."}
    },
    {
        "es": {"query": "¿Cuál es el estado de mi pedido?", "response": "Para verificar el estado de tu pedido, por favor introduce tu número de pedido en la página de seguimiento de pedidos."},
        "en": {"query": "What is the status of my order?", "response": "To check the status of your order, please enter your order number on the order tracking page."},
        "fr": {"query": "Quel est le statut de ma commande ?", "response": "Pour vérifier le statut de votre commande, veuillez saisir votre numéro de commande sur la page de suivi des commandes."}
    }
]

# 2. Instantiate the prompt generator
prompt_generator = MultilingualPromptGenerator(multilingual_knowledge_base)

# 3. Simulate a user query and language detection
user_query_spanish = "Necesito ayuda con mi cuenta."
source_lang = "es"
target_transfer_lang = "en" # English as the common transfer language

# 4. Construct the InCLT prompt
constructed_prompt = prompt_generator.construct_inclt_prompt(user_query_spanish, source_lang, target_transfer_lang)

# 5. Print the constructed prompt (which would then be sent to an LLM)
print(constructed_prompt)

# Simulate another query in French
user_query_french = "J'ai un problème avec ma livraison."
source_lang_fr = "fr"
constructed_prompt_fr = prompt_generator.construct_inclt_prompt(user_query_french, source_lang_fr, target_transfer_lang)
print("\n" + "-"*50 + "\n")
print(constructed_prompt_fr)