class CrosslingualExampleDatabase:
    def __init__(self):
        self.examples = {
            "product_return": [
                ("I want to return a product. How can I do that?", "Quiero devolver un producto. ¿Cómo puedo hacerlo?"),
                ("To return an item, please visit our returns page.", "Para devolver un artículo, visita nuestra página de devoluciones."),
            ],
            "shipping_info": [
                ("What is the shipping cost?", "¿Cuál es el costo de envío?"),
                ("Shipping costs vary based on your location and chosen method.", "Los costos de envío varían según tu ubicación y método elegido."),
            ],
            "account_reset": [
                ("How do I reset my password?", "¿Cómo restablezco mi contraseña?"),
                ("You can reset your password from the login page by clicking 'Forgot Password'.", "Puedes restablecer tu contraseña desde la página de inicio de sesión haciendo clic en 'Olvidé mi contraseña'."),
            ]
        }

    def get_examples(self, topic, num_examples=1):
        return self.examples.get(topic, [])[:num_examples]

class PromptEngineeringModule:
    def __init__(self, example_database):
        self.example_database = example_database

    def construct_prompt(self, user_query, target_language, topic=None):
        prompt_parts = ["You are a helpful multilingual customer support assistant."]

        if topic:
            cross_lingual_examples = self.example_database.get_examples(topic, num_examples=2)
            if cross_lingual_examples:
                prompt_parts.append("\\nHere are some examples of similar questions and answers in both English and {}:".format(target_language))
                for src_ex, tgt_ex in cross_lingual_examples:
                    prompt_parts.append(f"English: {src_ex}")
                    prompt_parts.append(f"{target_language}: {tgt_ex}")

        prompt_parts.append(f"\\nUser Query ({target_language}): {user_query}")
        prompt_parts.append(f"\\nAssistant Response ({target_language}):")

        return "\n".join(prompt_parts)

def simulate_multilingual_llm(prompt):
    # This is a simulated LLM response.
    # In a real application, this would involve calling an actual LLM API
    # and processing its output.
    if "devolver un producto" in prompt.lower() or "return a product" in prompt.lower():
        return "Para devolver un producto, por favor visita nuestra página de devoluciones en el sitio web."
    elif "costo de envío" in prompt.lower() or "shipping cost" in prompt.lower():
        return "El costo de envío depende de tu ubicación y el método de envío seleccionado."
    elif "restablecer mi contraseña" in prompt.lower() or "reset my password" in prompt.lower():
        return "Puedes restablecer tu contraseña desde la página de inicio de sesión. Haz clic en '¿Olvidaste tu contraseña?'."
    else:
        return "Lo siento, no tengo suficiente información para responder a eso. ¿Podrías ser más específico?"

# --- Main Chatbot Logic ---
if __name__ == "__main__":
    example_db = CrosslingualExampleDatabase()
    prompt_engineer = PromptEngineeringModule(example_db)

    print("\n--- Chatbot Simulation ---")

    # Scenario 1: Product return in Spanish
    user_query_1 = "Quiero devolver un producto."
    target_lang_1 = "Spanish"
    topic_1 = "product_return"
    print(f"\nUser: {user_query_1} (Language: {target_lang_1})")
    prompt_1 = prompt_engineer.construct_prompt(user_query_1, target_lang_1, topic_1)
    # print(f"\nConstructed Prompt:\n{prompt_1}") # Uncomment to see the full prompt
    response_1 = simulate_multilingual_llm(prompt_1)
    print(f"Chatbot: {response_1}")

    # Scenario 2: Shipping info in English
    user_query_2 = "What is the shipping cost?"
    target_lang_2 = "English"
    topic_2 = "shipping_info"
    print(f"\nUser: {user_query_2} (Language: {target_lang_2})")
    prompt_2 = prompt_engineer.construct_prompt(user_query_2, target_lang_2, topic_2)
    # print(f"\nConstructed Prompt:\n{prompt_2}") # Uncomment to see the full prompt
    response_2 = simulate_multilingual_llm(prompt_2)
    print(f"Chatbot: {response_2}")

    # Scenario 3: Password reset in Spanish (with a slightly different query)
    user_query_3 = "¿No puedo acceder a mi cuenta, cómo restablezco mi contraseña?"
    target_lang_3 = "Spanish"
    topic_3 = "account_reset"
    print(f"\nUser: {user_query_3} (Language: {target_lang_3})")
    prompt_3 = prompt_engineer.construct_prompt(user_query_3, target_lang_3, topic_3)
    # print(f"\nConstructed Prompt:\n{prompt_3}") # Uncomment to see the full prompt
    response_3 = simulate_multilingual_llm(prompt_3)
    print(f"Chatbot: {response_3}")

    # Scenario 4: Unknown query in French (without a specific topic for ICL here, though ICL could be applied generally)
    user_query_4 = "Je veux savoir l'heure d'ouverture."
    target_lang_4 = "French"
    topic_4 = None # No specific topic to pull ICL examples from for this one
    print(f"\nUser: {user_query_4} (Language: {target_lang_4})")
    prompt_4 = prompt_engineer.construct_prompt(user_query_4, target_lang_4, topic_4)
    # print(f"\nConstructed Prompt:\n{prompt_4}") # Uncomment to see the full prompt
    response_4 = simulate_multilingual_llm(prompt_4)
    print(f"Chatbot: {response_4}")

    print("\n--- End of Simulation ---")
