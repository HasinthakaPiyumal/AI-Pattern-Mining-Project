class InCLT_Prompt_Generator:
    def __init__(self, examples_en, examples_cross_lingual):
        self.examples_en = examples_en  # Examples in the pivot language (English)
        self.examples_cross_lingual = examples_cross_lingual  # Cross-lingual examples

    def _format_examples(self, examples):
        formatted_str = ""
        for ex in examples:
            formatted_str += f"Query: {ex['query']}\n"
            formatted_str += f"Response: {ex['response']}\n\n"
        return formatted_str

    def generate_prompt(self, user_query, source_lang, target_lang):
        # General English examples for cross-lingual transfer
        en_examples_str = self._format_examples(self.examples_en)

        # Cross-lingual examples specific to the source-target pair if available
        cross_lingual_specific_examples = []
        for ex in self.examples_cross_lingual:
            if ex['source_lang'] == source_lang and ex['target_lang'] == target_lang:
                cross_lingual_specific_examples.append({
                    'query': ex['query_source'],
                    'response': ex['response_target']
                })
        cross_lingual_examples_str = self._format_examples(cross_lingual_specific_examples)

        # Construct the final prompt
        prompt = f"""You are a helpful multilingual customer support assistant.

Here are some examples to guide your responses (English pivot):
{en_examples_str}
Here are some cross-lingual examples ({source_lang} to {target_lang}):
{cross_lingual_examples_str}

Based on the examples, please provide a concise response to the following query in {target_lang}:
Query: {user_query}
Response:"""
        return prompt

def simulate_llm_response(prompt):
    # This function simulates an LLM's response based on the prompt.
    # In a real application, this would involve calling an actual LLM API.
    print("\n--- Simulated LLM Input Prompt ---")
    print(prompt)
    print("--- End Simulated LLM Input ---\n")

    # Simple keyword-based simulation for demonstration
    if "delivery status" in prompt.lower() and "Spanish" in prompt:
        return "El estado de su pedido es en tránsito y se espera que llegue en 2-3 días hábiles."
    elif "return policy" in prompt.lower() and "French" in prompt:
        return "Notre politique de retour vous permet de retourner les articles dans les 30 jours suivant l'achat avec le reçu original."
    elif "password reset" in prompt.lower():
        return "Please visit our website and click on 'Forgot Password' to reset your password."
    else:
        return "Thank you for contacting us. We will get back to you shortly."

# --- Demonstration ---

# Hardcoded examples for the InCLT pattern
en_examples = [
    {"query": "What is your return policy?", "response": "Our return policy allows returns within 30 days with original receipt."},
    {"query": "How can I track my order?", "response": "You can track your order using the tracking number provided in your shipping confirmation email."}
]

cross_lingual_examples = [
    {"source_lang": "Spanish", "target_lang": "English", "query_source": "¿Cuál es el estado de mi pedido?", "response_target": "Your order is currently in transit."},
    {"source_lang": "French", "target_lang": "English", "query_source": "Je voudrais réinitialiser mon mot de passe.", "response_target": "You can reset your password on our website."},
    {"source_lang": "Spanish", "target_lang": "Spanish", "query_source": "¿Puedo cambiar mi dirección de envío?", "response_target": "Sí, puede cambiar su dirección de envío antes de que el pedido sea enviado."}
]

# Initialize the prompt generator
prompt_generator = InCLT_Prompt_Generator(en_examples, cross_lingual_examples)

# Scenario 1: Spanish query, expecting Spanish response
user_query_es = "¿Cuál es el estado de mi envío?"
source_lang_es = "Spanish"
target_lang_es = "Spanish"
prompt_es = prompt_generator.generate_prompt(user_query_es, source_lang_es, target_lang_es)
response_es = simulate_llm_response(prompt_es)
print(f"\nUser Query ({source_lang_es}): {user_query_es}")
print(f"Chatbot Response ({target_lang_es}): {response_es}")

# Scenario 2: French query, expecting English response
user_query_fr = "J'ai une question sur ma commande."
source_lang_fr = "French"
target_lang_fr = "English"
prompt_fr = prompt_generator.generate_prompt(user_query_fr, source_lang_fr, target_lang_fr)
response_fr = simulate_llm_response(prompt_fr)
print(f"\nUser Query ({source_lang_fr}): {user_query_fr}")
print(f"Chatbot Response ({target_lang_fr}): {response_fr}")

# Scenario 3: English query, expecting English response (standard ICL)
user_query_en = "How do I reset my account password?"
source_lang_en = "English"
target_lang_en = "English"
prompt_en = prompt_generator.generate_prompt(user_query_en, source_lang_en, target_lang_en)
response_en = simulate_llm_response(prompt_en)
print(f"\nUser Query ({source_lang_en}): {user_query_en}")
print(f"Chatbot Response ({target_lang_en}): {response_en}")