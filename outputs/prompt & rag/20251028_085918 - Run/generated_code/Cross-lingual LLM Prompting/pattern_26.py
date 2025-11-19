from langdetect import detect, DetectorFactory
from transformers import pipeline

DetectorFactory.seed = 0 # for consistent language detection results

class InContextLearner:
    def __init__(self):
        self.examples = []

    def add_example(self, source_lang, target_lang, source_query, target_query, source_answer, target_answer):
        self.examples.append({
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_query": source_query,
            "target_query": target_query,
            "source_answer": source_answer,
            "target_answer": target_answer,
        })

    def get_relevant_examples(self, query_lang, target_lang, user_query):
        # For simplicity, returning all examples. In a real system, this would involve semantic search.
        return self.examples

    def generate_inclt_prompt(self, user_query, source_lang, target_lang, examples):
        prompt_parts = []
        for ex in examples:
            if ex["source_lang"] == source_lang and ex["target_lang"] == target_lang:
                prompt_parts.append(f"{ex["source_query"]} -> {ex["target_query"]}")
                prompt_parts.append(f"{ex["source_answer"]} -> {ex["target_answer"]}\n")

        prompt_parts.append(f"User Query ({source_lang}): {user_query}")
        prompt_parts.append(f"Response ({target_lang}):")
        return "\n".join(prompt_parts)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Simulated LLM using Hugging Face transformers pipeline
# Using a small, general-purpose model for demonstration. Replace with a more suitable multilingual LLM if available.
simulated_llm_pipeline = pipeline("text-generation", model="distilgpt2")

def get_llm_response(prompt):
    # For demonstration, we'll just take the first generated sequence and clean it up.
    # In a real scenario, you'd integrate with a proper multilingual LLM API.
    response = simulated_llm_pipeline(prompt, max_new_tokens=50, num_return_sequences=1, do_sample=True, temperature=0.7)
    generated_text = response[0]["generated_text"]
    # Remove the prompt itself from the generated text
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].strip()
    return generated_text.strip()

# Simulated Multilingual Knowledge Base (for context, not direct lookup by chatbot logic)
simulated_kb = {
    "en": {
        "hello": "Hello! How can I assist you today?",
        "refund": "For a refund, please provide your order number.",
        "shipping": "Shipping typically takes 3-5 business days."
    },
    "es": {
        "hello": "¡Hola! ¿En qué puedo ayudarte hoy?",
        "refund": "Para un reembolso, por favor proporcione su número de pedido.",
        "shipping": "El envío suele tardar de 3 a 5 días hábiles."
    },
    "fr": {
        "hello": "Bonjour! Comment puis-je vous aider aujourd'hui?",
        "refund": "Pour un remboursement, veuillez fournir votre numéro de commande.",
        "shipping": "L'expédition prend généralement 3 à 5 jours ouvrables."
    }
}

if __name__ == "__main__":
    print("Initializing Multilingual Customer Support Chatbot with InCLT...")
    inclt_learner = InContextLearner()

    # Add cross-lingual in-context examples
    inclt_learner.add_example(
        source_lang="en", target_lang="es",
        source_query="What is your return policy?", target_query="¿Cuál es su política de devoluciones?",
        source_answer="Our return policy allows returns within 30 days of purchase.", target_answer="Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra."
    )
    inclt_learner.add_example(
        source_lang="es", target_lang="en",
        source_query="Necesito ayuda con mi pedido.", target_query="I need help with my order.",
        source_answer="Por favor, proporcione su número de pedido para que podamos ayudarle.", target_answer="Please provide your order number so we can assist you."
    )
    inclt_learner.add_example(
        source_lang="fr", target_lang="en",
        source_query="Comment puis-je contacter le support technique?", target_query="How can I contact technical support?",
        source_answer="Vous pouvez contacter le support technique via notre page de contact ou par téléphone.", target_answer="You can contact technical support via our contact page or by phone."
    )
    inclt_learner.add_example(
        source_lang="en", target_lang="fr",
        source_query="Where is my package?", target_query="Où est mon colis ?",
        source_answer="Please provide your tracking number to check the status.", target_answer="Veuillez fournir votre numéro de suivi pour vérifier le statut."
    )

    print("Chatbot ready. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        detected_lang = detect_language(user_input)
        print(f"Detected language: {detected_lang}")

        # For demonstration, let's always target English for internal processing/response generation
        # In a real app, target_lang could be dynamic or user-selected.
        target_lang = "en"

        relevant_examples = inclt_learner.get_relevant_examples(detected_lang, target_lang, user_input)
        inclt_prompt = inclt_learner.generate_inclt_prompt(user_input, detected_lang, target_lang, relevant_examples)

        print("--- Generated Prompt ---")
        print(inclt_prompt)
        print("------------------------")

        llm_response = get_llm_response(inclt_prompt)
        print(f"Chatbot ({target_lang}): {llm_response}")

    print("Chatbot session ended.")
