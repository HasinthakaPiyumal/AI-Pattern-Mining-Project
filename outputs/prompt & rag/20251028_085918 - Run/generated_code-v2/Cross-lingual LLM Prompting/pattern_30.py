from transformers import pipeline

# 1. Language Detection (Placeholder)
def detect_language(text):
    # In a real application, you would use a more robust language detection library
    # For this example, we'll make a simple assumption or use a placeholder.
    if "hola" in text.lower() or "qué tal" in text.lower():
        return "es"
    elif "hello" in text.lower() or "hi" in text.lower():
        return "en"
    elif "bonjour" in text.lower() or "salut" in text.lower():
        return "fr"
    else:
        return "en" # Default to English

# 2. In-Context Example Database (Conceptual)
in_context_examples = {
    "en-es": [
        {"source": "What is the return policy?", "target": "¿Cuál es la política de devoluciones?"},
        {"source": "How can I track my order?", "target": "¿Cómo puedo rastrear mi pedido?"}
    ],
    "es-en": [
        {"source": "¿Necesito una cuenta para comprar?", "target": "Do I need an account to purchase?"},
        {"source": "¿Pueden enviar a Argentina?", "target": "Can you ship to Argentina?"}
    ],
    "en-fr": [
        {"source": "Where is my parcel?", "target": "Où est mon colis ?"},
        {"source": "Do you have this in blue?", "target": "L'avez-vous en bleu ?"}
    ],
    "fr-en": [
        {"source": "Je voudrais changer la taille.", "target": "I would like to change the size."},
        {"source": "Est-ce que c'est en stock ?", "target": "Is it in stock?"}
    ]
}

class PromptEngineer:
    def __init__(self, examples=None):
        self.examples = examples if examples is not None else {}

    def _get_icl_examples(self, source_lang, target_lang, task_type="translation"):
        key = f"{source_lang}-{target_lang}"
        if key in self.examples:
            return self.examples[key]
        # Fallback if specific pair not found, maybe reverse or default
        print(f"Warning: No specific examples for {key}. Using general examples if available.")
        return []

    def construct_prompt(self, user_query, source_lang, target_lang, task_description="Translate the following customer query:"):
        icl_examples = self._get_icl_examples(source_lang, target_lang)
        prompt_parts = []

        if icl_examples:
            prompt_parts.append("Here are some examples of previous interactions:")
            for ex in icl_examples:
                prompt_parts.append(f"Source ({source_lang}): {ex['source']}")
                prompt_parts.append(f"Target ({target_lang}): {ex['target']}")
            prompt_parts.append("\nNow, complete the following task:")

        prompt_parts.append(f"{task_description}")
        prompt_parts.append(f"Source ({source_lang}): {user_query}")
        prompt_parts.append(f"Target ({target_lang}):")

        return "\n".join(prompt_parts)

# 3. Core LLM (Conceptual/Mock using a simple text2text generation pipeline)
# In a real scenario, you'd load a more powerful multilingual model.
# For demonstration, we'll use a small, fast model or a mock.

try:
    # Try to load a suitable model for demonstration
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
    def mock_llm_response(prompt):
        # This is a very basic mock. A real LLM would be more context-aware.
        # For cross-lingual transfer, we expect the LLM to understand the structure
        # introduced by the ICL examples.
        if "Target (es):" in prompt:
            text_to_translate = prompt.split("Source (en): ")[-1].split("\nTarget (es):")[-2].strip()
            result = translator(text_to_translate)
            return result[0]["translation_text"]
        elif "Target (en):" in prompt and "Source (es):" in prompt:
            text_to_translate = prompt.split("Source (es): ")[-1].split("\nTarget (en):")[-2].strip()
            # We need an es-en translator, let's create one or mock better
            es_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
            result = es_en_translator(text_to_translate)
            return result[0]["translation_text"]
        else:
            return "I am unable to process this request in the current language context."
except Exception as e:
    print(f"Warning: Could not load Hugging Face translation pipeline for demo. Using a simple string manipulation mock. Error: {e}")
    def mock_llm_response(prompt):
        # Fallback simple mock if model loading fails
        if "Source (en): What is the return policy?" in prompt and "Target (es):" in prompt:
            return "¿Cuál es la política de devoluciones?"
        elif "Source (es): ¿Pueden enviar a Argentina?" in prompt and "Target (en):" in prompt:
            return "Can you ship to Argentina?"
        elif "Target (" in prompt:
            return f"[MOCK LLM RESPONSE]: Processed query based on prompt. Output for {prompt.split('Target (')[-1].split('):')[0]} would be here."
        return "[MOCK LLM RESPONSE]: I received your query."


# 4. Chatbot Interface
def customer_support_chatbot(user_query, target_language="en"):
    source_language = detect_language(user_query)
    print(f"Detected source language: {source_language}")

    # Initialize Prompt Engineer with our conceptual examples
    prompt_engineer = PromptEngineer(in_context_examples)

    # Construct the prompt using InCLT Crosslingual Transfer Prompting
    # The target language for the LLM's response is the target_language parameter
    # but we leverage both source and target in examples.
    prompt = prompt_engineer.construct_prompt(user_query, source_language, target_language, task_description=f"Translate the following customer query from {source_language} to {target_language} and provide a helpful response:")

    print(f"\n--- Constructed Prompt ---\n{prompt}\n--------------------------")

    # Get response from the LLM (mocked or actual pipeline)
    llm_response = mock_llm_response(prompt)

    # For a real chatbot, you'd parse the LLM response to extract the actual answer
    # and then format it for the user. Here, we'll just return the LLM's output.
    return llm_response


if __name__ == "__main__":
    print("\n--- Chatbot Test Cases ---")

    # Test Case 1: English query, expect Spanish translation and response (conceptual)
    print("\nCustomer (EN): I need help with my recent order.")
    response1 = customer_support_chatbot("I need help with my recent order.", target_language="es")
    print(f"Chatbot (ES): {response1}")

    # Test Case 2: Spanish query, expect English translation and response (conceptual)
    print("\nCustomer (ES): ¿Puedo devolver este producto?")
    response2 = customer_support_chatbot("¿Puedo devolver este producto?", target_language="en")
    print(f"Chatbot (EN): {response2}")

    # Test Case 3: French query, expect English translation and response (conceptual)
    print("\nCustomer (FR): Où est mon colis ?")
    response3 = customer_support_chatbot("Où est mon colis ?", target_language="en")
    print(f"Chatbot (EN): {response3}")

    # Test Case 4: English query with specific product question, expect Spanish
    print("\nCustomer (EN): Do you have the new smartphone in stock?")
    response4 = customer_support_chatbot("Do you have the new smartphone in stock?", target_language="es")
    print(f"Chatbot (ES): {response4}")

    # Test Case 5: Spanish query about shipping, expect English
    print("\nCustomer (ES): ¿Cuánto tiempo tarda el envío a México?")
    response5 = customer_support_chatbot("¿Cuánto tiempo tarda el envío a México?", target_language="en")
    print(f"Chatbot (EN): {response5}")

    # Test Case 6: A query not explicitly in examples, to show generalization (conceptual)
    print("\nCustomer (EN): What are your store hours?")
    response6 = customer_support_chatbot("What are your store hours?", target_language="es")
    print(f"Chatbot (ES): {response6}")

    print("\n--- End of Chatbot Test Cases ---")
