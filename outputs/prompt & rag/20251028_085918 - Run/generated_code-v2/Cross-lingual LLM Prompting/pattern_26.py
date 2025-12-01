
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0 # for reproducible results

# 1. Simulated Parallel Data Store
# Each entry contains a query and answer in multiple languages, with English as the pivot/source.
parallel_data_store = {
    "account_balance": {
        "en": {"query": "What is my account balance?", "answer": "Your current account balance is $1,250.75."}, 
        "sw": {"query": "Mizani ya akaunti yangu ni ngapi?", "answer": "Salio la akaunti yako ni $1,250.75."}, # Swahili
        "fr": {"query": "Quel est le solde de mon compte?", "answer": "Le solde actuel de votre compte est de 1 250,75 $."}
    },
    "password_reset": {
        "en": {"query": "How do I reset my password?", "answer": "You can reset your password by visiting our website and clicking 'Forgot Password'."},
        "sw": {"query": "Ninawezaje kuweka upya nenosiri langu?", "answer": "Unaweza kuweka upya nenosiri lako kwa kutembelea tovuti yetu na kubofya 'Umesahau Nenosiri'."},
        "fr": {"query": "Comment réinitialiser mon mot de passe?", "answer": "Vous pouvez réinitialiser votre mot de passe en visitant notre site Web et en cliquant sur 'Mot de passe oublié'."}
    },
    "contact_support": {
        "en": {"query": "How can I contact support?", "answer": "You can contact support via phone at 1-800-555-0123 or email us at support@example.com."}, 
        "sw": {"query": "Ninawezaje kuwasiliana na huduma kwa wateja?", "answer": "Unaweza kuwasiliana na usaidizi kupitia simu kwa 1-800-555-0123 au tutumie barua pepe kwa support@example.com."}, # Swahili
        "fr": {"query": "Comment puis-je contacter le support?", "answer": "Vous pouvez contacter le support par téléphone au 1-800-555-0123 ou nous envoyer un courriel à support@example.com."}
    }
}

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en" # Default to English if detection fails

def get_in_context_examples(target_lang: str, num_examples: int = 2, source_lang: str = "en") -> list:
    examples = []
    keys = list(parallel_data_store.keys())
    
    for i in range(min(num_examples, len(keys))):
        key = keys[i]
        data = parallel_data_store[key]
        
        # Ensure both source and target languages are available for the example
        if source_lang in data and target_lang in data:
            example_query_src = data[source_lang]["query"]
            example_answer_src = data[source_lang]["answer"]
            example_query_tgt = data[target_lang]["query"]
            example_answer_tgt = data[target_lang]["answer"]
            
            # Format as a cross-lingual ICL example (source -> target)
            examples.append(f"Customer ({source_lang}): {example_query_src}\nAgent ({source_lang}): {example_answer_src}\nCustomer ({target_lang}): {example_query_tgt}\nAgent ({target_lang}): {example_answer_tgt}")
        elif target_lang in data: # Fallback: if source not available, just use target
             example_query_tgt = data[target_lang]["query"]
             example_answer_tgt = data[target_lang]["answer"]
             examples.append(f"Customer ({target_lang}): {example_query_tgt}\nAgent ({target_lang}): {example_answer_tgt}")

    return examples

def generate_icl_prompt(customer_query: str, target_lang: str, num_icl_examples: int = 2) -> str:
    icl_examples = get_in_context_examples(target_lang, num_icl_examples)
    
    prompt_parts = []
    prompt_parts.append("You are a helpful customer support agent. Provide concise and accurate answers.")
    prompt_parts.append("\\nHere are some examples of customer interactions to guide you:")
    prompt_parts.extend(icl_examples)
    prompt_parts.append(f"\\nCustomer ({target_lang}): {customer_query}\nAgent ({target_lang}):")
    
    return "\n".join(prompt_parts)

def simulate_llm_response(prompt: str) -> str:
    # This function simulates an LLM's behavior based on the prompt.
    # In a real application, this would involve calling an actual LLM API.
    
    # Simple keyword-based simulation for demonstration
    if "account balance" in prompt.lower() or "mizani ya akaunti" in prompt.lower() or "solde de mon compte" in prompt.lower():
        return "Your account balance information can be found in your online portal or mobile app."
    elif "reset password" in prompt.lower() or "weka upya nenosiri" in prompt.lower() or "réinitialiser mon mot de passe" in prompt.lower():
        return "To reset your password, please visit our 'Forgot Password' page on the website."
    elif "contact support" in prompt.lower() or "huduma kwa wateja" in prompt.lower() or "contacter le support" in prompt.lower():
        return "You can reach our support team by calling 1-800-555-0123 or emailing us."
    else:
        return "I'm sorry, I couldn't find a direct answer to your question. Please provide more details or try rephrasing."


def main():
    print("Welcome to the Multilingual Customer Support Chatbot with InCLT!")
    print("\n--- Demonstrating with a Swahili query ---")
    swahili_query = "Ninawezaje kuweka upya nenosiri langu?"
    detected_lang_sw = detect_language(swahili_query)
    print(f"Customer Query (Swahili): {swahili_query}")
    print(f"Detected Language: {detected_lang_sw}")
    
    icl_prompt_sw = generate_icl_prompt(swahili_query, detected_lang_sw)
    print("\nGenerated InCLT Prompt (Swahili examples):\n---\n" + icl_prompt_sw + "\n---")
    
    llm_response_sw = simulate_llm_response(icl_prompt_sw)
    print(f"Chatbot Response (Swahili): {llm_response_sw}")

    print("\n--- Demonstrating with a French query ---")
    french_query = "Quel est le solde de mon compte?"
    detected_lang_fr = detect_language(french_query)
    print(f"Customer Query (French): {french_query}")
    print(f"Detected Language: {detected_lang_fr}")
    
    icl_prompt_fr = generate_icl_prompt(french_query, detected_lang_fr)
    print("\nGenerated InCLT Prompt (French examples):\n---\n" + icl_prompt_fr + "\n---")
    
    llm_response_fr = simulate_llm_response(icl_prompt_fr)
    print(f"Chatbot Response (French): {llm_response_fr}")

    print("\n--- Demonstrating with an English query ---")
    english_query = "How can I contact support?"
    detected_lang_en = detect_language(english_query)
    print(f"Customer Query (English): {english_query}")
    print(f"Detected Language: {detected_lang_en}")
    
    icl_prompt_en = generate_icl_prompt(english_query, detected_lang_en)
    print("\nGenerated InCLT Prompt (English examples):\n---\n" + icl_prompt_en + "\n---")
    
    llm_response_en = simulate_llm_response(icl_prompt_en)
    print(f"Chatbot Response (English): {llm_response_en}")


if __name__ == "__main__":
    main()
