from langdetect import detect, DetectorFactory, LangDetectException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DetectorFactory.seed = 0 # for reproducible language detection

# Predefined ICL examples for different source languages to target English
icl_examples = {
    "es": [
        {"input_src": "Mi internet no funciona.", "input_tgt": "My internet is not working.", "output_src": "Por favor, reinicia tu router.", "output_tgt": "Please restart your router."}, # Spanish
        {"input_src": "¿Cómo puedo cambiar mi plan?", "input_tgt": "How can I change my plan?", "output_src": "Puedes cambiar tu plan en nuestra página web o llamando a soporte.", "output_tgt": "You can change your plan on our website or by calling support."}
    ],
    "fr": [
        {"input_src": "Mon compte est bloqué.", "input_tgt": "My account is blocked.", "output_src": "Veuillez vérifier votre email pour les instructions de réinitialisation.", "output_tgt": "Please check your email for reset instructions."}, # French
        {"input_src": "Je n'arrive pas à me connecter.", "input_tgt": "I cannot log in.", "output_src": "Assurez-vous d'utiliser le bon nom d'utilisateur et mot de passe.", "output_tgt": "Make sure you are using the correct username and password."}
    ],
    "de": [
        {"input_src": "Ich habe mein Passwort vergessen.", "input_tgt": "I forgot my password.", "output_src": "Sie können es auf unserer Website zurücksetzen.", "output_tgt": "You can reset it on our website."}, # German
        {"input_src": "Mein Gerät ist kaputt.", "input_tgt": "My device is broken.", "output_src": "Bitte kontaktieren Sie unseren technischen Support.", "output_tgt": "Please contact our technical support."}
    ]
}

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def generate_icl_prompt(query: str, detected_lang: str, target_lang: str = "en") -> str:
    prompt_parts = []
    if detected_lang in icl_examples:
        for example in icl_examples[detected_lang]:
            # Source language example
            prompt_parts.append(f"Customer ({detected_lang}): {example['input_src']}")
            prompt_parts.append(f"Support ({detected_lang}): {example['output_src']}")
            # Target language example (for cross-lingual transfer)
            prompt_parts.append(f"Customer ({target_lang}): {example['input_tgt']}")
            prompt_parts.append(f"Support ({target_lang}): {example['output_tgt']}")

    # Add the actual query
    prompt_parts.append(f"Customer ({detected_lang}): {query}")
    prompt_parts.append(f"Support ({target_lang}):") # The LLM will complete this

    return "\n".join(prompt_parts)


# Load pre-trained T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

def get_llm_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()


def chatbot_orchestrator():
    print("Cross-lingual Customer Support Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        detected_lang = detect_language(user_input)
        print(f"(Detected language: {detected_lang})")

        if detected_lang == "unknown":
            print("Bot: Sorry, I couldn't detect the language. Please try again or rephrase.")
            continue

        prompt = generate_icl_prompt(user_input, detected_lang)
        llm_response = get_llm_response(prompt)
        print(f"Bot: {llm_response}")

if __name__ == "__main__":
    chatbot_orchestrator()
