from transformers import pipeline
from langdetect import detect, DetectorFactory
import gradio as gr

DetectorFactory.seed = 0

# 3. InCLT Prompt Builder Module
IN_CONTEXT_EXAMPLES = [
    {
        "source_lang": "en",
        "target_lang": "es",
        "query_en": "How can I reset my password?",
        "answer_en": "You can reset your password by clicking on 'Forgot Password' on the login page.",
        "query_es": "¿Cómo puedo restablecer mi contraseña?",
        "answer_es": "Puede restablecer su contraseña haciendo clic en 'Olvidé mi contraseña' en la página de inicio de sesión."
    },
    {
        "source_lang": "en",
        "target_lang": "fr",
        "query_en": "What are your operating hours?",
        "answer_en": "Our operating hours are Monday to Friday, 9 AM to 5 PM EST.",
        "query_fr": "¿Cuáles son sus horarios de atención?",
        "answer_fr": "Nuestro horario de atención es de lunes a viernes, de 9 a.m. a 5 p.m. EST."
    },
    {
        "source_lang": "es",
        "target_lang": "en",
        "query_es": "¿Puedo cambiar mi dirección de envío?",
        "answer_es": "Sí, puedes cambiar tu dirección de envío en la sección de 'Mi Cuenta'.",
        "query_en": "Can I change my shipping address?",
        "answer_en": "Yes, you can change your shipping address in the 'My Account' section."
    }
]

def build_prompt(user_query, detected_lang, target_lang):
    prompt_parts = []
    for example in IN_CONTEXT_EXAMPLES:
        if (example["source_lang"] == detected_lang and example["target_lang"] == target_lang) or \
           (example["target_lang"] == detected_lang and example["source_lang"] == target_lang):
            prompt_parts.append(f"Source ({example['source_lang']}): Q: {example[f'query_{example['source_lang']}']} A: {example[f'answer_{example['source_lang']}']}")
            prompt_parts.append(f"Target ({example['target_lang']}): Q: {example[f'query_{example['target_lang']}']} A: {example[f'answer_{example['target_lang']}']}")

    # Add a general example if no specific cross-lingual match for simplicity
    if not prompt_parts and detected_lang == 'en': # Fallback for English queries
        prompt_parts.append(f"Q: How to contact support? A: You can reach support via email or phone.")
    elif not prompt_parts and detected_lang == 'es': # Fallback for Spanish queries
        prompt_parts.append(f"Q: ¿Cómo contactar soporte? A: Puedes contactar soporte por email o teléfono.")
    
    prompt_parts.append(f"User ({detected_lang}): Q: {user_query} A:")
    return "\n".join(prompt_parts)

# 4. MultilingualChatbot Class
class MultilingualChatbot:
    def __init__(self):
        self.pipeline = pipeline("text2text-generation", model="google/mt5-small")

    def get_response(self, user_input):
        try:
            detected_lang = detect(user_input)
        except:
            detected_lang = "en" # Default to English if detection fails
        
        # For simplicity, we'll try to respond in the detected language or English as a fallback
        target_lang = detected_lang if detected_lang in ['en', 'es', 'fr'] else 'en'

        prompt = build_prompt(user_input, detected_lang, target_lang)
        
        # The pipeline returns a list of dictionaries, extract the 'generated_text'
        response = self.pipeline(prompt, max_new_tokens=50)[0]['generated_text']
        return response

# 5. Gradio User Interface
chatbot = MultilingualChatbot()

def chatbot_interface(message):
    return chatbot.get_response(message)

if __name__ == "__main__":
    gr.Interface(
        fn=chatbot_interface,
        inputs=gr.Textbox(lines=2, placeholder="Ask me anything..."),
        outputs="text",
        title="Multilingual Customer Support Chatbot (InCLT Prompting)",
        description="This chatbot uses InCLT Crosslingual Transfer Prompting to answer queries in multiple languages. Try asking in English, Spanish, or French!"
    ).launch()