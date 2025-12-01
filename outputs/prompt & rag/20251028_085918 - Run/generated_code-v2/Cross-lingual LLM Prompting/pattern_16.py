from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import torch
from langdetect import detect, DetectorFactory
import gradio as gr

DetectorFactory.seed = 0

class MultilingualChatbot:
    def __init__(self):
        self.model_name = "facebook/mbart-large-50"
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.lang_map = {
            "en": "en_XX",
            "es": "es_XX",
            "fr": "fr_XX",
            "de": "de_DE",
            "it": "it_IT"
        }

        self.lang_tag_map = {
            "en": "EN",
            "es": "ES",
            "fr": "FR",
            "de": "DE",
            "it": "IT"
        }

    def _get_icl_examples(self, num_examples=2):
        examples = [
            {
                "en_query": "What are your operating hours?",
                "en_response": "Our operating hours are Monday to Friday, 9 AM to 5 PM.",
                "es_query": "¿Cuáles son sus horas de operación?",
                "es_response": "Nuestro horario de atención es de lunes a viernes, de 9 AM a 5 PM.",
                "fr_query": "Quelles sont vos heures d'ouverture ?",
                "fr_response": "Nos heures d'ouverture sont du lundi au vendredi, de 9h à 17h."
            },
            {
                "en_query": "How do I reset my password?",
                "en_response": "You can reset your password by clicking on 'Forgot Password' on the login page.",
                "es_query": "¿Cómo restablezco mi contraseña?",
                "es_response": "Puede restablecer su contraseña haciendo clic en 'Olvidé mi contraseña' en la página de inicio de sesión.",
                "fr_query": "Comment réinitialiser mon mot de passe ?",
                "fr_response": "Vous pouvez réinitialiser votre mot de passe en cliquant sur 'Mot de passe oublié' sur la page de connexion."
            },
            {
                "en_query": "Where can I find support?",
                "en_response": "You can find support on our website under the 'Help' section or contact us via email.",
                "es_query": "¿Dónde puedo encontrar soporte?",
                "es_response": "Puede encontrar soporte en nuestro sitio web en la sección 'Ayuda' o contactarnos por correo electrónico.",
                "fr_query": "Où puis-je trouver de l'aide ?",
                "fr_response": "Vous pouvez trouver de l'aide sur notre site web dans la section 'Aide' ou nous contacter par e-mail."
            }
        ]
        return examples[:num_examples]

    def _construct_prompt(self, user_query, user_lang_short, icl_examples):
        prompt_parts = []
        user_lang_tag = self.lang_tag_map.get(user_lang_short, user_lang_short.upper())

        for example in icl_examples:
            for lang_code in ["en", "es", "fr"]:
                query_key = f"{lang_code}_query"
                response_key = f"{lang_code}_response"
                if query_key in example and response_key in example:
                    lang_tag = self.lang_tag_map.get(lang_code, lang_code.upper())
                    prompt_parts.append(f"User ({lang_tag}): {example[query_key]}")
                    prompt_parts.append(f"Assistant ({lang_tag}): {example[response_key]}")
        
        prompt_parts.append(f"User ({user_lang_tag}): {user_query}")
        prompt_parts.append(f"Assistant ({user_lang_tag}):")

        return "\n".join(prompt_parts)

    def predict(self, user_query):
        if not user_query.strip():
            return "Please enter a query."

        try:
            user_lang_short = detect(user_query)
            if user_lang_short not in self.lang_map:
                user_lang_short = "en"
        except Exception:
            user_lang_short = "en"

        user_lang_mbart = self.lang_map.get(user_lang_short, "en_XX")

        icl_examples = self._get_icl_examples()
        prompt = self._construct_prompt(user_query, user_lang_short, icl_examples)

        self.tokenizer.src_lang = user_lang_mbart
        
        encoded_input = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        forced_bos_token_id = self.tokenizer.lang_code_to_id[user_lang_mbart]

        output_tokens = self.model.generate(
            **encoded_input,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=150,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        full_response_decoded = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        user_lang_tag = self.lang_tag_map.get(user_lang_short, user_lang_short.upper())
        expected_assistant_prefix = f"Assistant ({user_lang_tag}):"
        
        last_prefix_idx = full_response_decoded.rfind(expected_assistant_prefix)

        if last_prefix_idx != -1:
            extracted_response = full_response_decoded[last_prefix_idx + len(expected_assistant_prefix):].strip()
            
            if extracted_response.lower().startswith(user_query.lower()):
                extracted_response = extracted_response[len(user_query):].strip()
                if extracted_response.lower().startswith(f"assistant ({user_lang_tag.lower()}):"):
                     extracted_response = extracted_response[len(f"assistant ({user_lang_tag.lower()}):"):]

            if not extracted_response or extracted_response.isspace():
                return "I'm sorry, I couldn't generate a clear response for your query. Please try rephrasing."
            
            return extracted_response.strip()
        else:
            return "I'm sorry, I couldn't generate a clear response for your query in the expected format. Please try rephrasing."

chatbot = MultilingualChatbot()

iface = gr.Interface(
    fn=chatbot.predict,
    inputs=gr.Textbox(lines=3, placeholder="Type your query here in English, Spanish, or French..."),
    outputs=gr.Textbox(label="Assistant Response", lines=5),
    title="Multilingual Customer Support Chatbot (InCLT Crosslingual Transfer Prompting)",
    description="This chatbot uses In-Context Learning examples in multiple languages to boost its cross-lingual understanding and response generation. Try asking questions in English, Spanish, or French!"
)

iface.launch()