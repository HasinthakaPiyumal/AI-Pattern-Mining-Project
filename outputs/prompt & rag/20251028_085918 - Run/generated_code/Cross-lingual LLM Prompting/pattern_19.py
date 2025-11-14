
import random
from typing import List, Dict

# Placeholder for transformers library if a real LLM were integrated
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from langdetect import detect, DetectorFactory
# DetectorFactory.seed = 0 # for reproducible language detection

# Gradio for the UI
import gradio as gr

class ChatbotConfig:
    SYSTEM_PROMPT = "You are a helpful customer support agent. Provide concise and accurate responses."
    # Dummy LLM response for demonstration
    DUMMY_LLM_RESPONSES = {
        "en": "Thank you for your query. We are looking into it.",
        "es": "Gracias por su consulta. Lo estamos revisando.",
        "fr": "Merci pour votre demande. Nous l'examinons.",
        "de": "Vielen Dank für Ihre Anfrage. Wir prüfen dies.",
        "it": "Grazie per la tua richiesta. Lo stiamo esaminando."
    }

# 1. Multilingual LLM Core (Simulated for this environment)
class MultilingualLLMSimulator:
    def __init__(self):
        # In a real application, you would load your model here
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        pass

    def generate_response(self, prompt: str, target_lang: str) -> str:
        """Simulates LLM response generation."""
        print(f"--- Sending prompt to LLM ---\n{prompt}\n---")
        # In a real scenario, you would tokenize, generate, and decode
        # inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # outputs = self.model.generate(
        #     **inputs,
        #     forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang],
        #     max_new_tokens=100
        # )
        # response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # For this demonstration, return a dummy response based on target language
        return ChatbotConfig.DUMMY_LLM_RESPONSES.get(target_lang.lower(), ChatbotConfig.DUMMY_LLM_RESPONSES["en"])

# 2. Prompt Engineering Module (InCLT Implementation)
class PromptEngineer:
    def __init__(self):
        self.example_database = [
            # Source-only examples
            {"source_lang": "en", "target_lang": "en", "query": "How do I reset my password?", "response": "Please visit our website's 'Forgot Password' link and follow the instructions.", "domain": "account"},
            {"source_lang": "es", "target_lang": "es", "query": "¿Cómo puedo cambiar mi dirección de correo electrónico?", "response": "Puede actualizar su dirección de correo electrónico en la sección 'Configuración de la cuenta' de su perfil.", "domain": "account"},
            {"source_lang": "fr", "target_lang": "fr", "query": "Quel est le délai de livraison ?", "response": "Le délai de livraison standard est de 3 à 5 jours ouvrables.", "domain": "shipping"},

            # Target-only examples (could be different queries from source-only for diversity)
            {"source_lang": "en", "target_lang": "en", "query": "Where is my order?", "response": "You can track your order using the tracking number provided in your shipping confirmation email.", "domain": "shipping"},
            {"source_lang": "es", "target_lang": "es", "query": "¿Cuál es su política de devolución?", "response": "Nuestra política de devolución permite devoluciones dentro de los 30 días posteriores a la compra.", "domain": "returns"},
            {"source_lang": "fr", "target_lang": "fr", "query": "Comment contacter le support client ?", "response": "Vous pouvez nous contacter via le chat en direct sur notre site web ou par e-mail.", "domain": "contact"},

            # Cross-lingual examples
            {"source_lang": "es", "target_lang": "en", "query": "No puedo iniciar sesión en mi cuenta.", "response": "If you are unable to log in, please try resetting your password using the 'Forgot Password' link.", "domain": "account"},
            {"source_lang": "en", "target_lang": "fr", "query": "My payment failed. What should I do?", "response": "Veuillez vérifier vos informations de paiement ou essayer une autre méthode de paiement.", "domain": "billing"},
            {"source_lang": "fr", "target_lang": "es", "query": "Je voudrais annuler ma commande.", "response": "Para cancelar su pedido, por favor vaya a la sección 'Mis Pedidos' y seleccione la opción de cancelar.", "domain": "orders"},
            {"source_lang": "it", "target_lang": "en", "query": "Non riesco ad accedere al mio account. Cosa devo fare?", "response": "If you are unable to access your account, please use the 'Forgot Password' option or contact support.", "domain": "account"},
            {"source_lang": "en", "target_lang": "de", "query": "How do I upgrade my subscription?", "response": "Um Ihr Abonnement zu aktualisieren, gehen Sie zu Ihren 'Kontoeinstellungen' und wählen Sie 'Abonnement verwalten'.", "domain": "billing"},
            {"source_lang": "de", "target_lang": "en", "query": "Ich habe mein Produkt noch nicht erhalten.", "response": "If you haven't received your product, please check the tracking information or contact us with your order number.", "domain": "shipping"},
        ]

    def _select_examples(self, user_source_lang: str, user_target_lang: str, num_examples: int = 5) -> List[Dict]:
        """Selects a diverse set of examples based on source and target languages."""
        selected_examples = []

        # Prioritize cross-lingual examples that match user's source and target
        cross_lingual_matches = [e for e in self.example_database
                                 if e["source_lang"] == user_source_lang and e["target_lang"] == user_target_lang
                                 and e["source_lang"] != e["target_lang"]]

        # Add these first
        random.shuffle(cross_lingual_matches)
        selected_examples.extend(cross_lingual_matches[:min(len(cross_lingual_matches), num_examples // 2)])

        remaining_slots = num_examples - len(selected_examples)
        if remaining_slots <= 0:
            return selected_examples

        # Then add examples where target language matches (response language)
        target_lang_matches = [e for e in self.example_database
                               if e["target_lang"] == user_target_lang
                               and e not in selected_examples]
        random.shuffle(target_lang_matches)
        selected_examples.extend(target_lang_matches[:min(len(target_lang_matches), remaining_slots // 2)])

        remaining_slots = num_examples - len(selected_examples)
        if remaining_slots <= 0:
            return selected_examples

        # Then add examples where source language matches (query language)
        source_lang_matches = [e for e in self.example_database
                               if e["source_lang"] == user_source_lang
                               and e not in selected_examples]
        random.shuffle(source_lang_matches)
        selected_examples.extend(source_lang_matches[:min(len(source_lang_matches), remaining_slots // 2)])

        remaining_slots = num_examples - len(selected_examples)
        if remaining_slots <= 0:
            return selected_examples

        # Finally, fill with any other diverse examples if needed
        other_examples = [e for e in self.example_database if e not in selected_examples]
        random.shuffle(other_examples)
        selected_examples.extend(other_examples[:remaining_slots])

        return selected_examples[:num_examples]


    def assemble_prompt(self, user_query: str, user_source_lang: str, user_target_lang: str) -> str:
        """Constructs the full prompt with in-context examples."""
        prompt_parts = [ChatbotConfig.SYSTEM_PROMPT]
        
        selected_examples = self._select_examples(user_source_lang, user_target_lang)
        
        for example in selected_examples:
            prompt_parts.append(f"\n\n{example['source_lang'].capitalize()} Query: \"{example['query']}\"")
            prompt_parts.append(f"{example['target_lang'].capitalize()} Response: \"{example['response']}\"")

        prompt_parts.append(f"\n\nUser Query [Source Language: {user_source_lang.capitalize()}, Target Language: {user_target_lang.capitalize()}]: \"{user_query}\"")
        prompt_parts.append(f"{user_target_lang.capitalize()} Response:")

        return "".join(prompt_parts)

# Main Chatbot Logic
class MultilingualChatbot:
    def __init__(self):
        self.llm_simulator = MultilingualLLMSimulator()
        self.prompt_engineer = PromptEngineer()

    def get_chatbot_response(self, user_query: str, source_lang_input: str, target_lang_input: str) -> str:
        # 4. Language Detection (simplified/optional)
        # For a real app, integrate langdetect. For this example, we trust user input or default.
        detected_source_lang = source_lang_input.lower() if source_lang_input else "en" # Default to English if not provided
        actual_target_lang = target_lang_input.lower() if target_lang_input else "en" # Default to English if not provided
        
        # 3. Prompt Engineering
        full_prompt = self.prompt_engineer.assemble_prompt(
            user_query,
            detected_source_lang,
            actual_target_lang
        )

        # 1. Multilingual LLM Core
        llm_response = self.llm_simulator.generate_response(full_prompt, actual_target_lang)
        return llm_response

# 3. Input/Output and Interface (Gradio)
chatbot = MultilingualChatbot()

def chatbot_interface(query: str, source_lang: str, target_lang: str) -> str:
    if not query:
        return "Please enter a query."
    return chatbot.get_chatbot_response(query, source_lang, target_lang)

# Define available languages for dropdowns
LANGUAGES = ["en", "es", "fr", "de", "it"]

if __name__ == "__main__":
    demo = gr.Interface(
        fn=chatbot_interface,
        inputs=[
            gr.Textbox(lines=2, placeholder="Enter your query here...", label="Your Query"),
            gr.Dropdown(choices=LANGUAGES, value="en", label="Source Language"),
            gr.Dropdown(choices=LANGUAGES, value="en", label="Target Language (for response)")
        ],
        outputs="text",
        title="Multilingual Customer Support Chatbot (InCLT Pattern)",
        description="This chatbot demonstrates the InCLT Crosslingual Transfer Prompting pattern. Provide a query, source language, and desired target language for the response. In-context examples are dynamically generated to enhance cross-lingual understanding. (LLM response is simulated for this environment)"
    )

    demo.launch()
