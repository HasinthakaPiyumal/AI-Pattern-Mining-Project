import streamlit as st
import speech_recognition as sr
from PIL import Image
import io

# --- Global Configurations / Models (Mocked or lightweight) ---

# Mock backend for e-commerce data
MOCK_ECOMMERCE_DB = {
    "order_12345": {"status": "Shipped", "item": "Laptop", "tracking": "TRK123"},
    "order_67890": {"status": "Processing", "item": "Mouse", "tracking": None},
    "product_laptop": {"price": "$1200", "description": "High performance laptop with 16GB RAM."},
    "product_mouse": {"price": "$25", "description": "Ergonomic wireless mouse."},
}

# Language mapping for display
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh",
}

# --- Multimodal Perception Tools ---

def get_speech_to_text(audio_file_bytes):
    """Converts audio bytes to text using SpeechRecognition (mocked for simplicity)."""
    r = sr.Recognizer()
    try:
        # For a real application, you would save audio_file_bytes to a temporary file
        # and then use sr.AudioFile(temp_file_path) or use a cloud API.
        # Example of using a real recognizer (requires internet for Google Speech Recognition):
        # with sr.AudioFile(io.BytesIO(audio_file_bytes)) as source:
        #     audio = r.record(source)
        #     text = r.recognize_google(audio)
        #     return text
        
        st.info("Simulating speech-to-text processing...")
        # Placeholder for actual speech recognition. Returns a generic phrase.
        return "I have a problem with my order or want to return something."
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Speech service error: {e}"

def get_image_description(image_file_bytes):
    """Generates a description for an image using a placeholder (mocked for simplicity)."""
    try:
        image = Image.open(io.BytesIO(image_file_bytes))
        st.info(f"Simulating image analysis. Image dimensions: {image.size[0]}x{image.size[1]}")
        # In a real application, you would integrate a computer vision model (e.g., CLIP, BLIP) here
        # to generate a more detailed description of the image content.
        return "User uploaded an image. It might be a damaged product or an order detail."
    except Exception as e:
        return f"Error processing image: {e}"

def translate_text_mock(text, target_lang, source_lang="en"):
    """Mocks translation functionality. In a real app, use `transformers` or cloud APIs."""
    if source_lang == target_lang:
        return text

    st.info(f"Simulating translation from {source_lang} to {target_lang}.")
    
    # In a real application, you would use a translation model:
    # from transformers import pipeline
    # translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}")
    # return translator(text)[0]['translation_text']

    # Simple dictionary-based mock for common phrases
    mock_phrases = {
        "en_es": {
            "I have a problem with my order or want to return something.": "Tengo un problema con mi pedido o quiero devolver algo.",
            "What is your order number?": "¿Cuál es tu número de pedido?",
            "Your order 12345 status is Shipped.": "El estado de tu pedido 12345 es Enviado.",
            "I couldn't find order 12345. Please check the number and try again.": "No pude encontrar el pedido 12345. Por favor, verifica el número e inténtalo de nuevo.",
            "That doesn't look like an order number. Please provide your 5-digit order number.": "Eso no parece un número de pedido. Por favor, proporciona tu número de pedido de 5 dígitos.",
            "To process a return, please provide your order number and the reason for return.": "Para procesar una devolución, por favor, proporciona tu número de pedido y el motivo de la devolución.",
            "Thank you. We have initiated the return process for order 67890. You will receive an email shortly with instructions.": "Gracias. Hemos iniciado el proceso de devolución para el pedido 67890. Recibirás un correo electrónico en breve con las instrucciones.",
            "Please provide the order number for the item you wish to return.": "Por favor, proporciona el número de pedido del artículo que deseas devolver.",
            "Could you please specify which product you are interested in? Or provide a product ID/name.": "¿Podrías especificar qué producto te interesa? O proporciona un ID/nombre de producto.",
            "I understand you're having an issue. Please describe your problem in more detail.": "Entiendo que tienes un problema. Por favor, describe tu problema con más detalle.",
            "Hello! How can I assist you today? You can ask about order status, returns, or products.": "¡Hola! ¿Cómo puedo ayudarte hoy? Puedes preguntar sobre el estado del pedido, devoluciones o productos."
        },
        "en_fr": {
            "I have a problem with my order or want to return something.": "J'ai un problème avec ma commande ou je souhaite retourner quelque chose.",
            "What is your order number?": "Quel est votre numéro de commande ?",
            "Your order 12345 status is Shipped.": "Le statut de votre commande 12345 est Expédié.",
            "I couldn't find order 12345. Please check the number and try again.": "Je n'ai pas trouvé la commande 12345. Veuillez vérifier le numéro et réessayer.",
            "That doesn't look like an order number. Please provide your 5-digit order number.": "Cela ne ressemble pas à un numéro de commande. Veuillez fournir votre numéro de commande à 5 chiffres.",
            "To process a return, please provide your order number and the reason for return.": "Pour traiter un retour, veuillez fournir votre numéro de commande et le motif du retour.",
            "Thank you. We have initiated the return process for order 67890. You will receive an email shortly with instructions.": "Merci. Nous avons initié le processus de retour pour la commande 67890. Vous recevrez un e-mail sous peu avec les instructions.",
            "Please provide the order number for the item you wish to return.": "Veuillez fournir le numéro de commande de l'article que vous souhaitez retourner.",
            "Could you please specify which product you are interested in? Or provide a product ID/name.": "Pourriez-vous spécifier le produit qui vous intéresse ? Ou fournir un ID/nom de produit.",
            "I understand you're having an issue. Please describe your problem in more detail.": "Je comprends que vous rencontrez un problème. Veuillez décrire votre problème plus en détail.",
            "Hello! How can I assist you today? You can ask about order status, returns, or products.": "Bonjour ! Comment puis-je vous aider aujourd'hui ? Vous pouvez poser des questions sur le statut de la commande, les retours ou les produits."
        }
        # Add more translations for other languages as needed
    }

    # Translate from target_lang to English first if necessary for processing
    # For this mock, we assume the internal processing is in English.
    # For responses, we translate from English to the target_lang.

    if target_lang != "en":
        for phrase_en, phrase_translated in mock_phrases.get(f"en_{target_lang}", {}).items():
            if text == phrase_en:
                return phrase_translated
    
    # Fallback for untranslated phrases
    return f"[{target_lang.upper()} Translation of: '{text}']"

# --- Foundation Model for Intent Understanding and Dialogue Management (Simulated) ---

class LLMSimulator:
    def __init__(self):
        self.conversation_history = []
        self.awaiting_order_number = False
        self.awaiting_return_details = False

    def reset_state(self):
        self.awaiting_order_number = False
        self.awaiting_return_details = False

    def get_intent(self, text_input):
        """Simulates LLM intent recognition based on keywords."""
        text_input_lower = text_input.lower()
        if "order" in text_input_lower and "status" in text_input_lower:
            return "order_status"
        elif "return" in text_input_lower or "send back" in text_input_lower or "damaged product" in text_input_lower:
            return "return_request"
        elif "product" in text_input_lower or "item" in text_input_lower:
            return "product_inquiry"
        elif "problem" in text_input_lower or "issue" in text_input_lower or "help" in text_input_lower or "image context" in text_input_lower:
            return "general_inquiry"
        elif any(char.isdigit() for char in text_input_lower) and ("order" in text_input_lower or self.awaiting_order_number or self.awaiting_return_details):
            return "order_number_provided"
        else:
            return "unclear"

    def extract_order_number(self, text_input):
        """Simple regex to extract a numerical order number."""
        import re
        match = re.search(r"order\s*(\d+)", text_input, re.IGNORECASE)
        if match:
            return match.group(1)
        # Also look for standalone numbers that could be order IDs if expecting one
        match = re.search(r"\b(\d{5,})\b", text_input) # Simple 5+ digit number
        if match:
            return match.group(1)
        return None

    def generate_response(self, text_input_en, user_lang):
        """Simulates LLM response generation and dialogue management.
        text_input_en: The user's input, already translated to English for processing.
        user_lang: The original language code of the user for translating the response.
        """
        processed_input = text_input_en # Assume this is in English for internal logic

        # Dialogue state management
        if self.awaiting_order_number:
            order_num = self.extract_order_number(processed_input)
            if order_num:
                self.reset_state()
                order_info = MOCK_ECOMMERCE_DB.get(f"order_{order_num}")
                if order_info:
                    response_en = f"Your order {order_num} status is {order_info['status']}."
                    if order_info['tracking']:
                        response_en += f" Tracking number: {order_info['tracking']}."
                    return translate_text_mock(response_en, user_lang, "en")
                else:
                    return translate_text_mock(f"I couldn't find order {order_num}. Please check the number and try again.", user_lang, "en")
            else:
                return translate_text_mock("That doesn't look like an order number. Please provide your 5-digit order number.", user_lang, "en")

        if self.awaiting_return_details:
            order_num = self.extract_order_number(processed_input)
            if order_num:
                self.reset_state()
                # In a real system, you'd log this and start a return process in the backend
                return translate_text_mock(f"Thank you. We have initiated the return process for order {order_num}. You will receive an email shortly with instructions.", user_lang, "en")
            else:
                return translate_text_mock("Please provide the order number for the item you wish to return.", user_lang, "en")

        intent = self.get_intent(processed_input)
        response_en = ""

        if intent == "order_status":
            self.awaiting_order_number = True
            response_en = "What is your order number?"
        elif intent == "return_request":
            self.awaiting_return_details = True
            response_en = "To process a return, please provide your order number and the reason for return."
        elif intent == "product_inquiry":
            response_en = "Could you please specify which product you are interested in? Or provide a product ID/name."
        elif intent == "general_inquiry":
            response_en = "I understand you're having an issue. Please describe your problem in more detail."
        elif intent == "order_number_provided":
            order_num = self.extract_order_number(processed_input)
            if order_num:
                order_info = MOCK_ECOMMERCE_DB.get(f"order_{order_num}")
                if order_info:
                    response_en = f"Your order {order_num} status is {order_info['status']}."
                    if order_info['tracking']:
                        response_en += f" Tracking number: {order_info['tracking']}."
                else:
                    response_en = f"I couldn't find order {order_num}. Please check the number and try again."
            else:
                response_en = "Please provide a valid order number."
            self.reset_state() # Reset after processing direct order number
        else: # Unclear intent or general greeting
            self.reset_state()
            response_en = "Hello! How can I assist you today? You can ask about order status, returns, or products."
            
        return translate_text_mock(response_en, user_lang, "en")

# --- Streamlit Application ---

def main():
    st.set_page_config(page_title="Multimodal E-commerce Support", layout="wide")
    st.title("🛍️ Multimodal & Multilingual E-commerce Support Assistant")
    st.markdown("This AI assistant can understand your queries via text, speech, or images, and in multiple languages.")

    # Sidebar for language selection and clearing chat
    with st.sidebar:
        st.header("Settings")
        user_lang_name = st.selectbox(
            "Select your preferred language:",
            list(LANGUAGES.keys()),
            index=0 # Default to English
        )
        user_lang_code = LANGUAGES[user_lang_name]

        if st.button("Clear Chat", help="Clears the conversation history and resets the AI state."):
            st.session_state.messages = []
            st.session_state.llm_simulator.reset_state()
            st.rerun()

    # Initialize chat history and LLM simulator
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm_simulator" not in st.session_state:
        st.session_state.llm_simulator = LLMSimulator()

    llm_simulator = st.session_state.llm_simulator

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Main input area
    input_text = st.chat_input("Type your message here...")
    audio_file = st.file_uploader("Or upload an audio file (.wav, .mp3)", type=["wav", "mp3"])
    image_file = st.file_uploader("Or upload an image file (.png, .jpg)", type=["png", "jpg", "jpeg"])

    user_input_processed_en = "" # All inputs converted to English for LLM processing
    user_submitted_input = False

    if input_text:
        st.session_state.messages.append({"role": "user", "content": input_text})
        # Translate user input to English for LLM processing
        user_input_processed_en = translate_text_mock(input_text, "en", user_lang_code)
        st.chat_message("user").markdown(input_text) # Display original user input
        user_submitted_input = True

    if audio_file:
        audio_bytes = audio_file.read()
        st.session_state.messages.append({"role": "user", "content": f"🎤 Audio input: {audio_file.name}"})
        st.chat_message("user").markdown(f"🎤 Audio input: {audio_file.name}")
        
        st.audio(audio_bytes, format=f"audio/{audio_file.type.split('/')[-1]}")
        
        speech_text = get_speech_to_text(audio_bytes)
        st.info(f"Transcribed Text (English Mock): {speech_text}")
        if speech_text:
            # Translate transcribed text to English for LLM processing
            user_input_processed_en = translate_text_mock(speech_text, "en", user_lang_code)
            st.session_state.messages.append({"role": "user", "content": f"*(Transcription)*: {speech_text}"})
            user_submitted_input = True

    if image_file:
        image_bytes = image_file.read()
        st.session_state.messages.append({"role": "user", "content": f"🖼️ Image input: {image_file.name}"})
        st.chat_message("user").image(image_bytes, caption=image_file.name, width=200)
        
        image_description = get_image_description(image_bytes)
        st.info(f"Image Description (English Mock): {image_description}")
        if image_description:
            # Append image description to the processed input for LLM context
            # If there was existing text input, append it; otherwise, this is the main input
            if user_input_processed_en:
                user_input_processed_en += f" [IMAGE CONTEXT]: {image_description}"
            else:
                user_input_processed_en = f"[IMAGE CONTEXT]: {image_description}"
            st.session_state.messages.append({"role": "user", "content": f"*(Image description)*: {image_description}"})
            user_submitted_input = True

    # Process input with LLM if any valid input was submitted
    if user_submitted_input and user_input_processed_en:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ai_response_translated = llm_simulator.generate_response(user_input_processed_en, user_lang_code)
                st.markdown(ai_response_translated)
                st.session_state.messages.append({"role": "assistant", "content": ai_response_translated})
                
# Run the app
if __name__ == "__main__":
    main()