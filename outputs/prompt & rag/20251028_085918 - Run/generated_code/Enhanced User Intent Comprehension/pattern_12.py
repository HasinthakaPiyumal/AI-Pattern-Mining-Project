import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel
from loguru import logger
from PIL import Image
import io
import base64

# --- Configuration (using python-dotenv would be ideal for real projects) ---
# For this example, we'll use direct environment variable placeholders or defaults
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your_openai_key")

# --- Mock Services/Libraries (for demonstration purposes) ---

class MockSpeechToText:
    """Simulates a Speech-to-Text service."""
    def transcribe(self, audio_file: UploadFile) -> str:
        logger.info(f"Mock Speech-to-Text processing audio file: {audio_file.filename}")
        # In a real scenario, this would use a library like `SpeechRecognition`
        # or call a cloud ASR API (e.g., Whisper, Google Cloud Speech-to-Text).
        # For demo, we'll just return a placeholder.
        return f"User said: 'Please help me with my order number {audio_file.filename.split('.')[0].replace('_', '')} and I have an issue with the delivery.'"

class MockImageAnalysis:
    """Simulates an Image Analysis service."""
    def analyze(self, image_file: UploadFile) -> str:
        logger.info(f"Mock Image Analysis processing image file: {image_file.filename}")
        try:
            image = Image.open(io.BytesIO(image_file.file.read()))
            # In a real scenario, this would use `transformers` (e.g., CLIP, BLIP)
            # or a cloud Vision API to extract objects, text, etc.
            # For demo, we'll describe basic image properties and look for keywords.
            width, height = image.size
            mode = image.mode
            analysis_result = f"Image detected: {image_file.filename}, size {width}x{height}, mode {mode}."
            if "invoice" in image_file.filename.lower() or "bill" in image_file.filename.lower():
                analysis_result += " Content might be an invoice or billing document."
            return analysis_result
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return "Could not analyze image content."

class MockMachineTranslation:
    """Simulates a Machine Translation service."""
    def translate(self, text: str, target_lang: str = "en", source_lang: str = "auto") -> str:
        logger.info(f"Mock Machine Translation from {source_lang} to {target_lang} for text: {text[:30]}...")
        # In a real scenario, this would use `transformers` (e.g., Helsinki-NLP)
        # or a cloud Translation API (e.g., Google Cloud Translation).
        # For demo, we'll perform a very basic keyword-based translation simulation.
        if target_lang == "en":
            if "bonjour" in text.lower():
                return text.lower().replace("bonjour", "hello") + " (translated from French)"
            if "hola" in text.lower():
                return text.lower().replace("hola", "hello") + " (translated from Spanish)"
        elif target_lang == "fr":
            if "hello" in text.lower():
                return text.lower().replace("hello", "bonjour") + " (traduit en français)"
        return text # Return original if no specific translation rule matches

class MockLLM:
    """Simulates a Large Language Model for intent, entity, and response generation."""
    def process_query(self, text: str, session_context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Mock LLM processing query: {text[:50]}...")
        intent = "unknown"
        entities = {}
        response_text = "I'm sorry, I couldn't understand your request fully. Could you please rephrase?"
        clarification_needed = False

        text_lower = text.lower()

        if "order" in text_lower or "delivery" in text_lower or "shipment" in text_lower:
            intent = "order_inquiry"
            if "status" in text_lower:
                intent = "order_status_check"
            for keyword in ["#", "ord", "id"]: # Simple entity extraction for order numbers
                if keyword in text_lower:
                    parts = text_lower.split(keyword)
                    if len(parts) > 1:
                        entity_candidate = parts[1].split()[0].strip().replace(':', '').replace('.', '')
                        if entity_candidate.isalnum() and len(entity_candidate) > 3:
                            entities["order_id"] = entity_candidate
                            break

            if "order_id" not in entities and not session_context.get("clarifying_order_id", False):
                response_text = "I can help with order inquiries. Could you please provide your order number?"
                clarification_needed = True
                session_context["clarifying_order_id"] = True
            else:
                order_id = entities.get("order_id") or session_context.get("order_id")
                if order_id:
                    response_text = f"Okay, I'll check the status for order {order_id}."
                    session_context["order_id"] = order_id # Store in session
                    session_context["clarifying_order_id"] = False
                else:
                    response_text = "Please provide a valid order number."
                    clarification_needed = True

        elif "password" in text_lower or "account access" in text_lower or "login" in text_lower:
            intent = "account_management"
            if "reset" in text_lower:
                intent = "password_reset"
                response_text = "I can help you reset your password. Please visit our 'Forgot Password' page and follow the instructions."
            else:
                response_text = "For account access issues, please describe your problem in more detail. Are you looking to reset your password or something else?"
                clarification_needed = True

        elif "product" in text_lower or "item" in text_lower or "details" in text_lower:
            intent = "product_inquiry"
            response_text = "What product are you interested in? I can provide details or availability."

        elif "hello" in text_lower or "hi" in text_lower:
            intent = "greeting"
            response_text = f"Hello! How can I assist you today?"

        # Personalized learning aspect: if previous context exists, use it
        if session_context.get("last_intent") == "order_inquiry" and "yes" in text_lower:
            response_text = f"Great! What about order {session_context.get('order_id')} would you like to know?"
            intent = "follow_up_order_inquiry"
            clarification_needed = False

        session_context["last_intent"] = intent # Update for next turn

        return {
            "intent": intent,
            "entities": entities,
            "response_text": response_text,
            "clarification_needed": clarification_needed,
            "session_context": session_context # Return updated context
        }

class SessionManager:
    """Manages conversational context/session state (in-memory for demo)."""
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_session(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id, {})

    def update_session(self, session_id: str, data: Dict[str, Any]):
        self.sessions[session_id] = {**self.sessions.get(session_id, {}), **data}
        logger.info(f"Updated session {session_id}: {self.sessions[session_id]}")

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session {session_id}")

class MockExternalSystemIntegration:
    """Simulates integrations with external systems like CRM, Order Management."""
    def get_order_status(self, order_id: str) -> str:
        logger.info(f"Mock ESI: Fetching status for order {order_id}")
        if order_id == "12345":
            return "Order 12345 is currently 'In Transit' and expected by tomorrow."
        elif order_id == "67890":
            return "Order 67890 was successfully delivered on Monday."
        return f"Could not find details for order {order_id}."

    def get_product_details(self, product_name: str) -> str:
        logger.info(f"Mock ESI: Fetching details for product {product_name}")
        if "laptop" in product_name.lower():
            return "The 'ProBook X' is our latest model with 16GB RAM and a 1TB SSD. Priced at $1200."
        return f"No specific details found for product '{product_name}'."

    def initiate_password_reset(self, user_id: str) -> str:
        logger.info(f"Mock ESI: Initiating password reset for user {user_id}")
        return "A password reset link has been sent to your registered email address."


# --- FastAPI Application --- 

app = FastAPI(
    title="Advanced Multi-modal Customer Support Bot",
    description="Bot leveraging Enhanced User Intent Comprehension for diverse inputs."
)

# Initialize services
speech_to_text_module = MockSpeechToText()
image_analysis_module = MockImageAnalysis()
machine_translation_module = MockMachineTranslation()
llm_core = MockLLM()
session_manager = SessionManager()
es_integration = MockExternalSystemIntegration()

class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    entities: Dict[str, Any]
    clarification_needed: bool
    debug_info: Optional[Dict[str, Any]] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(
    session_id: str = Form(..., description="Unique identifier for the conversation session."),
    text_input: Optional[str] = Form(None, description="Text input from the user."),
    audio_file: Optional[UploadFile] = File(None, description="Audio input from the user (e.g., .wav, .mp3)."),
    image_file: Optional[UploadFile] = File(None, description="Image input from the user (e.g., .png, .jpg)."),
    language: str = Form("en", description="Preferred language of the user (e.g., 'en', 'fr', 'es').")
):
    logger.info(f"Received chat request for session {session_id}. Lang: {language}")
    user_query_text = ""
    debug_info = {
        "original_text_input": text_input,
        "processed_audio": None,
        "processed_image": None,
        "translated_input": None,
        "session_before_llm": None
    }

    # 1. Perceptual Augmentation Layer
    if audio_file:
        try:
            user_query_text += speech_to_text_module.transcribe(audio_file) + " "
            debug_info["processed_audio"] = "Transcribed from audio"
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Audio processing failed.")

    if image_file:
        try:
            user_query_text += image_analysis_module.analyze(image_file) + " "
            debug_info["processed_image"] = "Analyzed from image"
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Image processing failed.")

    if text_input:
        user_query_text += text_input + " "

    user_query_text = user_query_text.strip()

    if not user_query_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid input provided (text, audio, or image)." )

    # Machine Translation (if not English)
    if language != "en":
        original_query = user_query_text
        user_query_text = machine_translation_module.translate(user_query_text, target_lang="en", source_lang=language)
        debug_info["translated_input"] = {"original": original_query, "translated": user_query_text}

    # 2. Language Understanding Layer (Core LLM)
    session_context = session_manager.get_session(session_id)
    debug_info["session_before_llm"] = session_context.copy() # Capture state before LLM

    llm_output = llm_core.process_query(user_query_text, session_context)

    # Update session with LLM-modified context
    session_manager.update_session(session_id, llm_output["session_context"])

    # 3. Dialogue Management Layer & External System Integration
    bot_response = llm_output["response_text"]
    current_intent = llm_output["intent"]
    extracted_entities = llm_output["entities"]

    if not llm_output["clarification_needed"]:
        if current_intent == "order_status_check" and "order_id" in extracted_entities:
            order_id = extracted_entities["order_id"]
            status_info = es_integration.get_order_status(order_id)
            bot_response = f"For order {order_id}: {status_info}"
        elif current_intent == "password_reset":
            # In a real app, user_id might be inferred from authentication
            user_id = "current_user" # Placeholder
            reset_info = es_integration.initiate_password_reset(user_id)
            bot_response = reset_info
        elif current_intent == "product_inquiry" and "product_name" in extracted_entities:
            product_name = extracted_entities["product_name"]
            product_details = es_integration.get_product_details(product_name)
            bot_response = product_details

    # 4. Response Generation Layer (Multi-modal Output Formatter - currently text-only)
    final_response = bot_response

    # Translate response back to user's preferred language if different from English
    if language != "en":
        final_response = machine_translation_module.translate(final_response, target_lang=language, source_lang="en")

    logger.info(f"Bot response for session {session_id}: {final_response[:50]}...")

    return ChatResponse(
        session_id=session_id,
        response=final_response,
        intent=current_intent,
        entities=extracted_entities,
        clarification_needed=llm_output["clarification_needed"],
        debug_info=debug_info
    )

# --- Streamlit Frontend (for demonstration, run separately) ---
# To run this Streamlit app, save this file as `customer_support_bot.py`
# and then run `streamlit run customer_support_bot.py` from your terminal.
# You also need to start the FastAPI server first (e.g., `uvicorn customer_support_bot:app --reload`)

if __name__ == "__main__":
    import streamlit as st
    import requests

    st.set_page_config(page_title="Multi-modal Customer Support Bot Demo", layout="wide")
    st.title("🤖 Advanced Multi-modal Customer Support Bot")
    st.markdown("This demo showcases a bot that understands diverse inputs (text, audio, image) and responds intelligently.")

    # FastAPI endpoint URL
    FASTAPI_URL = "http://localhost:8000/chat"

    # Session ID management
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(os.urandom(16).hex()) # Generate a unique session ID
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.sidebar.header("Bot Settings")
    current_language = st.sidebar.selectbox("Select Language", ["en", "fr", "es"], index=0)
    st.sidebar.text(f"Current Session ID: {st.session_state.session_id}")
    if st.sidebar.button("Clear Session"): # Option to clear session
        st.session_state.session_id = str(os.urandom(16).hex())
        st.session_state.messages = []
        # Optionally, send a clear session request to FastAPI if implemented
        st.experimental_rerun()

    # Display chat history
    st.write("### Conversation History")
    chat_container = st.container(height=400, border=True)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Input forms
    st.write("### Your Input")
    with st.form("chat_form", clear_on_submit=True):
        text_input = st.text_input("Type your message here:", key="text_input_key", label_visibility="collapsed")
        audio_file = st.file_uploader("Upload Audio (e.g., .wav):", type=["wav", "mp3"], key="audio_input_key")
        image_file = st.file_uploader("Upload Image (e.g., .png, .jpg):", type=["png", "jpg", "jpeg"], key="image_input_key")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            submit_button = st.form_submit_button("Send to Bot")

    if submit_button and (text_input or audio_file or image_file):
        user_message_content = []
        files = {}
        form_data = {"session_id": st.session_state.session_id, "language": current_language}

        if text_input:
            user_message_content.append(f"Text: {text_input}")
            form_data["text_input"] = text_input
        if audio_file:
            user_message_content.append(f"Audio: {audio_file.name}")
            files["audio_file"] = (audio_file.name, audio_file.getvalue(), audio_file.type)
        if image_file:
            user_message_content.append(f"Image: {image_file.name}")
            files["image_file"] = (image_file.name, image_file.getvalue(), image_file.type)

        user_message = " | ".join(user_message_content)
        st.session_state.messages.append({"role": "user", "content": user_message})

        with st.spinner("Bot is thinking..."):
            try:
                # Use requests.post with files and data for form-data encoding
                response = requests.post(FASTAPI_URL, files=files, data=form_data)
                response.raise_for_status()  # Raise an exception for bad status codes
                bot_response_data = response.json()

                st.session_state.messages.append({"role": "assistant", "content": bot_response_data["response"]})

                st.sidebar.json(bot_response_data["debug_info"])
                st.sidebar.write(f"**Intent:** {bot_response_data['intent']}")
                st.sidebar.write(f"**Entities:** {bot_response_data['entities']}")

            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with the bot: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "I'm having trouble connecting to my services right now. Please try again later."})
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "An unexpected error occurred while processing your request."})

        st.experimental_rerun() # Rerun to update chat history and clear input

    elif submit_button:
        st.warning("Please provide some input (text, audio, or image) to chat with the bot.")

    st.markdown("---")
    st.markdown("**How to run this demo:**")
    st.markdown("1. Save this code as `customer_support_bot.py`.")
    st.markdown("2. Install required libraries: `pip install fastapi uvicorn[standard] loguru Pillow python-dotenv streamlit requests`")
    st.markdown("3. Start the FastAPI server in one terminal: `uvicorn customer_support_bot:app --reload --port 8000`")
    st.markdown("4. Start the Streamlit frontend in another terminal: `streamlit run customer_support_bot.py`")
    st.markdown("5. Access the Streamlit app in your browser (usually `http://localhost:8501`).")

