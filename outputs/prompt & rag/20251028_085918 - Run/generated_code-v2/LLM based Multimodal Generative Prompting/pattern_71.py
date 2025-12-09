import streamlit as st
import os
from dotenv import load_dotenv
from transformers import pipeline
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from loguru import logger

load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Mock Data/Knowledge Base ---
PRODUCT_CATALOG = {
    "smartphone": "Latest model smartphone with A17 chip, 128GB storage, and a 6.1-inch OLED display. Price: $799.",
    "laptop": "High-performance laptop with M2 Pro chip, 16GB RAM, and 512GB SSD. Price: $1499.",
    "headphones": "Noise-cancelling over-ear headphones with 30-hour battery life. Price: $199."
}

FAQ_KB = {
    "shipping": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
    "returns": "You can return items within 30 days of purchase with the original receipt."
}

# --- Input Processing Services ---
# Speech-to-Text Service (Whisper via Transformers)
# Using a small model for demonstration purposes. Larger models are more accurate.
@st.cache_resource
def get_asr_pipeline():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
asr_pipeline = get_asr_pipeline()

def speech_to_text(audio_file_path):
    if audio_file_path:
        try:
            text = asr_pipeline(audio_file_path, generate_kwargs={"task": "transcribe"})["text"]
            logger.info(f"ASR output: {text}")
            return text
        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            return ""
    return ""

# Image Analysis Service (Placeholder)
# In a real scenario, this would use CLIP/BLIP to generate a descriptive caption or tags.
def analyze_image(image_file):
    if image_file:
        # For demonstration, we'll just return a generic description
        # In a real app, use: model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # image = Image.open(image_file.buffer).convert("RGB")
        # inputs = processor(image, return_tensors="pt")
        # out = model.generate(**inputs)
        # return processor.decode(out[0], skip_special_tokens=True)
        logger.info(f"Image analysis triggered for: {image_file.name}")
        return f"User provided an image named '{image_file.name}'. A potential product of interest related to this image." # Placeholder
    return ""

# Machine Translation Service (MarianMT via Transformers)
@st.cache_resource
def get_translation_pipeline(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    try:
        return pipeline("translation", model=model_name)
    except Exception as e:
        logger.warning(f"Could not load translation model {model_name}: {e}. Falling back to no translation.")
        return None

def translate_text(text, source_lang="auto", target_lang="en"):
    if text and source_lang != target_lang: # Simple check, can be improved with lang detection
        translation_pipeline = get_translation_pipeline(source_lang, target_lang)
        if translation_pipeline:
            try:
                translated_text = translation_pipeline(text)[0]["translation_text"]
                logger.info(f"Translated text from {source_lang} to {target_lang}: {translated_text}")
                return translated_text
            except Exception as e:
                logger.error(f"Error in machine translation: {e}")
                return text
    return text

# --- Backend Services/APIs (Mock) ---
def get_product_info(query: str) -> str:
    logger.info(f"Retrieving product info for: {query}")
    for product, desc in PRODUCT_CATALOG.items():
        if product in query.lower():
            return desc
    return "No specific product information found for your query. Can you please be more specific?"

def get_order_status(order_id: str) -> str:
    logger.info(f"Retrieving order status for ID: {order_id}")
    if order_id == "12345":
        return "Order 12345 is currently being processed and is expected to ship within 2 business days."
    elif order_id == "67890":
        return "Order 67890 has been delivered on October 26, 2023."
    return f"Order {order_id} not found. Please double-check your order ID."

def query_internal_kb(query: str) -> str:
    logger.info(f"Querying internal KB for: {query}")
    for topic, answer in FAQ_KB.items():
        if topic in query.lower():
            return answer
    return "I'm sorry, I couldn't find an answer in our knowledge base for that query."

# --- Orchestration Layer (LangChain Agent) ---
@st.cache_resource
def get_langchain_agent():
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

    tools = [
        Tool(
            name="ProductInfoRetriever",
            func=get_product_info,
            description="Useful for retrieving detailed information about products from the catalog."
        ),
        Tool(
            name="OrderStatusLookup",
            func=get_order_status,
            description="Useful for looking up the status of a customer's order. Input should be an order ID (e.g., '12345')."
        ),
        Tool(
            name="KnowledgeBaseQuery",
            func=query_internal_kb,
            description="Useful for finding answers to general customer support questions from the internal knowledge base (e.g., shipping, returns)."
        ),
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent

agent_executor = get_langchain_agent()

def process_multimodal_query(text_input, audio_file, image_file):
    combined_input = []

    if text_input:
        # Basic language detection for translation - can be improved
        # For simplicity, assuming anything not explicitly English needs translation for LLM
        # This is a very basic heuristic, in a real app, use a dedicated language detection library
        lang_detected = "en"
        if any(c in text_input for c in "àèéìòùÀÈÉÌÒÙçÇñÑ") or any(word in text_input.lower() for word in ["bonjour", "hola", "guten tag"]):
             lang_detected = "fr" # Example: detect French
        
        processed_text = translate_text(text_input, source_lang=lang_detected, target_lang="en")
        combined_input.append(f"User's text query: {processed_text}")

    if audio_file:
        audio_text = speech_to_text(audio_file)
        if audio_text:
            combined_input.append(f"User's voice input transcribed: {audio_text}")

    if image_file:
        image_desc = analyze_image(image_file)
        if image_desc:
            combined_input.append(f"User provided an image: {image_desc}")

    if not combined_input:
        return "Please provide some input (text, voice, or image) for me to assist you."

    final_query = " ".join(combined_input)
    logger.info(f"Final query to LLM agent: {final_query}")

    try:
        response = agent_executor.run(final_query)
        return response
    except Exception as e:
        logger.error(f"Error processing query with LangChain agent: {e}")
        return "I apologize, but I encountered an error while trying to process your request. Please try again or rephrase your query."

# --- Streamlit Frontend ---
st.title("🛒 Multimodal E-commerce Chatbot")
st.markdown("Hello! I'm your AI assistant. You can ask me questions using text, voice, or by uploading an image.")

# Text Input
text_query = st.text_area("Type your question here:", key="text_input")

# Voice Input
st.subheader("Voice Input")
audio_file = st.file_uploader("Upload an audio file (.wav, .mp3)", type=["wav", "mp3"], key="audio_input")

# Image Input
st.subheader("Image Input")
image_file = st.file_uploader("Upload an image file (.jpg, .png)", type=["jpg", "png", "jpeg"], key="image_input")

if st.button("Get Assistance"):
    if not text_query and not audio_file and not image_file:
        st.warning("Please provide some input to get assistance.")
    else:
        with st.spinner("Processing your multimodal query..."):
            response = process_multimodal_query(text_query, audio_file, image_file)
            st.subheader("Chatbot Response:")
            st.info(response)
