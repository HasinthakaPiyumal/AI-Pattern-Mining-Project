import streamlit as st
import speech_recognition as sr
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langdetect import detect, DetectorFactory
import os

DetectorFactory.seed = 0

if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

@st.cache_resource
def load_speech_recognizer():
    r = sr.Recognizer()
    return r

@st.cache_resource
def load_image_captioning_model():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_translation_models():
    romance_to_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ROMANCE-en")
    romance_to_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ROMANCE-en")

    en_to_romance_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")
    en_to_romance_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")

    return {
        "romance_to_en": {"tokenizer": romance_to_en_tokenizer, "model": romance_to_en_model},
        "en_to_romance": {"tokenizer": en_to_romance_tokenizer, "model": en_to_romance_model},
    }

recognizer = load_speech_recognizer()
image_processor, image_captioner = load_image_captioning_model()
translation_models = load_translation_models()

def speech_to_text_from_microphone(recognizer_instance):
    with sr.Microphone() as source:
        st.info("Say something!")
        audio = recognizer_instance.listen(source)
        try:
            text = recognizer_instance.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
            return None

def speech_to_text_from_audio_file(recognizer_instance, audio_file):
    try:
        audio_data = sr.AudioFile(audio_file)
        with audio_data as source:
            audio = recognizer_instance.record(source)
        text = recognizer_instance.recognize_google(audio)
        st.success(f"Transcribed audio: {text}")
        return text
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

def generate_image_caption(processor, model, image):
    try:
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50, num_beams=5)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        st.info(f"Image caption: {generated_caption}")
        return generated_caption
    except Exception as e:
        st.error(f"Error generating image caption: {e}")
        return None

def translate_text(text, target_language_code, source_language_code):
    if not text or source_language_code == target_language_code:
        return text

    if target_language_code == "en" and source_language_code in ["es", "fr", "pt", "it", "ro"]:
        tokenizer = translation_models["romance_to_en"]["tokenizer"]
        model = translation_models["romance_to_en"]["model"]
        src_lang_tag = f">>{source_language_code}<<"
        input_text = f"{src_lang_tag} {text}"
    elif source_language_code == "en" and target_language_code in ["es", "fr", "pt", "it", "ro"]:
        tokenizer = translation_models["en_to_romance"]["tokenizer"]
        model = translation_models["en_to_romance"]["model"]
        tgt_lang_tag = f">>{target_language_code}<<"
        input_text = f"{tgt_lang_tag} {text}"
    else:
        st.warning(f"Translation not explicitly supported for {source_language_code} to {target_language_code} with loaded models. Returning original text.")
        return text

    try:
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        translated_tokens = model.generate(**inputs, max_new_tokens=100)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return text

@st.cache_resource
def setup_langchain_chatbot():
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
    template = """The following is a friendly conversation between a human and an AI. The AI is a helpful and polite customer support assistant for an e-commerce store.\nThe AI is designed to understand multimodal inputs, including text, speech, and image descriptions.\nThe human might provide input in different languages. The AI should respond in the human's original language if possible, otherwise in English.\nIf the human uploads an image, the AI will get a description of the image.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"""
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation.prompt = PROMPT
    return conversation

chatbot = setup_langchain_chatbot()

st.title("Multimodal E-commerce Customer Support Chatbot")
st.sidebar.header("Settings")
user_preferred_language = st.sidebar.selectbox(
    "Preferred Response Language (if different from detected)",
    options=["en", "es", "fr", "pt", "it", "ro", "auto"],
    index=0
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_original_lang" not in st.session_state:
    st.session_state.last_original_lang = "en"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input_text = st.chat_input("Type your message here...")
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

col1, col2 = st.columns(2)
with col1:
    if st.button("Speak (requires microphone)", key="speak_button"):
        st_audio_input = speech_to_text_from_microphone(recognizer)
        if st_audio_input:
            user_input_text = st_audio_input
with col2:
    if uploaded_audio and st.button("Transcribe Audio File", key="transcribe_audio_button"):
        st_audio_input = speech_to_text_from_audio_file(recognizer, uploaded_audio)
        if st_audio_input:
            user_input_text = st_audio_input

if user_input_text or uploaded_image:
    full_user_query = ""
    current_input_lang = "en"

    if uploaded_image:
        image = Image.open(uploaded_image)
        image_caption = generate_image_caption(image_processor, image_captioner, image)
        if image_caption:
            full_user_query += f" (Image description: {image_caption})"

    if user_input_text:
        try:
            current_input_lang = detect(user_input_text)
            st.info(f"Detected input language: {current_input_lang}")
        except:
            current_input_lang = "en"

        translated_to_en_text = translate_text(user_input_text, target_language_code="en", source_language_code=current_input_lang)
        full_user_query += translated_to_en_text
        st.session_state.last_original_lang = current_input_lang

    if full_user_query.strip():
        st.session_state.messages.append({"role": "user", "content": full_user_query})
        with st.chat_message("user"):
            st.markdown(full_user_query)

        with st.spinner("Thinking..."):
            llm_response_en = chatbot.predict(input=full_user_query)

            response_target_lang = user_preferred_language if user_preferred_language != "auto" else st.session_state.get("last_original_lang", "en")

            if response_target_lang != "en":
                translated_response = translate_text(llm_response_en, target_language_code=response_target_lang, source_language_code="en")
                final_response = translated_response
                st.info(f"Response translated to {response_target_lang}")
            else:
                final_response = llm_response_en

            st.session_state.messages.append({"role": "assistant", "content": final_response})
            with st.chat_message("assistant"):
                st.markdown(final_response)