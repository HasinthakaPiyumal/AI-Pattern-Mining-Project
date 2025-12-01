import gradio as gr
import speech_recognition as sr
from transformers import pipeline
from PIL import Image
import io

# --- Module 1: Speech Recognition ---
def transcribe_audio(audio_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Speech recognition service error; {e}"

# --- Module 2: Image Analysis (Placeholder) ---
def analyze_image(image_file):
    if image_file is None:
        return "No image uploaded."
    # In a real application, this would involve a complex medical image analysis model
    # For this example, we'll just acknowledge the image and provide a generic description.
    try:
        img = Image.open(image_file.name)
        return f"Image analysis: Detected an image of size {img.size[0]}x{img.size[1]}. Further analysis would reveal medical features."
    except Exception as e:
        return f"Error analyzing image: {e}"

# --- Module 3: Machine Translation ---
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
reverse_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

def translate_to_english(text, src_lang="es"):
    if src_lang == "en":
        return text
    # Assuming the 'translator' is pre-configured for src_lang to English
    translated = reverse_translator(text)[0]["translation_text"]
    return translated

def translate_from_english(text, dest_lang="es"):
    if dest_lang == "en":
        return text
    translated = translator(text)[0]["translation_text"]
    return translated

# --- Module 4: Knowledge Base (Placeholder) ---
medical_knowledge_base = {"fever": "Fever is an increase in body temperature above the normal range, usually 98.6°F (37°C). It's often a sign of an infection.",
                          "headache": "A headache is a pain in any region of the head. Headaches may occur on one or both sides of the head, be isolated to a certain location, radiate across the head from one point, or feel like a tight band around the head.",
                          "broken arm": "A broken arm is a fracture in one of the bones of the arm. It typically requires immobilization with a cast or splint."
                         }

def query_knowledge_base(query):
    for keyword, info in medical_knowledge_base.items():
        if keyword in query.lower():
            return info
    return None

# --- Module 5: LLM Backend (Placeholder) ---
def llm_process(combined_input, user_lang="en"):
    # In a real application, this would be an actual LLM call (e.g., OpenAI API, local Llama model)
    # For this example, we'll use a rule-based response and knowledge base lookup.

    response = ""
    kb_response = query_knowledge_base(combined_input)
    if kb_response:
        response += f"Based on medical knowledge: {kb_response} "

    if "diagnose" in combined_input.lower() or "diagnosis" in combined_input.lower():
        response += "Please provide more detailed symptoms for a potential diagnosis. I am an AI assistant and cannot provide definitive medical diagnoses. Consult a doctor."
    elif "treatment" in combined_input.lower():
        response += "Treatment recommendations depend on a specific diagnosis. Consult a healthcare professional for personalized advice."
    elif "hello" in combined_input.lower() or "hi" in combined_input.lower():
        response += "Hello! How can I assist you with your healthcare query today?"
    elif not kb_response:
        response += f"I received your query: '{combined_input}'. I am a healthcare assistant. How can I help further?"
    
    if user_lang != "en":
        response = translate_from_english(response, dest_lang=user_lang)
    return response

# --- Main Interface Logic ---
def multimodal_assistant(audio_input, image_input, text_input, language_choice):
    speech_text = ""
    image_description = ""
    user_text = text_input

    # Process audio input
    if audio_input is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_input)
        speech_text = transcribe_audio("temp_audio.wav")
        print(f"Transcribed Audio: {speech_text}")

    # Process image input
    if image_input is not None: # Gradio passes the file object directly for image components
        image_description = analyze_image(image_input)
        print(f"Image Analysis: {image_description}")

    # Consolidate text inputs
    combined_raw_text = []
    if speech_text and speech_text != "Could not understand audio":
        combined_raw_text.append(speech_text)
    if image_description and image_description != "No image uploaded.":
        combined_raw_text.append(image_description)
    if user_text:
        combined_raw_text.append(user_text)
    
    final_input_for_llm = " ".join(combined_raw_text)

    # Translate to English for LLM processing if necessary
    translated_llm_input = translate_to_english(final_input_for_llm, src_lang=language_choice)

    # LLM processing
    llm_response = llm_process(translated_llm_input, user_lang=language_choice)
    
    return llm_response

# --- Gradio Interface ---
iface = gr.Interface(
    fn=multimodal_assistant,
    inputs=[
        gr.Audio(type="filepath", label="Speak your query"),
        gr.Image(type="filepath", label="Upload Medical Image"),
        gr.Textbox(label="Type your query"),
        gr.Dropdown(["en", "es"], label="Select Language", value="en")
    ],
    outputs=gr.Textbox(label="Assistant's Response"),
    title="Multimodal Healthcare Assistant",
    description="Interact with the AI assistant using voice, images, and text in multiple languages."
)

iface.launch()