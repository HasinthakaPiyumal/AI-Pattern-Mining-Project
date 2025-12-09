import speech_recognition as sr
import cv2
from PIL import Image
from googletrans import Translator
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import uvicorn
import io
import numpy as np
from transformers import pipeline


class MultimodalInput(BaseModel):
    text_input: str = None
    target_language: str = "en"
    patient_context: str = ""


app = FastAPI()

# Initialize modules
recognizer = sr.Recognizer()
translator = Translator()

# Placeholder for a simple LLM. In a real application, you'd load a more sophisticated model.
# Using a conversational pipeline for demonstration purposes.
llm_pipeline = pipeline("text-generation", model="gpt2")

# Placeholder for a simple image analysis model. 
# In a real application, you'd load a specialized medical imaging model.
# For demonstration, it just returns a mock description.
def analyze_image_mock(image_bytes):
    try:
        # Convert bytes to numpy array for OpenCV
        image_np = np.array(Image.open(io.BytesIO(image_bytes)))
        # Example: Check image dimensions
        height, width, _ = image_np.shape
        if height > 500 and width > 500:
            return "Large image detected, potentially an X-ray. Further analysis required by a specialized model."
        else:
            return "Small image detected, possibly a photo of a symptom. Description based on general visual features."
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


def transcribe_audio(audio_bytes):
    try:
        with io.BytesIO(audio_bytes) as audio_file_io:
            with sr.AudioFile(audio_file_io) as source:
                audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Speech Recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"


def translate_text(text, dest_language):
    try:
        if not text:
            return ""
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception as e:
        return f"Error translating text: {str(e)}"


def process_with_llm(transcribed_text, image_description, text_input, patient_context):
    combined_input = ""
    if patient_context:
        combined_input += f"Patient Context: {patient_context}. "
    if transcribed_text:
        combined_input += f"Spoken notes: {transcribed_text}. "
    if image_description:
        combined_input += f"Image analysis: {image_description}. "
    if text_input:
        combined_input += f"Additional text input: {text_input}. "

    if not combined_input:
        return "No relevant input provided for analysis."

    prompt = f"Based on the following information, provide a medical assessment, potential diagnosis, and treatment suggestions: {combined_input} "

    try:
        # Using LLM pipeline to generate a response
        # For a real application, you'd craft more specific prompts and potentially use `langchain` or `llama_index`
        # for more advanced RAG and tool orchestration.
        response = llm_pipeline(prompt, max_new_tokens=200, num_return_sequences=1)
        return response[0]["generated_text"]
    except Exception as e:
        return f"Error processing with LLM: {str(e)}"


@app.post("/assist")
async def smart_clinic_assistant(
    audio_file: UploadFile = File(None),
    image_file: UploadFile = File(None),
    text_input: str = Form(None),
    target_language: str = Form("en"),
    patient_context: str = Form(""),
):
    transcribed_text = ""
    image_description = ""
    processed_text_input = text_input

    if audio_file:
        audio_bytes = await audio_file.read()
        transcribed_text = transcribe_audio(audio_bytes)

    if image_file:
        image_bytes = await image_file.read()
        image_description = analyze_image_mock(image_bytes)

    if text_input and target_language != "en":
        processed_text_input = translate_text(text_input, target_language)

    llm_response = process_with_llm(
        transcribed_text=transcribed_text,
        image_description=image_description,
        text_input=processed_text_input,
        patient_context=patient_context,
    )

    return {"response": llm_response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# --- Streamlit UI Example (Conceptual - run separately) ---
# To run this Streamlit app, save it as a separate .py file (e.g., frontend.py)
# and run `streamlit run frontend.py` in your terminal.

# import streamlit as st
# import requests

# st.title("Smart Clinic Assistant")

# st.header("Upload Inputs")

# audio_upload = st.file_uploader("Upload Audio (e.g., doctor's notes)", type=["wav", "mp3"])
# image_upload = st.file_uploader("Upload Medical Image (e.g., X-ray, symptom photo)", type=["png", "jpg", "jpeg"])
# text_input_field = st.text_area("Type additional notes or patient description")
# target_lang = st.selectbox("Target Language for Translation (if input text is not English)", ["en", "es", "fr", "de", "zh-cn"])
# patient_context_field = st.text_area("Provide patient's medical history or context")

# if st.button("Get Assistant's Insights"):
#     files = {}
#     data = {
#         "text_input": text_input_field,
#         "target_language": target_lang,
#         "patient_context": patient_context_field,
#     }

#     if audio_upload:
#         files["audio_file"] = (audio_upload.name, audio_upload.getvalue(), audio_upload.type)
#     if image_upload:
#         files["image_file"] = (image_upload.name, image_upload.getvalue(), image_upload.type)
    
#     st.write("Sending request to assistant...")
#     try:
#         response = requests.post("http://localhost:8000/assist", files=files, data=data)
#         if response.status_code == 200:
#             st.subheader("Assistant's Response:")
#             st.write(response.json()["response"])
#         else:
#             st.error(f"Error from assistant: {response.status_code} - {response.text}")
#     except requests.exceptions.ConnectionError:
#         st.error("Could not connect to the FastAPI backend. Make sure it's running at http://localhost:8000.")


