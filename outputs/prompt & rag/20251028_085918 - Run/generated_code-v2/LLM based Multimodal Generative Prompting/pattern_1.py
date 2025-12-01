import os
import io
from dotenv import load_dotenv
import speech_recognition as sr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator
from fastapi import FastAPI, UploadFile, File, Form
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
import uvicorn
import multiprocessing
import requests

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("OPENAI_API_KEY not found in environment variables.")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def transcribe_speech(audio_bytes):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return f"Speech recognition service error: {e}"
    except Exception as e:
        return f"Error during speech transcription: {e}"

def analyze_image(image_bytes):
    try:
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = blip_processor(raw_image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        description = blip_processor.decode(out[0], skip_special_tokens=True)[0]
        return description
    except Exception as e:
        return f"Error analyzing image: {e}"

def translate_text(text, dest_lang='en'):
    if not text.strip():
        return ""
    try:
        translator = Translator()
        translated = translator.translate(text, dest=dest_lang)
        return translated.text
    except Exception as e:
        return f"Error translating text: {e}"

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0.3)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent for an e-commerce platform. Provide concise and helpful responses based on the user's query and any provided context."),
    ("user", "{query}")
])
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(
    text_input: str = Form(""),
    audio_file: UploadFile = File(None),
    image_file: UploadFile = File(None)
):
    context_parts = []

    if audio_file:
        audio_bytes = await audio_file.read()
        transcribed_text = transcribe_speech(audio_bytes)
        if transcribed_text:
            context_parts.append(f"User spoke: {transcribed_text}")

    if image_file:
        image_bytes = await image_file.read()
        image_description = analyze_image(image_bytes)
        if image_description:
            context_parts.append(f"User uploaded image described as: {image_description}")

    if text_input:
        try:
            detected_lang = Translator().detect(text_input).lang
            if detected_lang != 'en':
                translated_text = translate_text(text_input, dest_lang='en')
                context_parts.append(f"User typed (original '{detected_lang}'): {translated_text}")
            else:
                context_parts.append(f"User typed: {text_input}")
        except Exception as e:
            context_parts.append(f"User typed: {text_input} (translation error: {e})")

    combined_query = " ".join(context_parts)
    if not combined_query.strip():
        return {"response": "Please provide some input (text, speech, or image)."}

    try:
        llm_response = chain.invoke({"query": combined_query})
        return {"response": llm_response}
    except Exception as e:
        return {"response": f"An error occurred with the LLM: {e}"}

def gradio_interface(text_input, audio_input, image_input):
    FASTAPI_URL = "http://127.0.0.1:8000/chat"
    files = {}
    data = {"text_input": text_input if text_input else ""}

    if audio_input:
        files["audio_file"] = ("audio.wav", open(audio_input, "rb").read(), "audio/wav")
    if image_input:
        files["image_file"] = ("image.png", open(image_input, "rb").read(), "image/png")

    try:
        response = requests.post(FASTAPI_URL, files=files, data=data)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the FastAPI backend. Make sure it's running."
    except requests.exceptions.RequestException as e:
        return f"Error during request to FastAPI: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Text Query"),
        gr.Audio(type="filepath", label="Voice Query"),
        gr.Image(type="filepath", label="Image Query")
    ],
    outputs="text",
    title="Smart Multimodal Customer Support",
    description="Ask a question using text, voice, or an image. The AI agent will process it and respond."
)

def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

def run_gradio():
    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_fastapi)
    p2 = multiprocessing.Process(target=run_gradio)

    p1.start()
    p2.start()

    p1.join()
    p2.join()