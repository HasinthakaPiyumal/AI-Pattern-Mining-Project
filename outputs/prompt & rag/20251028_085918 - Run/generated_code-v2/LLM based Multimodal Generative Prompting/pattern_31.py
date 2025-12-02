import io
import speech_recognition as sr
from PIL import Image
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ChatResponse(BaseModel):
    response: str
    original_query: str
    processed_inputs: dict

class MockLLM:
    def __init__(self):
        pass

    def generate_response(self, combined_input: str) -> str:
        if "product issue" in combined_input.lower():
            return "I understand you have a product issue. Please describe it in more detail, perhaps with an image."
        elif "shipping" in combined_input.lower():
            return "I can help with shipping inquiries. What is your order number?"
        elif "hello" in combined_input.lower() or "hi" in combined_input.lower():
            return "Hello! How can I assist you today?"
        else:
            return f"Thank you for your input: '{combined_input}'. I am processing your request. How else can I help?"

llm = MockLLM()

def transcribe_audio(audio_bytes: bytes) -> str:
    try:
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = r.record(source)
            text = r.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""
    except Exception:
        return ""

def analyze_image(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        mode = img.mode
        description = f"The user uploaded an image. Its dimensions are {width}x{height} and color mode is {mode}. "
        description += "Based on a quick scan, it looks like a typical e-commerce product image."
        return description
    except Exception:
        return "Failed to analyze image."

translator_en_fr = None
translator_fr_en = None
try:
    translator_en_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
    translator_fr_en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
except Exception:
    pass

def translate_text(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    if source_lang == target_lang:
        return text

    detected_lang = "en"
    if source_lang != "auto":
        detected_lang = source_lang
    else:
        if any(char in text for char in 'éàèùç'):
            detected_lang = "fr"

    if detected_lang == target_lang:
        return text

    try:
        if detected_lang == "en" and target_lang == "fr" and translator_en_fr:
            translated = translator_en_fr(text)[0]['translation_text']
            return translated
        elif detected_lang == "fr" and target_lang == "en" and translator_fr_en:
            translated = translator_fr_en(text)[0]['translation_text']
            return translated
        else:
            return f"[Translated from {detected_lang} to {target_lang}] {text}"
    except Exception:
        return text

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_bot(
    text_input: str = Form(""),
    audio_file: UploadFile = File(None),
    image_file: UploadFile = File(None),
    target_language: str = Form("en")
):
    combined_query_parts = []
    processed_inputs = {}

    original_query_text = text_input

    if audio_file:
        audio_bytes = await audio_file.read()
        transcribed_text = transcribe_audio(audio_bytes)
        if transcribed_text:
            combined_query_parts.append(f"User spoke: \"{transcribed_text}\"")
            processed_inputs["audio"] = transcribed_text
            original_query_text = transcribed_text

    if image_file:
        image_bytes = await image_file.read()
        image_description = analyze_image(image_bytes)
        if image_description:
            combined_query_parts.append(f"User uploaded an image. Image analysis: \"{image_description}\"")
            processed_inputs["image"] = image_description

    if text_input:
        text_for_llm = text_input
        if target_language != "en":
            translated_text_to_en = translate_text(text_input, source_lang=target_language, target_lang="en")
            if translated_text_to_en != text_input:
                combined_query_parts.append(f"User typed (originally in {target_language}): \"{text_input}\" which translates to \"{translated_text_to_en}\"")
                processed_inputs["text_original"] = text_input
                processed_inputs["text_translated_to_en"] = translated_text_to_en
                text_for_llm = translated_text_to_en
            else:
                combined_query_parts.append(f"User typed: \"{text_input}\"")
                processed_inputs["text"] = text_input
        else:
            combined_query_parts.append(f"User typed: \"{text_input}\"")
            processed_inputs["text"] = text_input


    if not combined_query_parts:
        raise HTTPException(status_code=400, detail="No valid input provided (text, audio, or image).")

    full_llm_query = " ".join(combined_query_parts).strip()

    llm_response = llm.generate_response(full_llm_query)

    final_response = translate_text(llm_response, source_lang="en", target_lang=target_language)

    return ChatResponse(
        response=final_response,
        original_query=original_query_text,
        processed_inputs=processed_inputs
    )

@app.get("/")
async def root():
    return {"message": "Multimodal Customer Support Chatbot API. Use /chat/ endpoint for interactions."}