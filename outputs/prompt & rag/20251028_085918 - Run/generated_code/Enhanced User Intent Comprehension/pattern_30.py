from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import speech_recognition as sr
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import io
import os

app = FastAPI()

r = sr.Recognizer()

image_captioner = None
try:
    image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
except Exception:
    pass

translator_en_es = None
translator_es_en = None
try:
    translator_en_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
    translator_es_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
except Exception:
    pass

llm_pipeline = None
try:
    llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    llm_model = AutoModelForCausalLM.from_pretrained("gpt2")
    llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
except Exception:
    pass


class InputProcessor:
    def process_speech(self, audio_file_content: bytes) -> Optional[str]:
        try:
            audio_data = sr.AudioData(audio_file_content, 16000, 2)
            text = r.recognize_google(audio_data)
            return text
        except Exception:
            return None

    def analyze_image(self, image_file_content: bytes) -> Optional[str]:
        if image_captioner:
            try:
                image = Image.open(io.BytesIO(image_file_content))
                caption_result = image_captioner(image)
                return caption_result[0]["generated_text"] if caption_result else None
            except Exception:
                return None
        else:
            return "Simulated image analysis: 'Damaged item detected.'"

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang:
            return text
        
        if source_lang == "en" and target_lang == "es" and translator_en_es:
            try:
                translated = translator_en_es(text)
                return translated[0]["translation_text"]
            except Exception:
                pass
        elif source_lang == "es" and target_lang == "en" and translator_es_en:
            try:
                translated = translator_es_en(text)
                return translated[0]["translation_text"]
            except Exception:
                pass
        
        return f"Simulated translation from {source_lang} to {target_lang}: {text}"

    def preprocess_text(self, text: str) -> str:
        return text.lower().strip()


class CustomerSupportLLM:
    def __init__(self, llm_pipeline_instance):
        self.llm_pipeline = llm_pipeline_instance

    def get_llm_response(self, text: str) -> str:
        if not self.llm_pipeline:
            if "order status" in text:
                return "Simulated: Your order is currently being processed and is expected to ship within 2-3 business days. Would you like to track it?"
            elif "damaged" in text or "broken" in text:
                return "Simulated: I'm sorry to hear your item arrived damaged. Please provide your order number and we'll initiate a replacement or refund process."
            elif "return" in text:
                return "Simulated: To initiate a return, please visit our returns page and follow the instructions. Do you need help finding it?"
            elif "product inquiry" in text or "about product" in text:
                return "Simulated: Can you please specify which product you are interested in? I can provide details about its features, availability, or pricing."
            else:
                return "Simulated: Thank you for contacting customer support. How can I assist you further?"

        prompt = f"The user says: '{text}'. As an e-commerce customer support agent, respond to their query. If the intent is unclear, ask a clarifying question. Keep your response concise."
        
        try:
            response = self.llm_pipeline(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                truncation=True,
                return_full_text=False
            )
            return response[0]["generated_text"].strip()
        except Exception:
            return "I apologize, but I am currently experiencing technical difficulties. Please try again later."


input_processor = InputProcessor()
customer_support_llm = CustomerSupportLLM(llm_pipeline)


@app.post("/support")
async def support_endpoint(
    text_input: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    language_preference: str = Form("en")
):
    processed_text = ""
    image_description = ""

    if audio_file:
        audio_content = await audio_file.read()
        speech_text = input_processor.process_speech(audio_content)
        if speech_text:
            processed_text += speech_text + " "

    if image_file:
        image_content = await image_file.read()
        image_description = input_processor.analyze_image(image_content)
        if image_description:
            processed_text += image_description + " "

    if text_input:
        processed_text += text_input

    final_input_text = input_processor.preprocess_text(processed_text.strip())

    if not final_input_text:
        return {"response": "Please provide some input (text, audio, or image)."}

    if language_preference != "en":
        translated_to_en = input_processor.translate_text(final_input_text, language_preference, "en")
        llm_input = translated_to_en
    else:
        llm_input = final_input_text

    llm_raw_response = customer_support_llm.get_llm_response(llm_input)

    if language_preference != "en":
        final_response = input_processor.translate_text(llm_raw_response, "en", language_preference)
    else:
        final_response = llm_raw_response

    return {"response": final_response}