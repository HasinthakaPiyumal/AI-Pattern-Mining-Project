from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
from PIL import Image
import io
import speech_recognition as sr

app = FastAPI()

class MockSpeechRecognizerBackend:
    def recognize_google(self, audio_data, language="en-US"):
        print(f"Mock Speech Recognition for language: {language}")
        return "this is a test audio input from speech"

_mock_recognizer_backend = MockSpeechRecognizerBackend()

def analyze_image_mock(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        return f"a product image with dimensions {width}x{height} showing a generic item"
    except Exception as e:
        return f"could not analyze image: {e}"

def translate_text_mock(text: str, source_lang: str, target_lang: str = "en") -> str:
    if source_lang == "es" and target_lang == "en":
        return text.replace("Hola", "Hello").replace("producto", "product").replace("orden", "order")
    elif source_lang == "fr" and target_lang == "en":
        return text.replace("Bonjour", "Hello").replace("produit", "product").replace("commande", "order")
    return text

def unify_inputs(text: Optional[str] = None, speech_text: Optional[str] = None,
                 image_description: Optional[str] = None, original_lang: str = "en") -> str:
    parts = []
    if speech_text:
        parts.append(f"User said (speech, original lang: {original_lang}): \"{speech_text}\".")
    if image_description:
        parts.append(f"User showed (image): \"{image_description}\".")
    if text:
        parts.append(f"User typed (text, original lang: {original_lang}): \"{text}\".")

    if not parts:
        return "No discernible input."

    return " ".join(parts)

def llm_process_query(unified_input: str) -> dict:
    intent = "unknown"
    response_text = "I'm sorry, I couldn't understand your request. Please provide more details."

    unified_input_lower = unified_input.lower()

    if "product" in unified_input_lower and ("details" in unified_input_lower or "information" in unified_input_lower):
        intent = "get_product_info"
        response_text = "To get product details, please provide the product name or ID."
    elif "order" in unified_input_lower and ("status" in unified_input_lower or "track" in unified_input_lower):
        intent = "check_order_status"
        response_text = "Please provide your order number to track its status."
    elif "return" in unified_input_lower or "refund" in unified_input_lower:
        intent = "initiate_return_refund"
        response_text = "For returns or refunds, please provide your order number and the reason for return."
    elif "hello" in unified_input_lower or "hi" in unified_input_lower:
        intent = "greeting"
        response_text = "Hello! How can I assist you with your e-commerce needs today?"
    elif "test audio input" in unified_input_lower:
        intent = "speech_test"
        response_text = "I processed your audio input from speech. How can I help further?"
    elif "generic item" in unified_input_lower:
        intent = "image_description_test"
        response_text = "I processed your image. Please tell me more about what you're looking for or if you have any questions about it."
    elif "what is this product" in unified_input_lower and "image" in unified_input_lower:
        intent = "identify_product_from_image"
        response_text = "Based on the image, I see a generic item. Can you provide more details like brand or model number for specific product identification?"


    return {
        "intent": intent,
        "response": response_text,
        "debug_unified_input": unified_input
    }

@app.post("/text_input")
async def handle_text_input(
    text: str = Form(...),
    language: str = Form("en")
):
    processed_text = translate_text_mock(text, source_lang=language, target_lang="en") if language != "en" else text
    unified_input = unify_inputs(text=processed_text, original_lang=language)
    llm_output = llm_process_query(unified_input)
    return llm_output

@app.post("/speech_input")
async def handle_speech_input(
    audio_file: UploadFile = File(...),
    language: str = Form("en")
):
    _ = await audio_file.read()
    recognizer = sr.Recognizer()
    recognizer.recognize_google = _mock_recognizer_backend.recognize_google

    try:
        speech_text = recognizer.recognize_google(sr.AudioData(b'', 16000, 2), language=language)
        
        processed_speech_text = translate_text_mock(speech_text, source_lang=language, target_lang="en") if language != "en" else speech_text
        unified_input = unify_inputs(speech_text=processed_speech_text, original_lang=language)
        llm_output = llm_process_query(unified_input)
        return llm_output
    except sr.UnknownValueError:
        return {"error": "Speech recognition could not understand audio"}
    except sr.RequestError as e:
        return {"error": f"Could not request results from speech recognition service; {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during speech processing: {e}"}

@app.post("/image_input")
async def handle_image_input(
    image_file: UploadFile = File(...)
):
    image_content = await image_file.read()
    image_description = analyze_image_mock(image_content)
    unified_input = unify_inputs(image_description=image_description)
    llm_output = llm_process_query(unified_input)
    return llm_output

@app.post("/multimodal_input")
async def handle_multimodal_input(
    text: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    language: str = Form("en")
):
    speech_text = None
    image_description = None

    if audio_file:
        _ = await audio_file.read()
        recognizer = sr.Recognizer()
        recognizer.recognize_google = _mock_recognizer_backend.recognize_google
        try:
            speech_text = recognizer.recognize_google(sr.AudioData(b'', 16000, 2), language=language)
        except sr.UnknownValueError:
            return {"error": "Speech recognition could not understand audio in multimodal input"}
        except sr.RequestError as e:
            return {"error": f"Could not request results from speech recognition service in multimodal input; {e}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred during speech processing in multimodal input: {e}"}

    if image_file:
        image_content = await image_file.read()
        image_description = analyze_image_mock(image_content)

    processed_text = translate_text_mock(text, source_lang=language, target_lang="en") if text and language != "en" else text
    processed_speech_text = translate_text_mock(speech_text, source_lang=language, target_lang="en") if speech_text and language != "en" else speech_text

    unified_input = unify_inputs(
        text=processed_text,
        speech_text=processed_speech_text,
        image_description=image_description,
        original_lang=language
    )
    llm_output = llm_process_query(unified_input)
    return llm_output