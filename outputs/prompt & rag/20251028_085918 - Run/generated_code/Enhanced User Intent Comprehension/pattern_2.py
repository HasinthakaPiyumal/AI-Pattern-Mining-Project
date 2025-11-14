
from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import uvicorn

from speech_recognizer import SpeechRecognizer
from image_analyzer import ImageAnalyzer
from translator import Translator
from llm_core import LLMHealthcareAssistant
from patient_profile_manager import PatientProfileManager

app = FastAPI()

speech_recognizer = SpeechRecognizer()
image_analyzer = ImageAnalyzer()
translator = Translator()
llm_assistant = LLMHealthcareAssistant()
patient_profile_manager = PatientProfileManager()

@app.post("/query/")
async def handle_query(
    text_input: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    language: str = Form("en") # ISO 639-1 code
):
    processed_text = None
    response_language = language

    # 1. Handle Multi-modal Input
    if audio_file:
        audio_content = await audio_file.read()
        processed_text = speech_recognizer.recognize_speech(audio_content)
        print(f"Speech recognized: {processed_text}")

    if image_file:
        image_content = await image_file.read()
        image_analysis_result = image_analyzer.analyze_image(image_content)
        # Integrate image analysis result into text for LLM
        if processed_text:
            processed_text += f"\n[Image Analysis: {image_analysis_result}]"
        else:
            processed_text = f"[Image Analysis: {image_analysis_result}]"
        print(f"Image analyzed: {image_analysis_result}")

    if text_input:
        if processed_text:
            processed_text += f"\n[User Text Input: {text_input}]"
        else:
            processed_text = text_input
        print(f"Direct text input: {text_input}")

    if not processed_text:
        return {"error": "No valid input provided (text, audio, or image)."}

    # 2. Handle Multi-lingual Input (if not English initially)
    original_query_lang = translator.detect_language(processed_text)
    if original_query_lang != "en" and original_query_lang != "unknown": # Assuming English as primary system language
        translated_query = translator.translate_text(processed_text, original_query_lang, "en")
        print(f"Query translated from {original_query_lang} to en: {translated_query}")
        query_for_llm = translated_query
        response_language = original_query_lang # Respond in user's original language
    else:
        query_for_llm = processed_text

    # 3. Leverage LLM for Intent Comprehension and Response Generation
    # Simulate personalized context (e.g., from patient_id)
    patient_id = "patient_123" # This would come from authentication/session
    patient_context = patient_profile_manager.get_patient_context(patient_id)
    print(f"Patient context for {patient_id}: {patient_context}")

    llm_response = llm_assistant.process_query(query_for_llm, patient_context)
    print(f"LLM Raw Response: {llm_response}")

    # 4. Translate Response back to User's Language if needed
    final_response_text = llm_response
    if response_language != "en":
        final_response_text = translator.translate_text(llm_response, "en", response_language)
        print(f"Response translated from en to {response_language}: {final_response_text}")

    # 5. Update personalized learning (simplified)
    patient_profile_manager.update_patient_context(patient_id, query_for_llm, llm_response)

    return {"response": final_response_text}

@app.get("/")
async def root():
    return {"message": "Intelligent Healthcare Assistant is running! Use /query to interact."}

if __name__ == "__main__":
    # To run: uvicorn main:app --reload --port 8000
    # Example curl commands for testing:
    # Text: curl -X POST "http://127.0.0.1:8000/query/" -F "text_input=What are the symptoms of flu?" -F "language=en"
    # Audio (requires an audio file named test.wav): curl -X POST "http://127.0.0.1:8000/query/" -F "audio_file=@test.wav" -F "language=en"
    # Image (requires an image file named test.jpg): curl -X POST "http://127.0.0.1:8000/query/" -F "image_file=@test.jpg" -F "text_input=What is this rash?" -F "language=en"
    print("Run with: uvicorn main:app --reload --port 8000")
    print("Refer to comments in main.py for example curl commands.")
