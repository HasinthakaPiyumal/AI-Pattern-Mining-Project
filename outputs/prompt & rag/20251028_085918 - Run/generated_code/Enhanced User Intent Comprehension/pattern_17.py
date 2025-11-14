from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import uvicorn
import base64

# Placeholder for speech recognition library
# In a real application, you'd integrate with services like Google Speech-to-Text, AWS Transcribe, or a local model.
class SpeechRecognizer:
    def recognize(self, audio_data: bytes, language: str = "en-US") -> str:
        # Simulate speech recognition
        print(f"Simulating speech recognition for {len(audio_data)} bytes in {language}")
        # For demonstration, we'll assume a very basic recognition for known patterns or just return a placeholder.
        if audio_data and language == "en-US":
            return "I have a headache and feel nauseous."
        elif audio_data and language == "es-ES":
            return "Tengo dolor de cabeza y náuseas."
        return "Could not understand speech. Please try again."


# Placeholder for image analysis library
# In a real application, you'd integrate with services like Google Vision API, AWS Rekognition, or a local CV model.
class ImageAnalyzer:
    def analyze(self, image_data: bytes) -> str:
        # Simulate image analysis for common medical-related images (e.g., skin rash, eye condition)
        print(f"Simulating image analysis for {len(image_data)} bytes")
        # For demonstration, return a predefined description
        return "Image shows signs consistent with a skin rash, possibly dermatitis."


# Placeholder for machine translation library
# In a real application, you'd integrate with services like Google Translate, DeepL, or a local NMT model.
class Translator:
    def translate(self, text: str, target_language: str) -> str:
        print(f"Simulating translation of '{text}' to {target_language}")
        if target_language == "en" and text == "Tengo dolor de cabeza y náuseas.":
            return "I have a headache and feel nauseous."
        elif target_language == "es" and text == "I have a headache and feel nauseous.":
            return "Tengo dolor de cabeza y náuseas."
        return f"[Translated to {target_language}: {text}]"


# Placeholder for a fine-tuned Large Language Model (LLM)
# In a real application, this would be a loaded Hugging Face model, OpenAI API call, or similar.
# We'll simulate its behavior with rule-based responses and prompt engineering.
class MedicalLLM:
    def __init__(self):
        self.medical_knowledge_base = [
            "Headache: Common symptoms include pain in the head, often accompanied by sensitivity to light or sound.",
            "Nausea: A sensation of unease and discomfort in the stomach, often preceding vomiting.",
            "Skin Rash: An area of inflamed or irritated skin, often itchy or red.",
            "Dermatitis: Inflammation of the skin, characterized by itching, redness, and skin lesions.",
            "Fever: An abnormally high body temperature, usually accompanied by shivering, headache, and in severe cases, delirium."
        ]
        self.symptom_to_condition = {
            "headache": "Migraine or Tension Headache",
            "nauseous": "Gastroenteritis or Migraine",
            "skin rash": "Dermatitis or Allergic Reaction",
            "fever": "Infection"
        }

    def _infer_intent_and_clarify(self, text_input: str) -> str:
        text_input_lower = text_input.lower()
        response = ""
        inferred_symptoms = []

        if "headache" in text_input_lower:
            inferred_symptoms.append("headache")
        if "nauseous" in text_input_lower or "nausea" in text_input_lower:
            inferred_symptoms.append("nauseous")
        if "skin rash" in text_input_lower or "rash" in text_input_lower:
            inferred_symptoms.append("skin rash")
        if "fever" in text_input_lower:
            inferred_symptoms.append("fever")

        if inferred_symptoms:
            response += f"I understand you are experiencing {', and '.join(inferred_symptoms)}. "
            if len(inferred_symptoms) == 1:
                if inferred_symptoms[0] == "headache":
                    response += "Could you describe the type of headache? (e.g., throbbing, dull, sharp) And how long have you had it?"
                elif inferred_symptoms[0] == "nauseous":
                    response += "Are you experiencing any other symptoms like vomiting, diarrhea, or dizziness?"
                elif inferred_symptoms[0] == "skin rash":
                    response += "Where is the rash located? Is it itchy, painful, or blistering?"
                elif inferred_symptoms[0] == "fever":
                    response += "What is your approximate temperature? Are you experiencing chills or body aches?"
            else:
                response += "To help me understand better, could you elaborate on the onset and severity of each symptom?"
        else:
            response += "I'm having trouble identifying specific symptoms. Could you please describe your concerns in more detail?"

        return response

    def _provide_preliminary_diagnosis(self, symptoms: list) -> str:
        possible_conditions = []
        for symptom in symptoms:
            if symptom in self.symptom_to_condition:
                possible_conditions.append(self.symptom_to_condition[symptom])
        
        if possible_conditions:
            return f"Based on the symptoms you described, some possible conditions include: {', '.join(set(possible_conditions))}. Please note this is a preliminary assessment and not a definitive diagnosis. It's important to consult with a medical professional for accurate diagnosis and treatment."
        return "Based on the information provided, I cannot give a specific preliminary diagnosis. Please consult a doctor for further evaluation."

    def process_query(self, user_input: str) -> str:
        # Simulate instruction tuning and personalized learning through prompt engineering and state management (not fully implemented here for simplicity)
        # For a real LLM, this would involve a more sophisticated prompt template and potentially a RAG system.
        
        # First pass: Infer intent and ask clarifying questions
        clarification_response = self._infer_intent_and_clarify(user_input)
        
        # In a more advanced system, this would be an iterative dialogue.
        # For this example, we'll try to provide a preliminary diagnosis if enough info is present initially.
        inferred_symptoms = []
        user_input_lower = user_input.lower()
        if "headache" in user_input_lower:
            inferred_symptoms.append("headache")
        if "nauseous" in user_input_lower or "nausea" in user_input_lower:
            inferred_symptoms.append("nauseous")
        if "skin rash" in user_input_lower or "rash" in user_input_lower:
            inferred_symptoms.append("skin rash")
        if "fever" in user_input_lower:
            inferred_symptoms.append("fever")

        if inferred_symptoms and "I'm having trouble" not in clarification_response: # If some symptoms identified and initial clarification wasn't a complete failure
            diagnosis_response = self._provide_preliminary_diagnosis(inferred_symptoms)
            return f"{clarification_response}\n\n{diagnosis_response}"
        
        return clarification_response


app = FastAPI()
speech_recognizer = SpeechRecognizer()
image_analyzer = ImageAnalyzer()
translator = Translator()
medical_llm = MedicalLLM()


class TextInput(BaseModel):
    text: str
    language: str = "en"

class ImageInput(BaseModel):
    image_base64: str
    language: str = "en"


@app.post("/diagnose/text")
async def diagnose_from_text(input_data: TextInput):
    translated_text = input_data.text
    if input_data.language != "en":
        translated_text = translator.translate(input_data.text, "en")
    
    llm_response = medical_llm.process_query(translated_text)
    return {"status": "success", "original_input": input_data.text, "language": input_data.language, "llm_response": llm_response}


@app.post("/diagnose/speech")
async def diagnose_from_speech(file: UploadFile = File(...), language: str = Form("en-US")):
    audio_data = await file.read()
    recognized_text = speech_recognizer.recognize(audio_data, language)
    
    translated_text = recognized_text
    if not language.startswith("en"): # Check if not an English variant
        translated_text = translator.translate(recognized_text, "en")

    llm_response = medical_llm.process_query(translated_text)
    return {"status": "success", "recognized_text": recognized_text, "language": language, "llm_response": llm_response}


@app.post("/diagnose/image")
async def diagnose_from_image(image_base64: str = Form(...), language: str = Form("en")):
    try:
        image_data = base64.b64decode(image_base64)
    except Exception as e:
        return {"status": "error", "message": f"Invalid base64 image data: {e}"}

    analyzed_description = image_analyzer.analyze(image_data)
    
    translated_description = analyzed_description
    if language != "en":
        translated_description = translator.translate(analyzed_description, "en")

    llm_response = medical_llm.process_query(translated_description)
    return {"status": "success", "image_analysis": analyzed_description, "language": language, "llm_response": llm_response}


@app.get("/")
async def root():
    return {"message": "Welcome to the Intelligent Medical Assistant API. Use /diagnose/text, /diagnose/speech, or /diagnose/image to get preliminary diagnoses."}


if __name__ == "__main__":
    # To run this application:
    # 1. Save the code as medical_assistant_app.py
    # 2. Install uvicorn and fastapi: pip install uvicorn fastapi pydantic python-multipart
    # 3. Run from your terminal: uvicorn medical_assistant_app:app --reload
    # Then access the API at http://127.0.0.1:8000
    # You can test with tools like curl, Postman, or a web frontend.
    uvicorn.run(app, host="0.0.0.0", port=8000)
