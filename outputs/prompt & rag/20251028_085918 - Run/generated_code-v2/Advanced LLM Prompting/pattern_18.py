import spacy
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from typing import Dict, List

# --- 1. Medical Dictionary Service (Mock) ---
class MedicalDictionary:
    def __init__(self):
        self._definitions = {
            "discharge": {
                "en": "1. The process of releasing a patient from a hospital. 2. A fluid emission from the body.",
                "es": "1. El proceso de dar de alta a un paciente de un hospital. 2. Una emisión de fluidos del cuerpo.",
                "fr": "1. Le processus de libération d'un patient d'un hôpital. 2. Une émission de fluide du corps."
            },
            "fever": {
                "en": "An abnormally high body temperature, usually accompanied by shivering, headache, and in severe instances, delirium.",
                "es": "Una temperatura corporal anormalmente alta, generalmente acompañada de escalofríos, dolor de cabeza y, en casos graves, delirio.",
                "fr": "Une température corporelle anormalement élevée, généralement accompagnée de frissons, de maux de tête et, dans les cas graves, de délire."
            },
            "prescription": {
                "en": "An instruction written by a medical practitioner that authorizes a patient to be provided a medicine or treatment.",
                "es": "Una instrucción escrita por un médico que autoriza a un paciente a recibir un medicamento o tratamiento.",
                "fr": "Une instruction écrite par un médecin qui autorise un patient à recevoir un médicament ou un traitement."
            }
        }

    def get_definitions(self, term: str, target_lang: str) -> str:
        term_lower = term.lower()
        if term_lower in self._definitions:
            return self._definitions[term_lower].get(target_lang, self._definitions[term_lower]["en"])
        return ""

# --- 2. Medical Term Extractor ---
nlp = spacy.load("en_core_web_sm")

def extract_medical_terms(text: str, medical_dictionary: MedicalDictionary) -> List[str]:
    doc = nlp(text)
    extracted_terms = []
    # Simple approach: identify known medical terms from our mock dictionary
    # In a real scenario, this would involve sophisticated NER (e.g., with scispacy or custom models)
    for term in medical_dictionary._definitions.keys():
        if term in text.lower():
            extracted_terms.append(term)
    return list(set(extracted_terms)) # Remove duplicates

# --- 3. Prompt Augmentation Logic ---
def augment_prompt(original_text: str, terms_with_definitions: Dict[str, str], target_lang: str) -> str:
    augmentation = f"Please translate the following medical text into {target_lang}. Use the provided definitions for context:\n\n"
    for term, definition in terms_with_definitions.items():
        augmentation += f"Definition for '{term}': {definition}\n"
    
    augmentation += f"\nOriginal text: {original_text}"
    return augmentation

# --- 4. Generative AI Translation Model (Mock) ---
def mock_genai_translate(augmented_prompt: str, target_lang: str) -> str:
    # Simulate a call to a GenAI model
    # In a real application, this would involve an API call to OpenAI, Google Gemini, etc.
    # For demonstration, we'll just return a placeholder that includes the original text and target language
    
    # A very simplistic 'translation' for demonstration
    if "discharge" in augmented_prompt.lower() and "patient from a hospital" in augmented_prompt.lower():
        if target_lang == "es":
            return "El paciente ha sido dado de alta del hospital."
        elif target_lang == "fr":
            return "Le patient a été libéré de l'hôpital."

    return f"[MOCK TRANSLATION to {target_lang}]: {augmented_prompt}\n(Actual GenAI translation would go here, considering definitions.)"

# --- FastAPI Application ---
app = FastAPI()
medical_dict = MedicalDictionary()

class TranslationRequest(BaseModel):
    text: str
    target_language: str

@app.post("/translate_medical_text")
async def translate_medical_text_endpoint(request: TranslationRequest):
    text = request.text
    target_language = request.target_language

    # 1. Extract medical terms
    extracted_terms = extract_medical_terms(text, medical_dict)
    
    # 2. Retrieve contextual definitions
    terms_with_definitions = {}
    for term in extracted_terms:
        definition = medical_dict.get_definitions(term, target_language)
        if definition:
            terms_with_definitions[term] = definition
    
    # 3. Augment prompt
    augmented_prompt = augment_prompt(text, terms_with_definitions, target_language)
    
    # 4. Translate with Augmented Prompt (mocked GenAI)
    translated_text = mock_genai_translate(augmented_prompt, target_language)
    
    return {"translated_text": translated_text}

# --- Gradio Interface ---

def gradio_interface(text: str, target_language: str) -> str:
    # This function will call the FastAPI backend
    import requests
    try:
        response = requests.post("http://127.0.0.1:8000/translate_medical_text", json={
            "text": text,
            "target_language": target_language
        })
        response.raise_for_status()
        return response.json()["translated_text"]
    except requests.exceptions.ConnectionError:
        return "Error: FastAPI backend not running. Please start the FastAPI server first."
    except requests.exceptions.RequestException as e:
        return f"Error during translation: {e}"

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter medical text here..."),
        gr.Dropdown(["es", "fr", "en"], label="Target Language", value="es")
    ],
    outputs="textbox",
    title="Medical Terminology Translator with Contextual Dictionary Augmentation",
    description="Translate medical text using a Chain of Dictionary pattern for improved accuracy."
)

# To run this application:
# 1. Save this code as main.py
# 2. Install dependencies: pip install fastapi uvicorn spacy gradio requests
# 3. Download spacy model: python -m spacy download en_core_web_sm
# 4. Start the FastAPI server: uvicorn main:app --reload
# 5. Then, in a separate terminal, launch Gradio: python main.py (or run iface.launch() directly)
# Note: Gradio will attempt to launch a separate web server which will then connect to the FastAPI backend.
# For a single-script runnable demo, you can comment out the FastAPI app.run() and directly call the logic from gradio_interface
# but the request was to show both an API and a UI communicating.

# To run the Gradio interface directly for testing (without needing to run uvicorn separately) uncomment below and remove the FastAPI setup:
# if __name__ == "__main__":
#     iface.launch(share=True)

# The current setup expects uvicorn to be run separately for the FastAPI app.
# For a combined execution, one would typically embed FastAPI within Gradio using Blocks, or run them as separate processes.
# Given the request for both, the current structure is a clear separation of concerns.
