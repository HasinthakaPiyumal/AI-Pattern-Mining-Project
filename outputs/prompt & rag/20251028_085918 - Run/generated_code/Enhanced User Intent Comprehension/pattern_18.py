import os
import uuid
from typing import Optional

import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import pipeline
import uvicorn

# --- 1. FastAPI Application Setup ---
app = FastAPI(title="Smart Customer Support Assistant")

# In-memory session store for demonstration
sessions = {}

class TextRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

# --- 2. MultimodalProcessor Class ---
class MultimodalProcessor:
    def process_text(self, text: str) -> str:
        """Processes plain text input."""
        return text

    async def process_audio(self, audio_file: UploadFile) -> str:
        """Converts audio file to text using SpeechRecognition."""
        recognizer = sr.Recognizer()
        audio_content = await audio_file.read()
        audio_filename = f"temp_audio_{uuid.uuid4()}.wav"
        try:
            with open(audio_filename, "wb") as f:
                f.write(audio_content)
            
            with sr.AudioFile(audio_filename) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data) # Using Google Web Speech API for simplicity
                return text
        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Speech Recognition could not understand audio")
        except sr.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Could not request results from Google Speech Recognition service; {e}")
        finally:
            if os.path.exists(audio_filename):
                os.remove(audio_filename)

    async def process_image(self, image_file: UploadFile) -> str:
        """Saves image and generates a placeholder description."""
        image_content = await image_file.read()
        image_filename = f"temp_image_{uuid.uuid4()}.png"
        try:
            with open(image_filename, "wb") as f:
                f.write(image_content)
            
            # For a real application, you'd use a Vision-Language Model here
            # Example: from transformers import BlipForConditionalGeneration, BlipProcessor
            # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            # raw_image = Image.open(image_filename).convert('RGB')
            # inputs = processor(raw_image, return_tensors="pt")
            # out = model.generate(**inputs)
            # description = processor.decode(out[0], skip_special_tokens=True)

            # Placeholder description
            return f"User uploaded an image. Filename: {image_file.filename}. This image appears to be a product photo or a screenshot related to an order. Further analysis would be needed to extract specific details from the image."
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")
        finally:
            if os.path.exists(image_filename):
                os.remove(image_filename)

# --- 3. IntentRecognizer Class ---
class IntentRecognizer:
    def __init__(self):
        # Using a zero-shot classification pipeline for intent recognition
        # This model will be downloaded on first run.
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.candidate_labels = [
            "order_status",
            "product_inquiry",
            "return_request",
            "technical_support",
            "account_management",
            "payment_issue",
            "delivery_issue",
            "general_query",
            "greeting",
            "farewell"
        ]

    def recognize_intent(self, text: str) -> str:
        """Recognizes the user's intent from the given text."""
        if not text.strip():
            return "general_query"
            
        # Instruction tuning simulated via prompt engineering
        # We can add a more descriptive prompt to guide the model
        prompt = f"The user is asking a question related to customer support. What is the primary intent of this query?\nQuery: \"{text}\""
        
        # Limit sequence length for very long inputs if needed
        if len(prompt) > 512: # BART max length is 1024, but keeping some buffer
            prompt = prompt[:512]

        try:
            # The zero-shot classifier returns a list of dictionaries, we take the top one
            result = self.classifier(prompt, self.candidate_labels, multi_label=False)
            # Ensure result['labels'] and result['scores'] are sorted by score in descending order
            # The pipeline does this by default, so we just take the first item.
            detected_intent = result['labels'][0]
            score = result['scores'][0]
            
            # Basic thresholding to prevent highly uncertain classifications
            if score < 0.6: # Adjust threshold as needed
                return "general_query"
                
            return detected_intent
        except Exception as e:
            print(f"Error during intent recognition: {e}")
            return "general_query" # Fallback if something goes wrong

# --- 4. DialogueManager Class ---
class DialogueManager:
    def __init__(self):
        self.responses = {
            "greeting": "Hello! How can I assist you today?",
            "farewell": "Goodbye! Have a great day.",
            "order_status": "Please provide your order number, and I can check its status for you.",
            "product_inquiry": "I can help with product inquiries. What product are you interested in, or what information are you looking for?",
            "return_request": "To initiate a return, please provide your order number and the reason for the return.",
            "technical_support": "I understand you need technical support. Can you describe the issue in more detail?",
            "account_management": "For account management, please specify what you'd like to do, e.g., update your profile, change password.",
            "payment_issue": "If you're experiencing a payment issue, please describe it. Do you need help with a transaction or a billing error?",
            "delivery_issue": "Regarding a delivery issue, please tell me your order number and what problem you're encountering with the delivery.",
            "general_query": "I'm sorry, I'm not entirely sure how to help with that. Could you please rephrase your request or provide more details?"
        }

    def manage_dialogue(self, user_input_text: str, detected_intent: str, image_description: Optional[str] = None, session_id: str = "default") -> str:
        """Manages the conversational flow and generates responses."""
        
        if session_id not in sessions:
            sessions[session_id] = {"history": []}
        
        # Update session history (simple example)
        sessions[session_id]["history"].append({"user": user_input_text, "intent": detected_intent, "image_desc": image_description})
        
        response = self.responses.get(detected_intent, self.responses["general_query"])

        if image_description:
            response += f"\nBased on the image you shared: {image_description}"
        
        return response

# Initialize components
multimodal_processor = MultimodalProcessor()
intent_recognizer = IntentRecognizer()
dialogue_manager = DialogueManager()

# --- FastAPI Endpoints ---

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Smart Customer Support Assistant is running!"}

@app.post("/chat", summary="Process Text Input")
async def chat_text(request: TextRequest):
    """Processes a text message and returns a conversational response."""
    processed_text = multimodal_processor.process_text(request.text)
    detected_intent = intent_recognizer.recognize_intent(processed_text)
    response = dialogue_manager.manage_dialogue(processed_text, detected_intent, session_id=request.session_id or "default")
    return {"response": response, "intent": detected_intent}

@app.post("/upload_audio", summary="Process Audio Input")
async def upload_audio(audio_file: UploadFile = File(...), session_id: Optional[str] = None):
    """Processes an audio file, transcribes it, and returns a conversational response."""
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
    
    try:
        processed_text = await multimodal_processor.process_audio(audio_file)
        detected_intent = intent_recognizer.recognize_intent(processed_text)
        response = dialogue_manager.manage_dialogue(processed_text, detected_intent, session_id=session_id or "default")
        return {"response": response, "intent": detected_intent, "transcribed_text": processed_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_image", summary="Process Image Input")
async def upload_image(image_file: UploadFile = File(...), text_context: Optional[str] = None, session_id: Optional[str] = None):
    """Processes an image file, generates a description, and potentially incorporates text context."""
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")
    
    image_description = await multimodal_processor.process_image(image_file)
    
    detected_intent = "general_query"
    user_input_for_dialogue = "User uploaded an image."
    
    if text_context:
        user_input_for_dialogue += f" Along with the text: \"{text_context}\""
        detected_intent = intent_recognizer.recognize_intent(text_context) # Use text for intent if provided

    response = dialogue_manager.manage_dialogue(user_input_for_dialogue, detected_intent, image_description=image_description, session_id=session_id or "default")
    
    return {"response": response, "intent": detected_intent, "image_description": image_description}


# --- Run the FastAPI application ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

