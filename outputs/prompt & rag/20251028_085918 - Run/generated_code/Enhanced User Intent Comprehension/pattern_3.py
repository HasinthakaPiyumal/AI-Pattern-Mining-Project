
import gradio as gr
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
import speech_recognition as sr
import cv2
import numpy as np
from PIL import Image
import io
import re
import random

# --- 1. Global Setup (Dummy Models and Knowledge Base) ---

# Dummy LLM (simulating a transformer-based LLM)
def dummy_llm_predict(prompt):
    # Simulate a basic medical understanding and response
    if "symptoms" in prompt.lower() or "feel" in prompt.lower():
        response = "It sounds like you're experiencing some symptoms. While I can't provide a diagnosis, I recommend consulting a medical professional for an accurate assessment. Common symptoms might include fever, cough, or fatigue. Could you describe them in more detail?"
    elif "rash" in prompt.lower() or "skin" in prompt.lower():
        response = "A rash can have many causes. It's important to have it examined by a doctor. Common types include allergic reactions, infections, or skin conditions. Please avoid scratching it and consider a teleconsultation."
    elif "headache" in prompt.lower():
        response = "Headaches are common. They can be caused by stress, dehydration, or more serious conditions. If your headache is severe, persistent, or accompanied by other symptoms like blurred vision or numbness, please seek immediate medical attention."
    elif "medical history" in prompt.lower():
        response = "Understanding your medical history is crucial for accurate diagnosis. Please share this information directly with your healthcare provider. I can help answer general questions about conditions or medications if you provide specific details."
    elif "medication" in prompt.lower():
        response = "I can provide general information about medications, but always consult your doctor or pharmacist before taking or changing any medication. What medication are you curious about?"
    else:
        response = "Thank you for your query. I am an AI healthcare assistant, and while I can provide general information, I cannot offer medical diagnoses or prescriptions. Please consult a qualified healthcare professional for personalized medical advice."

    # Simulate adding resource guidance randomly
    if random.random() < 0.4: # 40% chance to add guidance
        response += " For reliable health information, you can visit organizations like the World Health Organization (WHO) or your local health authority websites."
    
    return response

# Dummy Medical Knowledge Base (for RAG simulation)
dummy_medical_knowledge = [
    "Common cold symptoms include runny nose, sore throat, cough, and congestion. It is usually caused by a virus.",
    "Influenza (flu) is a contagious respiratory illness caused by flu viruses. Symptoms are more severe than a cold and include fever, body aches, fatigue.",
    "Allergic reactions can cause skin rashes, hives, itching, and swelling. Identify and avoid triggers.",
    "Diabetes is a chronic condition that affects how your body turns food into energy. It requires careful management of blood sugar levels.",
    "High blood pressure (hypertension) often has no symptoms but can lead to serious health problems. Regular monitoring is important.",
    "Headaches can be tension headaches, migraines, or cluster headaches. Rest, pain relievers, and stress management can help.",
    "For severe symptoms like chest pain, sudden numbness, or difficulty breathing, seek emergency medical care immediately.",
    "Always consult a doctor for diagnosis and treatment of any medical condition."
]

# Dummy Embedding Model (simulating sentence-transformers)
def dummy_get_embedding(text):
    # A very simple hash-based 'embedding' for demonstration
    return [float(ord(c)) / 100 for c in text[:10]] if text else [0.0] * 10

# --- 2. Multimodal Integration & Preprocessing Layer ---

def preprocess_text(text):
    # Simulate basic NLP preprocessing (e.g., lowercasing, removing extra spaces)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation (simple)
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

# --- 3. Input Modality Layer & Multimodal Integration Functions ---

async def transcribe_audio_input(audio_file: UploadFile):
    r = sr.Recognizer()
    try:
        # SpeechRecognition needs a file-like object, but UploadFile is async
        # Need to read content and then use BytesIO
        content = await audio_file.read()
        audio_data = sr.AudioFile(io.BytesIO(content))
        with audio_data as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Speech recognition service error: {e}"
    except Exception as e:
        return f"Error processing audio: {e}"

def process_image_input(image_file_content: bytes):
    # Simulate image processing for symptom recognition
    try:
        # Convert bytes to numpy array then to OpenCV image
        nparr = np.frombuffer(image_file_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Placeholder for actual vision model (e.g., ViT or CNN)
        # For this example, we'll just 'detect' a rash if red pixels are dominant
        # This is a very simplistic and non-robust simulation
        if img is not None:
            height, width, _ = img.shape
            red_channel = img[:, :, 2] # Assuming BGR, Red is channel 2
            # Count pixels where red intensity is high
            red_pixel_count = np.sum(red_channel > 150) # Arbitrary threshold
            total_pixels = height * width

            if red_pixel_count / total_pixels > 0.10: # If more than 10% pixels are 'red'
                return "Image analysis suggests presence of redness/rash, further medical evaluation is recommended."
            else:
                return "Image processed. No obvious visual symptom detected by this simplified model."
        return "Failed to process image."
    except Exception as e:
        return f"Error processing image: {e}"

# --- 4. Intent Understanding & Context Management Layer (LLM Core) ---

def get_rag_context(query_embedding, knowledge_base):
    # Simulate retrieval from a vector database (Chroma/Faiss)
    # In a real system, this would involve comparing query_embedding with stored embeddings
    # For this dummy, we'll just check for keyword presence in our simple KB
    query_keywords = set(preprocess_text(query_embedding).split())
    relevant_docs = []
    for doc in knowledge_base:
        doc_keywords = set(preprocess_text(doc).split())
        if query_keywords.intersection(doc_keywords):
            relevant_docs.append(doc)
    
    # Return a concatenated string of relevant documents or a default message
    return "\n".join(relevant_docs) if relevant_docs else "No highly specific medical knowledge found for this query directly in our internal database, but general advice can be given."

def run_llm_chain(text_input, image_features=None, audio_text=None, conversation_history=None):
    # Construct a comprehensive prompt for the dummy LLM
    full_query = text_input
    if audio_text and audio_text != "Could not understand audio":
        full_query = f"User said: '{audio_text}'. Additionally, their text input is: '{text_input}'"
    if image_features and "rash" in image_features.lower():
        full_query += f" Based on an image provided, it was noted: {image_features}"
    
    # Preprocess the combined query
    processed_query = preprocess_text(full_query)
    
    # Simulate RAG by getting context based on the query
    query_embedding_text = f"{processed_query} {image_features if image_features else ''} {audio_text if audio_text else ''}"
    # In a real system, query_embedding would be from a real embedding model
    # For simulation, we'll pass the text itself to the dummy RAG for keyword matching
    rag_context = get_rag_context(query_embedding_text, dummy_medical_knowledge)

    # Build the prompt for the LLM, incorporating history and RAG context
    prompt_parts = []
    if conversation_history:
        prompt_parts.append(f"Conversation history: {conversation_history}")
    
    prompt_parts.append(f"Medical knowledge context: {rag_context}")
    prompt_parts.append(f"User query: {full_query}")
    prompt_parts.append("As an AI healthcare assistant, provide general health information and guidance based on the query and context. Do not diagnose or prescribe.")
    
    final_prompt = "\n\n".join(prompt_parts)

    # Get response from dummy LLM
    llm_response = dummy_llm_predict(final_prompt)
    
    return llm_response

# --- 5. FastAPI Backend ---

app = FastAPI()

# In-memory store for conversation history (for a simple demo)
conversation_history = ""

@app.post("/assist")
async def assist_endpoint(
    text_input: str = Form("", description="Text query from the user"),
    audio_file: UploadFile = File(None, description="Optional audio input"),
    image_file: UploadFile = File(None, description="Optional image input (e.g., symptom photo)")
):
    global conversation_history
    
    audio_text = None
    if audio_file:
        audio_text = await transcribe_audio_input(audio_file)
        if "Error" in audio_text or "Could not understand" in audio_text:
            print(f"Audio processing error: {audio_text}")
            # If audio fails, proceed with just text input
            audio_text = None

    image_features = None
    if image_file:
        image_content = await image_file.read()
        image_features = process_image_input(image_content)
        if "Error" in image_features:
            print(f"Image processing error: {image_features}")
            image_features = None

    # If text input is empty but audio was transcribed, use audio as primary text_input
    if not text_input and audio_text:
        text_input = audio_text

    if not text_input and not audio_text and not image_features:
        return {"response": "Please provide a text query, audio, or an image to get assistance."}
    
    # Run the LLM chain with all available modalities and history
    llm_response = run_llm_chain(
        text_input=text_input,
        image_features=image_features,
        audio_text=audio_text,
        conversation_history=conversation_history
    )

    # Update conversation history (simple append for demo)
    user_utterance = []
    if text_input: user_utterance.append(f"Text: {text_input}")
    if audio_text: user_utterance.append(f"Audio: {audio_text}")
    if image_features: user_utterance.append(f"Image: {image_features}")
    
    # Limit history length for a simple demo
    max_history_length = 500
    conversation_history = (conversation_history + "\nUser: " + "; ".join(user_utterance) + "\nAssistant: " + llm_response)[-max_history_length:]

    return {"response": llm_response}

# --- 6. Gradio User Interface ---

def gradio_interface(text_input, audio_input, image_input):
    # This function will call the FastAPI endpoint
    # For a self-contained Gradio app without a separate FastAPI server, 
    # we can directly call the logic, which simplifies deployment for this example.
    
    global conversation_history
    current_conversation_history = conversation_history # Capture current state

    audio_text_result = None
    if audio_input:
        # Gradio audio input gives (sample_rate, numpy_array)
        # Need to save it to a temporary WAV file for SpeechRecognition
        try:
            import soundfile as sf
            temp_audio_path = "temp_audio.wav"
            sf.write(temp_audio_path, audio_input[1], audio_input[0])
            
            r = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                audio = r.record(source)
            audio_text_result = r.recognize_google(audio)
        except sr.UnknownValueError:
            audio_text_result = "Could not understand audio"
        except sr.RequestError as e:
            audio_text_result = f"Speech recognition service error: {e}"
        except Exception as e:
            audio_text_result = f"Error processing audio: {e}"
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_audio_path): os.remove(temp_audio_path)


    image_features_result = None
    if image_input is not None:
        # Gradio image input is a numpy array (PIL Image converted to numpy)
        # Convert numpy array to bytes for our process_image_input
        try:
            is_success, im_buf_arr = cv2.imencode(".png", image_input)
            if is_success:
                image_bytes = im_buf_arr.tobytes()
                image_features_result = process_image_input(image_bytes)
            else:
                image_features_result = "Failed to encode image for processing."
        except Exception as e:
            image_features_result = f"Error processing image: {e}"

    # If text input is empty but audio was transcribed, use audio as primary text_input for LLM
    final_text_input = text_input
    if not final_text_input and audio_text_result and "Could not understand" not in audio_text_result and "Error" not in audio_text_result:
        final_text_input = audio_text_result

    if not final_text_input and not audio_text_result and not image_features_result:
        return "Please provide a text query, audio, or an image to get assistance.", current_conversation_history

    response_from_llm = run_llm_chain(
        text_input=final_text_input,
        image_features=image_features_result,
        audio_text=audio_text_result,
        conversation_history=current_conversation_history
    )

    # Update conversation history (simple append for demo)
    user_utterance_parts = []
    if final_text_input: user_utterance_parts.append(f"Text: {final_text_input}")
    if audio_text_result: user_utterance_parts.append(f"Audio Transcribed: {audio_text_result}")
    if image_features_result: user_utterance_parts.append(f"Image Analysis: {image_features_result}")
    
    user_log = "; ".join(user_utterance_parts)

    # Limit history length for a simple demo
    max_history_length = 1000 # Increased for better demo experience
    updated_history = f"User: {user_log}\nAssistant: {response_from_llm}\n"
    conversation_history = (current_conversation_history + updated_history)[-max_history_length:]

    return response_from_llm, conversation_history

# Define the Gradio interface directly for simplicity
# Note: Running Gradio directly in the same file as FastAPI will make FastAPI not directly accessible
# For a true API + UI separation, FastAPI would run on one port and Gradio would be a client.
# For this single-file generation, we'll demonstrate a Gradio-only execution of the logic.

if __name__ == "__main__":
    # To run this script:
    # 1. pip install fastapi uvicorn python-multipart speechrecognition opencv-python numpy Pillow gradio soundfile
    # 2. python healthcare_assistant.py
    
    print("Starting Healthcare Assistant Gradio UI...")
    print("Please ensure you have installed all required libraries: fastapi uvicorn python-multipart speechrecognition opencv-python numpy Pillow gradio soundfile")
    print("You may also need portaudio for SpeechRecognition if you encounter issues with audio input.")

    demo = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.Textbox(label="Text Query", placeholder="Describe your symptoms or ask a health question..."),
            gr.Audio(type="numpy", label="Speak your query"),
            gr.Image(type="numpy", label="Upload an image (e.g., of a rash)")
        ],
        outputs=[
            gr.Textbox(label="Assistant's Response"),
            gr.Textbox(label="Conversation History", interactive=False)
        ],
        title="AI Healthcare Assistant (Multimodal Demo)",
        description="This assistant leverages text, voice, and image input to understand your health-related queries and provide general information. It does not diagnose or prescribe."
    )
    demo.launch()

    # To run FastAPI separately (requires modifying the __main__ block and the gradio_interface function):
    # if __name__ == "__main__":
    #     uvicorn.run(app, host="0.0.0.0", port=8000)

    # If running Gradio as a client to FastAPI, you would uncomment the uvicorn.run and modify gradio_interface
    # to make HTTP requests to the FastAPI endpoint. For this single-file output, direct execution in Gradio is simpler.
