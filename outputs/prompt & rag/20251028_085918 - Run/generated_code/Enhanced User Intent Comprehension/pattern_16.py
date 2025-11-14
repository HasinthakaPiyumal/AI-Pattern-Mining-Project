import os
import json
from dotenv import load_dotenv
from loguru import logger
import gradio as gr
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForSeq2SeqLM, AutoModel
import torch
import cv2
import numpy as np
from PIL import Image
import soundfile as sf
import httpx

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory

from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import threading

# --- Configuration and Logging ---
load_dotenv()
logger.add("file.log", rotation="500 MB")

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# --- Model Loading ---

# ASR Model
ASR_MODEL_ID = "openai/whisper-small"
logger.info(f"Loading ASR model: {ASR_MODEL_ID} on {DEVICE}")
asr_pipeline = None
try:
    asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(ASR_MODEL_ID).to(DEVICE)
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=asr_processor.tokenizer,
        feature_extractor=asr_processor.feature_extractor,
        chunk_length_s=30,
        device=0 if DEVICE == "cuda" else -1,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    logger.info("ASR model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading ASR model: {e}")

# Image Captioning Model (e.g., BLIP)
IMAGE_CAPTIONING_MODEL_ID = "Salesforce/blip-image-captioning-base"
logger.info(f"Loading Image Captioning model: {IMAGE_CAPTIONING_MODEL_ID} on {DEVICE}")
image_captioner = None
try:
    image_captioner = pipeline("image-to-text", model=IMAGE_CAPTIONING_MODEL_ID, device=0 if DEVICE == "cuda" else -1)
    logger.info("Image Captioning model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Image Captioning model: {e}")

# Machine Translation Model (e.g., NLLB)
TRANSLATION_MODEL_ID = "facebook/nllb-200-distilled-600M"
logger.info(f"Loading Translation model: {TRANSLATION_MODEL_ID} on {DEVICE}")
translator = None
try:
    translator = pipeline("translation", model=TRANSLATION_MODEL_ID, src_lang="auto", tgt_lang="eng", device=0 if DEVICE == "cuda" else -1)
    logger.info("Translation model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Translation model: {e}")

# LLM for conversation
LLM_MODEL_ID = "distilbert/distilgpt2"  # A small, fast model for demonstration
logger.info(f"Loading LLM: {LLM_MODEL_ID} on {DEVICE}")
llm = None
try:
    llm_pipeline = pipeline(
        "text-generation",
        model=LLM_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device=0 if DEVICE == "cuda" else -1,
        max_new_tokens=250
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    logger.info("LLM loaded successfully.")
except Exception as e:
    logger.error(f"Error loading LLM: {e}")

# Embeddings for ChromaDB
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
logger.info(f"Loading Embedding model: {EMBEDDING_MODEL_ID}")
embeddings = None
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Embedding model: {e}")

# ChromaDB
VECTOR_DB_DIR = "./chroma_db"
vectordb = None
try:
    if embeddings:
        if os.path.exists(VECTOR_DB_DIR):
            vectordb = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
            logger.info(f"Loaded existing ChromaDB from {VECTOR_DB_DIR}")
        else:
            vectordb = Chroma.from_texts(["Initial knowledge base entry about customer support.", "Products offered: product A, product B, product C.", "Common issue: troubleshooting guide for product A."], embeddings, persist_directory=VECTOR_DB_DIR)
            vectordb.persist()
            logger.info(f"Initialized new ChromaDB at {VECTOR_DB_DIR}")
    else:
        logger.error("Embeddings model not loaded, cannot initialize ChromaDB.")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")

# LangChain Conversational Chain
qa_chain = None
if llm and vectordb:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    logger.info("LangChain QA chain initialized.")
else:
    logger.warning("LangChain QA chain not initialized due to missing LLM or Vectordb.")

# --- Helper Functions ---
def transcribe_speech(audio_path):
    if not asr_pipeline:
        return "ASR service is not available."
    try:
        # ASR pipeline expects a path or raw audio array
        result = asr_pipeline(audio_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Error transcribing speech from {audio_path}: {e}")
        return f"Failed to transcribe speech: {e}"

def analyze_image(image_path):
    if not image_captioner:
        return "Image analysis service is not available."
    try:
        # Image captioner pipeline expects a PIL Image or path
        image_pil = Image.open(image_path)
        result = image_captioner(image_pil)
        return result[0]["generated_text"]
    except Exception as e:
        logger.error(f"Error analyzing image from {image_path}: {e}")
        return f"Failed to analyze image: {e}"

def translate_text(text, target_lang="eng"):
    if not translator:
        return text  # Return original if translator not available
    try:
        # The pipeline automatically detects source language and translates to tgt_lang
        translated_text = translator(text, tgt_lang=target_lang)[0]["translation_text"]
        return translated_text
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        return text  # Return original on failure


# --- FastAPI Application ---
app = FastAPI()

@app.post("/chat")
async def chat_endpoint(
    text_input: str = Form(None),
    audio_file: UploadFile = File(None),
    image_file: UploadFile = File(None),
    chat_history: str = Form("[]") # chat_history will be a JSON string of list of lists
):
    processed_user_query = ""
    user_combined_input_summary = ""
    
    current_chat_history = json.loads(chat_history) # Deserialize the history

    # Process Audio
    if audio_file:
        audio_path = f"temp_{audio_file.filename}"
        try:
            with open(audio_path, "wb") as buffer:
                buffer.write(await audio_file.read())
            transcribed_text = transcribe_speech(audio_path)
            if "Failed to transcribe" not in transcribed_text:
                user_combined_input_summary += f"🎤 (Audio): {transcribed_text}. "
                processed_user_query += f"Audio input: {transcribed_text}. "
            else:
                user_combined_input_summary += f"🎤 (Audio): Error. "
                logger.warning(transcribed_text)
        finally:
            if os.path.exists(audio_path): os.remove(audio_path) # Clean up

    # Process Image
    if image_file:
        image_path = f"temp_{image_file.filename}"
        try:
            with open(image_path, "wb") as buffer:
                buffer.write(await image_file.read())
            image_description = analyze_image(image_path)
            if "Failed to analyze" not in image_description:
                user_combined_input_summary += f"🖼️ (Image): {image_description}. "
                processed_user_query += f"Image input: {image_description}. "
            else:
                user_combined_input_summary += f"🖼️ (Image): Error. "
                logger.warning(image_description)
        finally:
            if os.path.exists(image_path): os.remove(image_path) # Clean up

    # Process Text Input
    if text_input:
        user_combined_input_summary += f"📝 (Text): {text_input}. "
        processed_user_query += f"Text input: {text_input}. "
    
    processed_user_query = processed_user_query.strip()
    user_combined_input_summary = user_combined_input_summary.strip()

    if not processed_user_query:
        return {"response": "Please provide some input (text, audio, or image)."}

    llm_input_translated_to_eng = translate_text(processed_user_query, target_lang="eng")
    logger.info(f"FastAPI: Input for LLM (after internal translation to English): {llm_input_translated_to_eng}")

    if qa_chain:
        try:
            # Clear and re-populate LangChain's memory for each request based on `current_chat_history`
            # This makes the FastAPI endpoint stateless with respect to LangChain's memory,
            # relying on Gradio to send the full history.
            qa_chain.memory.clear()
            for human_msg, ai_msg in current_chat_history:
                # Only add if both messages are non-empty to avoid issues with partial history entries
                if human_msg and ai_msg:
                    qa_chain.memory.save_context({"input": human_msg}, {"output": ai_msg})
            
            result = qa_chain({"question": llm_input_translated_to_eng})
            bot_response = result["answer"]
            logger.info(f"FastAPI: LLM Raw Response: {bot_response}")
            
            final_bot_response = bot_response # Assuming English output based on internal processing
            
            return {"user_summary": user_combined_input_summary, "response": final_bot_response}

        except Exception as e:
            logger.error(f"FastAPI: Error during LLM interaction: {e}")
            return {"response": f"An internal error occurred: {e}"}
    else:
        return {"response": "LLM or RAG service is not available. Please check server logs."}


# --- Gradio Interface ---

# Gradio interface function that talks to FastAPI
def gradio_chat_interface(message, history, audio_input, image_input):
    # Prepare data for FastAPI
    files = {}
    data = {"text_input": message if message else "", "chat_history": json.dumps(history)}

    temp_files_to_clean = []

    if audio_input is not None:
        # Gradio audio input is a tuple (samplerate, numpy array)
        # Need to save it to a temp file for FastAPI UploadFile
        samplerate, audio_array = audio_input
        temp_audio_file = "temp_audio_gradio.wav"
        sf.write(temp_audio_file, audio_array, samplerate) # Ensure soundfile is installed: pip install soundfile
        files["audio_file"] = ("audio.wav", open(temp_audio_file, "rb"), "audio/wav")
        temp_files_to_clean.append(temp_audio_file)
        logger.info(f"Gradio: Prepared audio file for FastAPI: {temp_audio_file}")

    if image_input is not None:
        # Gradio image input is a PIL Image object
        temp_image_file = "temp_image_gradio.png"
        image_input.save(temp_image_file)
        files["image_file"] = ("image.png", open(temp_image_file, "rb"), "image/png")
        temp_files_to_clean.append(temp_image_file)
        logger.info(f"Gradio: Prepared image file for FastAPI: {temp_image_file}")

    user_summary_for_display = "User input: " # Fallback summary
    if message: user_summary_for_display += f"📝 {message} "
    if audio_input is not None: user_summary_for_display += f"🎤 (Audio detected) "
    if image_input is not None: user_summary_for_display += f"🖼️ (Image detected) "
    user_summary_for_display = user_summary_for_display.strip()

    try:
        # Send request to FastAPI
        response = httpx.post("http://localhost:8000/chat", data=data, files=files, timeout=60.0)
        response.raise_for_status() # Raise an exception for HTTP errors
        
        result = response.json()
        user_summary = result.get("user_summary", user_summary_for_display)
        bot_response = result.get("response", "No response from assistant.")
        
        history.append((user_summary, bot_response))
        
        return history, None
    
    except httpx.RequestError as e:
        logger.error(f"Gradio: HTTP Request failed: {e}")
        history.append((user_summary_for_display, f"Connection error: Could not reach backend. Please ensure FastAPI is running on http://localhost:8000. Error: {e}"))
        return history, None
    except httpx.HTTPStatusError as e:
        logger.error(f"Gradio: HTTP Status error: {e.response.status_code} - {e.response.text}")
        history.append((user_summary_for_display, f"Backend error: {e.response.status_code} - {e.response.text}"))
        return history, None
    except Exception as e:
        logger.error(f"Gradio: An unexpected error occurred: {e}")
        history.append((user_summary_for_display, f"An unexpected error occurred: {e}"))
        return history, None
    finally:
        # Clean up temp files
        for filename in temp_files_to_clean:
            if os.path.exists(filename):
                os.remove(filename)
                logger.info(f"Cleaned up temporary file: {filename}")
        # Close file handles
        for _, (_, file_obj, _) in files.items():
            file_obj.close()


with gr.Blocks() as demo:
    gr.Markdown("# Intelligent Multi-Modal Customer Support Assistant")
    gr.Markdown("Interact with the AI assistant using text, speech, or images.")

    chatbot = gr.ChatInterface(
        gradio_chat_interface,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Type your message here...", container=False, scale=7),
        clear_btn=None, # Will redefine below for custom clear logic
        submit_btn=None, # Will redefine below for custom submit logic
        additional_inputs=[
            gr.Audio(type="numpy", label="Speech Input"),
            gr.Image(type="pil", label="Image Input"),
        ],
        additional_inputs_accordion=gr.Accordion(label="Multi-modal Inputs", open=True),
        # Using custom submit button to clear additional inputs
        # ChatInterface automatically provides a submit button if submit_btn=None.
        # We need to manually add buttons to control behavior.
    )

    # Custom Clear and Submit buttons
    with gr.Row():
        text_submit_button = gr.Button("Send Text", variant="primary")
        audio_submit_button = gr.Button("Send Audio")
        image_submit_button = gr.Button("Send Image")
        clear_button = gr.Button("Clear Chat & Inputs", variant="secondary")

    # Trigger chat_interface with specific inputs
    text_submit_button.click(
        gradio_chat_interface,
        inputs=[chatbot.textbox, chatbot.chatbot, chatbot.additional_inputs[0], chatbot.additional_inputs[1]],
        outputs=[chatbot.chatbot, chatbot.textbox, chatbot.additional_inputs[0], chatbot.additional_inputs[1]],
        queue=False
    )

    audio_submit_button.click(
        gradio_chat_interface,
        inputs=[gr.State(""), chatbot.chatbot, chatbot.additional_inputs[0], gr.State(None)], # Pass empty string for text, None for image
        outputs=[chatbot.chatbot, chatbot.textbox, chatbot.additional_inputs[0], chatbot.additional_inputs[1]],
        queue=False
    )

    image_submit_button.click(
        gradio_chat_interface,
        inputs=[gr.State(""), chatbot.chatbot, gr.State(None), chatbot.additional_inputs[1]], # Pass empty string for text, None for audio
        outputs=[chatbot.chatbot, chatbot.textbox, chatbot.additional_inputs[0], chatbot.additional_inputs[1]],
        queue=False
    )

    # Clear inputs on chat clear
    clear_button.click(
        lambda: ([], "", None, None), # Clear history, textbox, audio, image
        inputs=[],
        outputs=[chatbot.chatbot, chatbot.textbox, chatbot.additional_inputs[0], chatbot.additional_inputs[1]],
        queue=False
    )


# Function to run Gradio and FastAPI
def run_servers():
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=uvicorn.run, args=(app,), kwargs={
        "host": os.getenv("FASTAPI_HOST", "0.0.0.0"),
        "port": int(os.getenv("FASTAPI_PORT", 8000)),
        "log_level": "info"
    })
    fastapi_thread.daemon = True # Allow main program to exit even if thread is running
    fastapi_thread.start()
    logger.info(f"FastAPI server starting on http://{os.getenv('FASTAPI_HOST', '0.0.0.0')}:{os.getenv('FASTAPI_PORT', 8000)}")

    # Start Gradio
    logger.info("Gradio UI starting...")
    demo.launch(
        server_name=os.getenv("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        share=False # Set to True to get a public link
    )

if __name__ == "__main__":
    run_servers()
