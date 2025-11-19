import gradio as gr
import requests
from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import uvicorn
import io

from transformers import pipeline, AutoProcessor, Blip2ForConditionalGeneration, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForSeq2Seq
import torch
from PIL import Image
import numpy as np

from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Initialize FastAPI app
app = FastAPI()

# --- Global Model and Component Initialization ---

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# STT Model (Whisper)
stt_model_id = "openai/whisper-small"
stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(stt_model_id).to(device)
stt_tokenizer = AutoTokenizer.from_pretrained(stt_model_id)
stt_processor = AutoProcessor.from_pretrained(stt_model_id)
whisper_pipeline = pipeline(
    "automatic-speech-recognition",
    model=stt_model,
    tokenizer=stt_tokenizer,
    processor=stt_processor,
    chunk_length_s=30,
    device=device,
)

# Image Analysis Model (BLIP-2)
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", torch_dtype=torch.float16 if device=="cuda" else torch.float32).to(device)

# NMT Model (for translation)
nmt_model_id = "Helsinki-NLP/opus-mt-en-fr" # Example: English to French
nmt_tokenizer_en_fr = AutoTokenizer.from_pretrained(nmt_model_id)
nmt_model_en_fr = AutoModelForSeq2Seq.from_pretrained(nmt_model_id).to(device)
nmt_pipeline_en_fr = pipeline("translation", model=nmt_model_en_fr, tokenizer=nmt_tokenizer_en_fr, device=device)

nmt_model_id_fr_en = "Helsinki-NLP/opus-mt-fr-en" # Example: French to English
nmt_tokenizer_fr_en = AutoTokenizer.from_pretrained(nmt_model_id_fr_en)
nmt_model_fr_en = AutoModelForSeq2Seq.from_pretrained(nmt_model_id_fr_en).to(device)
nmt_pipeline_fr_en = pipeline("translation", model=nmt_model_fr_en, tokenizer=nmt_tokenizer_fr_en, device=device)

# LLM (OpenAI as a placeholder)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# RAG - Embedding Model and Vector Store
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Simple in-memory ChromaDB for demonstration
# Populate with some dummy knowledge base entries
kb_documents = [
    Document(page_content="Our return policy allows returns within 30 days of purchase with a valid receipt."),
    Document(page_content="To reset your password, please visit our website and click 'Forgot Password'."),
    Document(page_content="We offer free shipping on all orders over $50."),
    Document(page_content="For technical support, you can call us at 1-800-TECH-HELP.")
]
vectorstore = Chroma.from_documents(documents=kb_documents, embedding=embeddings_model)
retriever = vectorstore.as_retriever()

# LangChain RAG Chain
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    ("user", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, rag_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- FastAPI Endpoints ---

@app.post("/chat")
async def chat_endpoint(
    text_input: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    language: str = Form("en")
):
    processed_text = []

    if audio_file:
        audio_bytes = await audio_file.read()
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        stt_result = whisper_pipeline(audio_np, generate_kwargs={"language": language})
        processed_text.append(f"[Audio Transcript]: {stt_result['text']}")

    if image_file:
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = blip_processor(image, return_tensors="pt").to(device, torch.float16 if device=="cuda" else torch.float32)
        out = blip_model.generate(**inputs, num_beams=1, max_new_tokens=20)
        image_caption = blip_processor.decode(out[0], skip_special_tokens=True)
        processed_text.append(f"[Image Description]: {image_caption}")

    if text_input:
        processed_text.append(text_input)
    
    combined_input = " ".join(processed_text)
    if not combined_input:
        return {"response": "Please provide some input (text, audio, or image)."}

    # Translate input to English if needed
    original_lang = language.lower()
    if original_lang != "en":
        if original_lang == "fr": # Example for French
            translated_input = nmt_pipeline_fr_en(combined_input)[0]["translation_text"]
        else:
            translated_input = combined_input # Fallback for unsupported languages
    else:
        translated_input = combined_input

    # LLM and RAG processing
    llm_response = rag_chain.invoke({"input": translated_input})
    agent_response = llm_response["answer"]

    # Translate response back to original language if needed
    if original_lang != "en":
        if original_lang == "fr": # Example for French
            final_response = nmt_pipeline_en_fr(agent_response)[0]["translation_text"]
        else:
            final_response = agent_response # Fallback
    else:
        final_response = agent_response

    return {"response": final_response}

# --- Gradio Interface ---

def get_backend_response(text, audio, image, language):
    files = {}
    data = {"language": language}
    if text:
        data["text_input"] = text
    if audio:
        files["audio_file"] = (audio.name, open(audio.name, "rb"), "audio/wav")
    if image:
        files["image_file"] = (image.name, open(image.name, "rb"), "image/jpeg")

    if not files and not text:
        return "Please provide some input (text, audio, or image)."

    try:
        response = requests.post("http://127.0.0.1:8000/chat", files=files, data=data)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with backend: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# Intelligent Multimodal Customer Support Agent")
    with gr.Row():
        text_input = gr.Textbox(label="Text Input", placeholder="Type your query here...")
    with gr.Row():
        audio_input = gr.Audio(label="Voice Input", type="filepath", sources=["microphone", "upload"])
        image_input = gr.Image(label="Image Input", type="filepath")
    with gr.Row():
        lang_dropdown = gr.Dropdown(choices=["en", "fr"], value="en", label="Input/Output Language")
    submit_btn = gr.Button("Submit")
    output_text = gr.Textbox(label="Agent Response", interactive=False)

    submit_btn.click(
        get_backend_response,
        inputs=[text_input, audio_input, image_input, lang_dropdown],
        outputs=output_text
    )


if __name__ == "__main__":
    import threading

    # Run FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "127.0.0.1", "port": 8000})
    fastapi_thread.daemon = True
    fastapi_thread.start()

    # Launch Gradio UI
    demo.launch(server_name="0.0.0.0", server_port=7860)
