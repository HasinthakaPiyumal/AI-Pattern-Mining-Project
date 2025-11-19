import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost:8001/v1/completions")

@app.post("/chat")
async def chat_with_llm(message: dict):
    user_message = message.get("message", "")
    if not user_message:
        return {"response": "Please provide a message."}

    # Construct a simple prompt for the customer support chatbot
    prompt = f"You are an intelligent customer support assistant for an e-commerce platform. Respond to the customer's inquiry concisely and helpfully.\n\nCustomer: {user_message}\nAssistant:"

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "facebook/opt-125m", # Replace with your actual vLLM deployed model name
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7,
        "stop": ["\nCustomer:", "\nAssistant:"]
    }

    try:
        response = requests.post(VLLM_SERVER_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        vllm_response = response.json()
        generated_text = vllm_response["choices"][0]["text"]
        return {"response": generated_text.strip()}
    except requests.exceptions.RequestException as e:
        return {"response": f"Error communicating with LLM server: {e}"}


# Gradio Frontend Integration
def gradio_chat_interface(message, history):
    global VLLM_SERVER_URL # Ensure VLLM_SERVER_URL is accessible
    fastapi_url = "http://localhost:8000/chat"
    payload = {"message": message}
    try:
        response = requests.post(fastapi_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        return f"Error contacting backend: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# E-commerce Customer Support Chatbot")
    gr.Markdown("Ask me anything about your orders, products, or returns!")
    chatbot = gr.ChatInterface(
        fn=gradio_chat_interface,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Ask me a question...", container=False, scale=7),
        title="Customer Support Bot",
        description="An AI-powered chatbot to assist with e-commerce inquiries. Uses vLLM for optimized inference.",
        theme="soft",
        examples=["Where is my order?", "How do I return an item?", "Tell me about product X"],
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear Chat",
    )

# To run FastAPI: uvicorn chatbot_app:app --reload --port 8000
# To run Gradio: python -m gradio chatbot_app.py
# Note: Gradio will run its own server. For this combined file, you'd typically run Gradio, and Gradio would call the FastAPI endpoints.
# However, for a truly integrated run with a single script, you might embed Gradio within FastAPI or run them separately.
# For simplicity and adherence to the single-file request, the Gradio interface is defined here.
# A more robust setup would have `app.py` for FastAPI and `chatbot_ui.py` for Gradio, with Gradio calling the deployed FastAPI.
# To run this file as a single entity for demonstration:
# You would typically run the FastAPI app separately and the Gradio app separately.
# However, for this combined output, you can run `python chatbot_app.py` after `pip install uvicorn fastapi python-dotenv requests gradio`
# and ensure `VLLM_SERVER_URL` is set in a `.env` file.
# The `if __name__ == "__main__"` block is commented out because Gradio typically has its own `launch()` method.
# To run FastAPI: `uvicorn chatbot_app:app --host 0.0.0.0 --port 8000`
# To run Gradio (after FastAPI is running): `python chatbot_app.py` (and it will use the FastAPI backend)

# Conceptual vLLM Server Setup (run this in a separate terminal before running the Python script):
# export VLLM_SERVER_URL="http://localhost:8001/v1/completions"
# python -m vllm.entrypoints.api_server --model facebook/opt-125m --port 8001
# (Replace 'facebook/opt-125m' with your desired LLM)


if __name__ == "__main__":
    # This block allows running both FastAPI and Gradio from a single script.
    # In a production setup, they would likely be deployed separately.
    import uvicorn
    import threading

    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()

    # Launch Gradio interface
    demo.launch(server_name="0.0.0.0", server_port=7860)
