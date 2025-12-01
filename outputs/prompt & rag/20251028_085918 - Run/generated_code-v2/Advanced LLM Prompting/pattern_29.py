"""
This file contains the combined code for a Zero-Shot Customer Support Chatbot.
It includes the prompt template, FastAPI backend, and Gradio frontend.

To run this application:
1. Save the content below as `zero_shot_chatbot_app.py`.
2. Install the necessary libraries: `pip install fastapi uvicorn transformers gradio requests`
3. Run the FastAPI backend: `uvicorn zero_shot_chatbot_app:app --port 8000` (in one terminal)
4. Run the Gradio frontend: `python zero_shot_chatbot_app.py` (in another terminal, after FastAPI is running)
"""

# --- FILE: prompts.py ---START
ZERO_SHOT_PROMPT_TEMPLATE = """You are an e-commerce customer support agent. Your goal is to assist customers by providing clear and concise information based *solely* on their query and your general knowledge. Do not ask clarifying questions unless absolutely necessary.

Available information types:
- Order Status (e.g., "Where is my order?")
- Product Information (e.g., "Tell me about product X")
- Return Policy (e.g., "How do I return an item?")
- Shipping Information (e.g., "What are your shipping options?")

Customer Query: {query}
Customer Support Agent: """
# --- FILE: prompts.py ---END


# --- FILE: app.py ---START
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import pipeline
import uvicorn
import threading # Used to run uvicorn in a separate thread if desired for integrated execution

app = FastAPI()

# Load a pre-trained model for text generation (distilgpt2 for demonstration)
# This simulates the LLM. For production, a more capable model would be used.
print("Loading text generation pipeline (distilgpt2)... This may take a moment.")
generator = pipeline("text-generation", model="distilgpt2", trust_remote_code=True)
print("Text generation pipeline loaded.")

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_query = data.get("query", "")

    if not user_query:
        return JSONResponse({"response": "Please provide a query."})

    # Construct the zero-shot prompt
    full_prompt = ZERO_SHOT_PROMPT_TEMPLATE.format(query=user_query)

    # Generate a response using the LLM
    # We take the first generated sequence and remove the prompt itself from the output
    try:
        response = generator(full_prompt, max_new_tokens=100, num_return_sequences=1, truncation=True)
        generated_text = response[0]['generated_text']
        chatbot_response = generated_text[len(full_prompt):].strip()
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        chatbot_response = "I apologize, but I encountered an error while processing your request. Please try again."

    return JSONResponse({"response": chatbot_response})

# --- FILE: app.py ---END


# --- FILE: chatbot_ui.py ---START
import gradio as gr
import requests
import time

# Function to run FastAPI app in a separate thread
def run_fastapi():
    # Use a custom entry point for uvicorn to avoid conflicts if `__name__ == "__main__"` is also in the main script
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

def chat_interface(user_message):
    # Give FastAPI a moment to start if running in the same script
    time.sleep(1) 
    try:
        response = requests.post("http://127.0.0.1:8000/chat", json={"query": user_message})
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        return data.get("response", "Error: No response from chatbot.")
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the backend. Please ensure the FastAPI server is running at http://127.0.0.1:8000."
    except requests.exceptions.RequestException as e:
        return f"Error: An unexpected error occurred: {e}"

# Create the Gradio interface
iface = gr.Interface(
    fn=chat_interface,
    inputs=gr.Textbox(lines=2, placeholder="Type your query here..."),
    outputs="text",
    title="Zero-Shot Customer Support Chatbot",
    description="Ask any e-commerce related question without specific examples. The bot will try to answer based on its general knowledge."
)

# Main execution block
if __name__ == "__main__":
    print("Starting FastAPI server in a separate thread...")
    # Run FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True  # Allows the thread to exit when the main program exits
    fastapi_thread.start()

    print("Launching Gradio interface...")
    iface.launch(inbrowser=True)
    print("Gradio interface closed. FastAPI server should also terminate.")
# --- FILE: chatbot_ui.py ---END