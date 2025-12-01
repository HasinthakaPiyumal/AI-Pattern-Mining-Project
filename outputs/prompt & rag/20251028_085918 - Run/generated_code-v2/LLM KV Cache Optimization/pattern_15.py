from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
import requests


# 1. FastAPI Application (Chatbot API)
app = FastAPI()

# Initialize vLLM LLM globally to load the model once
# For demonstration, using a small model. In production, choose a suitable enterprise-grade LLM.
# Ensure your environment has the necessary GPU and vLLM dependencies installed.
llm = LLM(model="facebook/opt-125m")

# Define sampling parameters for text generation
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Generate text using vLLM
        # vLLM's LLM.generate handles batching and PagedAttention internally
        outputs = llm.generate([request.query], sampling_params)
        
        generated_text = outputs[0].outputs[0].text
        return ChatResponse(response=generated_text)
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")


# 3. Client Application (Conceptual - for testing the FastAPI endpoint)
def run_client_example(query: str):
    url = "http://127.0.0.1:8000/chat"
    headers = {"Content-Type": "application/json"}
    data = {"query": query}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print("Client Response:", response.json())
    except requests.exceptions.ConnectionError:
        print("Client Error: Could not connect to the FastAPI server. Please ensure the server is running at http://127.0.0.1:8000")
    except requests.exceptions.RequestException as e:
        print(f"Client Error: {e}")


if __name__ == "__main__":
    # To run the FastAPI server:
    # Save this file as, e.g., 'chatbot_platform.py'
    # Run 'uvicorn chatbot_platform:app --host 0.0.0.0 --port 8000'
    # The client can then be run in a separate terminal or after the server is started.
    print("To start the FastAPI server, run the following command in your terminal:")
    print("uvicorn chatbot_platform:app --host 0.0.0.0 --port 8000")
    print("\nOnce the server is running, you can test it with the 'run_client_example' function manually or via a separate script.")
    print("Example client call: run_client_example(\"What is the capital of France?\")")

    # Example of how to manually run the client after the server is up
    # import time
    # time.sleep(5) # Give server time to start if run sequentially (not recommended for actual use)
    # run_client_example("Hello, how can I help you today?")
