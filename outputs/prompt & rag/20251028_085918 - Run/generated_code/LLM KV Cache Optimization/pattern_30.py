from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import requests
import asyncio
import time
import uvicorn

class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

llm_instance = None
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global llm_instance
    print(f"Loading LLM model: {MODEL_NAME}...")
    llm_instance = LLM(model=MODEL_NAME, trust_remote_code=True)
    print("LLM model loaded.")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: QueryRequest):
    if llm_instance is None:
        return ChatResponse(response="LLM not initialized yet. Please wait.")

    prompt = request.query
    outputs = llm_instance.generate(prompt, sampling_params)
    generated_text = ""
    for output in outputs:
        generated_text = output.outputs[0].text
        break

    return ChatResponse(response=generated_text)

async def simulate_client_requests():
    print("\n--- Starting Client Simulation ---")
    base_url = "http://localhost:8000/chat"

    queries = [
        "What is your return policy?",
        "What is your return policy for electronics?",
        "How can I track my order?",
        "How can I track my recent order?",
        "Do you offer free shipping?",
        "What is your refund process?"
    ]

    print(f"Targeting FastAPI server at {base_url}")
    print("Ensure the FastAPI server is running in a separate terminal or process.")
    print("The KV Cache Reuse benefit would be visible in vLLM's internal metrics/logs.")

    for i, query in enumerate(queries):
        start_time = time.time()
        try:
            response = requests.post(base_url, json={"query": query}, timeout=5)
            response.raise_for_status()
            chat_response = response.json()
            end_time = time.time()

            print(f"\nQuery {i+1} [{end_time - start_time:.4f}s]: '{query}'")
            print(f"Response: '{chat_response['response'].strip()}'")
        except requests.exceptions.ConnectionError:
            print(f"\nError: Could not connect to the server at {base_url}.")
            print("Please ensure the FastAPI server is running.")
            break
        except requests.exceptions.Timeout:
            print(f"\nError: Request timed out for query '{query}'. Is the server heavily loaded?")
            break
        except requests.exceptions.RequestException as e:
            print(f"\nAn error occurred for query '{query}': {e}")
            if response.status_code:
                print(f"Server responded with status code: {response.status_code}")
                print(f"Server error details: {response.text}")
            break
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    print("--- E-commerce Customer Support Chatbot with KV Cache Reuse ---")
    print("To run the FastAPI server:")
    print("1. Save this code as `chatbot_system.py`")
    print("2. Install dependencies: `pip install fastapi uvicorn pydantic vllm requests`")
    print("3. Ensure you have a GPU and sufficient VRAM for the LLM model.")
    print("4. Run the server: `uvicorn chatbot_system:app --host 0.0.0.0 --port 8000`")
    print("\nTo run the client simulation (after the server is started in a separate terminal):")
    print("1. In a new terminal, run: `python -c \"import asyncio, requests, time, sys; from chatbot_system import simulate_client_requests; asyncio.run(simulate_client_requests())\"`")
    print("   (Note: You need to ensure the `chatbot_system.py` file is accessible in the python path or run from its directory.)")
