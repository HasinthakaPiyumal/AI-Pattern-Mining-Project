from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio
from typing import List, Dict, Any


class MockCompletionOutput:
    def __init__(self, text: str):
        self.text = text


class MockRequestOutput:
    def __init__(self, prompt: str, outputs: List[MockCompletionOutput]):
        self.prompt = prompt
        self.outputs = outputs


class MockVLLMClient:
    def __init__(self, model: str):
        self.model = model

    async def generate(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[MockRequestOutput]:
        await asyncio.sleep(0.1)  # Simulate async LLM inference delay
        results = []
        for prompt in prompts:
            # Simulate LLM response based on a simple rule or fixed text
            if "hello" in prompt.lower():
                response = "Hello! How can I assist you today?"
            elif "product" in prompt.lower() and "issue" in prompt.lower():
                response = "Could you please describe the product issue in more detail?"
            elif "order status" in prompt.lower():
                response = "Please provide your order number so I can check its status."
            else:
                response = "I'm an AI co-pilot. How can I help with this customer query?"
            
            results.append(MockRequestOutput(prompt=prompt, outputs=[MockCompletionOutput(text=response)]))
        return results


class ChatRequest(BaseModel):
    session_id: str
    message: str
    conversation_history: List[Dict[str, str]] = []


class ChatResponse(BaseModel):
    session_id: str
    response: str
    full_conversation: List[Dict[str, str]]


app = FastAPI()

# Initialize a mock vLLM client
mock_vllm_client = MockVLLMClient(model="mock-llama-7b")

# This would typically be a global or per-session store for conversation history
# For simplicity, we'll use a dictionary here.
conversation_store: Dict[str, List[Dict[str, str]]] = {}


@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    current_conversation = conversation_store.get(request.session_id, [])
    current_conversation.append({"role": "user", "content": request.message})

    # Prepare prompt for LLM (simplified)
    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in current_conversation]) + "\nagent:"

    # Simulate vLLM inference
    sampling_params = {"temperature": 0.7, "max_tokens": 100}
    vllm_response = await mock_vllm_client.generate(prompts=[prompt], sampling_params=sampling_params)

    llm_response_text = vllm_response[0].outputs[0].text.strip()

    current_conversation.append({"role": "agent", "content": llm_response_text})
    conversation_store[request.session_id] = current_conversation

    return ChatResponse(
        session_id=request.session_id,
        response=llm_response_text,
        full_conversation=current_conversation,
    )


def simulate_frontend_interaction():
    print("\n--- Simulating Frontend Interaction ---")
    session_id = "agent_session_123"

    async def send_request(message: str, history: List[Dict[str, str]]):
        async with httpx.AsyncClient() as client:
            payload = {
                "session_id": session_id,
                "message": message,
                "conversation_history": history
            }
            resp = await client.post("http://127.0.0.1:8000/chat", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def main_frontend_flow():
        current_history = []

        # First interaction
        user_message_1 = "Hello, I have a question about my order."
        print(f"Agent sends: {user_message_1}")
        response_1 = await send_request(user_message_1, current_history)
        current_history = response_1["full_conversation"]
        print(f"Co-pilot suggests: {response_1['response']}")
        print(f"Current conversation: {current_history}\n")

        # Second interaction
        user_message_2 = "My order number is XYZ123 and it hasn't shipped yet."
        print(f"Agent sends: {user_message_2}")
        response_2 = await send_request(user_message_2, current_history)
        current_history = response_2["full_conversation"]
        print(f"Co-pilot suggests: {response_2['response']}")
        print(f"Current conversation: {current_history}\n")

        # Third interaction
        user_message_3 = "I am having an issue with a product, model ABC."
        print(f"Agent sends: {user_message_3}")
        response_3 = await send_request(user_message_3, current_history)
        current_history = response_3["full_conversation"]
        print(f"Co-pilot suggests: {response_3['response']}")
        print(f"Current conversation: {current_history}\n")

    try:
        import httpx
        asyncio.run(main_frontend_flow())
    except ImportError:
        print("Install 'httpx' (pip install httpx) to run the frontend simulation.")
    except Exception as e:
        print(f"Error during frontend simulation: {e}")


if __name__ == "__main__":
    print("Starting FastAPI server with mock vLLM client...")
    print("Run 'uvicorn main:app --reload' in a terminal and then execute this script for frontend simulation.")
    print("Or, if running from an IDE, ensure uvicorn is started first in a separate process.")

    # This block allows running the server directly for testing
    # For real deployment, uvicorn main:app would be used.
    # To run both server and client in one script, you'd need to manage processes or threads.
    # For this example, we separate the server start and client simulation for clarity.
    # To test, run `uvicorn main:app --reload` in one terminal,
    # then run this script (`python main.py`) in another terminal to see frontend simulation.

    # You can uncomment the following line to start uvicorn directly in the same process (blocking)
    # uvicorn.run(app, host="127.0.0.1", port=8000)

    # If you want to run the client simulation after starting the server (manually or in a separate process):
    # simulate_frontend_interaction()
