from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
import uvicorn

from vllm import LLM, SamplingParams

app = FastAPI()

# Initialize vLLM globally
# For demonstration, using a small model like "facebook/opt-125m"
# In a real scenario, you'd use a larger model and ensure proper GPU setup.
llm = LLM(model="facebook/opt-125m", trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

# In-memory store for conversational history
conversation_history: Dict[str, List[str]] = {}

# Mock E-commerce Database Functions
def get_product_info(product_name: str) -> str:
    products = {"laptop": "A high-performance laptop with 16GB RAM and 512GB SSD.",
                "smartphone": "Latest model smartphone with a 6.7-inch display and 128GB storage.",
                "headphone": "Noise-cancelling headphones with long battery life."
                }
    return products.get(product_name.lower(), "Information not found for this product.")

def get_order_status(order_id: str) -> str:
    orders = {"12345": "Your order #12345 has been shipped and is expected to arrive on July 25th.",
              "67890": "Your order #67890 is being processed."
              }
    return orders.get(order_id, "Order not found.")

class QueryRequest(BaseModel):
    session_id: str
    query: str

@app.post("/chat")
async def chat_with_assistant(request: QueryRequest):
    session_id = request.session_id
    user_query = request.query

    # Get previous conversation history
    history = conversation_history.get(session_id, [])

    # Combine history and current query for LLM input
    # A more sophisticated prompt engineering approach would be used here
    context_prompt = "\n".join(history + [f"Customer: {user_query}"])

    # Simple intent recognition and information retrieval logic based on keywords
    retrieved_info = ""
    if "product" in user_query.lower() and "info" in user_query.lower():
        for product in ["laptop", "smartphone", "headphone"]:
            if product in user_query.lower():
                retrieved_info = get_product_info(product)
                break
    elif "order" in user_query.lower() and "status" in user_query.lower():
        import re
        match = re.search(r'#?(\d{5})', user_query)
        if match:
            order_id = match.group(1)
            retrieved_info = get_order_status(order_id)

    # Construct the final prompt for the LLM
    full_prompt = f"""The following is a conversation with an AI customer support assistant for an e-commerce platform. The assistant is helpful, polite, and provides relevant information.

{context_prompt}
"""
    if retrieved_info:
        full_prompt += f"\nRelevant Information: {retrieved_info}"
    full_prompt += "\nAssistant:"

    # Generate response using vLLM
    outputs = llm.generate([full_prompt], sampling_params)
    assistant_response = outputs[0].outputs[0].text.strip()

    # Update conversation history
    conversation_history[session_id] = history + [f"Customer: {user_query}", f"Assistant: {assistant_response}"]

    return {"session_id": session_id, "response": assistant_response}

if __name__ == "__main__":
    # To run this, save it as main.py and execute: uvicorn main:app --host 0.0.0.0 --port 8000
    # Ensure you have vLLM and its dependencies (like CUDA-enabled PyTorch) installed and configured.
    uvicorn.run(app, host="0.0.0.0", port=8000)
