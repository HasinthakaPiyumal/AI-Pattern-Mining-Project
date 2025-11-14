from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

# Initialize FastAPI app
app = FastAPI()

# Initialize vLLM with a pre-trained model
# Using a small model for demonstration purposes (e.g., 'facebook/opt-125m')
# For a production system, you would choose a more capable LLM.
llm = LLM(model="facebook/opt-125m")

# Define Pydantic model for request body
class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handles customer queries and returns chatbot responses using vLLM.
    """
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop=["\n", "<|im_end|>", "<|endoftext|>"] # Common stop tokens
    )

    # Generate response using vLLM
    # vLLM automatically handles KV cache optimizations like PagedAttention and KV Cache Reuse.
    outputs = llm.generate(request.prompt, sampling_params)

    # Extract the generated text
    if outputs and outputs[0].outputs:
        generated_text = outputs[0].outputs[0].text.strip()
    else:
        generated_text = "Sorry, I could not generate a response."

    return {"response": generated_text}