from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn

app = FastAPI()

# Initialize vLLM LLM instance
# Note: Replace 'meta-llama/Llama-2-7b-chat-hf' with your desired model
# Ensure the model is downloaded or accessible to vLLM
llm = LLM(model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", trust_remote_code=True)

# Global dictionary to store conversation history
conversation_history = {}

class ChatMessage(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(chat_message: ChatMessage):
    session_id = chat_message.session_id
    user_message = chat_message.message

    if session_id not in conversation_history:
        conversation_history[session_id] = []

    # Add user message to history
    conversation_history[session_id].append({"role": "user", "content": user_message})

    # Construct prompt from history
    prompt_messages = []
    for entry in conversation_history[session_id]:
        if entry["role"] == "user":
            prompt_messages.append(f"[INST] {entry["content"]} [/INST]")
        elif entry["role"] == "assistant":
            prompt_messages.append(f"{entry["content"]}")
    
    # For simplicity, we'll just join them. In a real scenario, you might want more sophisticated prompt formatting.
    prompt = "\n".join(prompt_messages) + "\n"

    # Sampling parameters for vLLM
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

    # Generate response using vLLM
    try:
        outputs = llm.generate(prompt, sampling_params)
        bot_response = outputs[0].outputs[0].text.strip()
    except Exception as e:
        bot_response = f"Error: Could not generate response. {e}"
        print(f"vLLM generation error: {e}")

    # Add bot response to history
    conversation_history[session_id].append({"role": "assistant", "content": bot_response})

    return {"response": bot_response}

# To run this file: uvicorn app:app --host 0.0.0.0 --port 8000 --reload