from fastapi import FastAPI
from vllm import LLM, SamplingParams
import asyncio

app = FastAPI()

# Initialize vLLM with a pre-trained model
# Using a smaller model for demonstration. For a real co-pilot, a larger model would be used.
llm = LLM(model="facebook/opt-125m")
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

@app.post("/generate")
async def generate_response(prompt: str, history: list = None):
    conversation_history = ""
    if history:
        for turn in history:
            conversation_history += f"{turn['role']}: {turn['content']}\n"
    
    full_prompt = f"{conversation_history}Customer: {prompt}\nCo-pilot:"
    
    outputs = llm.generate([full_prompt], sampling_params)
    
    generated_text = outputs[0].outputs[0].text
    return {"response": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)