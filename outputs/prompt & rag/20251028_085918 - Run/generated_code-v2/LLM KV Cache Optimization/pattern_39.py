from fastapi import FastAPI, Request
from vllm import LLM, SamplingParams

app = FastAPI()

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
llm = LLM(model=model_name, trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt")

    if not prompt:
        return {"error": "Prompt not provided"}, 400

    outputs = await llm.generate(prompts=[prompt], sampling_params=sampling_params)

    generated_text = ""
    if outputs and outputs[0].outputs:
        generated_text = outputs[0].outputs[0].text
    
    return {"response": generated_text}