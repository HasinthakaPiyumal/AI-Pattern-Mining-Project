from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn

app = FastAPI()

# Initialize vLLM LLM serving backend
# This will load the model into GPU memory. Choose a suitable model.
llm = LLM(model="HuggingFaceH4/zephyr-7b-beta", trust_remote_code=True)

class Query(BaseModel):
    customer_query: str

@app.post("/chat")
async def chat_with_assistant(query: Query):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
    
    # vLLM automatically handles dynamic batching and PagedAttention
    # for efficient KV cache management.
    outputs = llm.generate(query.customer_query, sampling_params)
    
    response_text = ""
    for output in outputs:
        for generated_text in output.outputs:
            response_text += generated_text.text
    
    return {"assistant_response": response_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)