from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load a pre-trained LLM (using distilgpt2 for demonstration and speed)
generator = pipeline("text-generation", model="distilgpt2")

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    # Directly feed the query to the LLM
    response = generator(request.query, max_new_tokens=50, num_return_sequences=1)
    
    # Extract the generated text
    answer = response[0]["generated_text"]
    
    # In a real-world scenario, you might want to clean up the prompt from the answer
    # For simplicity, we'll return the raw generated text for now.
    if answer.startswith(request.query):
        answer = answer[len(request.query):].strip()

    return {"question": request.query, "answer": answer}