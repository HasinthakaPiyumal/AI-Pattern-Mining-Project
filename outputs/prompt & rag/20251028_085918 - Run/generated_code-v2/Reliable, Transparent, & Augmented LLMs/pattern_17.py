import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

client = OpenAI(api_key=openai_api_key)

app = FastAPI()

class Ticket(BaseModel):
    description: str

@app.post("/triage")
async def triage_ticket(ticket: Ticket):
    prompt = f"Evaluate the following customer support ticket description for urgency. Respond only with 'Urgent' or 'Not Urgent'.\n\nTicket: {ticket.description}"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that triages customer support tickets for urgency."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=5,
            stop=["\n"]
        )
        
        llm_response_content = response.choices[0].message.content.strip()
        
        if "urgent" in llm_response_content.lower():
            urgency = "Urgent"
        elif "not urgent" in llm_response_content.lower():
            urgency = "Not Urgent"
        else:
            urgency = "Undetermined"

        return {"urgency": urgency}

    except Exception as e:
        return {"error": str(e), "urgency": "Error"}

