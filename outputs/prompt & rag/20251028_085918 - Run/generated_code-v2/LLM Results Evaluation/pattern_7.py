import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import gradio as gr

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables from .env file
load_dotenv()

# --- FastAPI Backend --- #
app = FastAPI()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it.")

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.7)

# Define evaluation roles and base instructions
ROLES = [
    "Pedagogical Expert",
    "Student Learner",
    "Subject Matter Specialist",
    "Accessibility Advocate",
]

BASE_INSTRUCTIONS = (
    "You are an AI assistant tasked with evaluating educational content. "
    "Please provide constructive feedback based on your assigned role. "
    "Focus on clarity, engagement, accuracy, and overall effectiveness. "
    "Keep your feedback concise and actionable."
)

# Pydantic models for request and response
class EvaluationRequest(BaseModel):
    content: str

class RoleEvaluation(BaseModel):
    role: str
    feedback: str

class EvaluationResponse(BaseModel):
    evaluations: List[RoleEvaluation]

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_content_backend(request: EvaluationRequest):
    content = request.content
    all_feedback = []

    async def get_feedback_for_role(role: str):
        prompt_template = PromptTemplate.from_template(
            f"Act as a {role}. {BASE_INSTRUCTIONS}\n\nEducational Content:\n\n{{content}}\n\nProvide your evaluation from the perspective of a {role}:"
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = await chain.ainvoke({"content": content})
        return RoleEvaluation(role=role, feedback=response["text"])

    tasks = [get_feedback_for_role(role) for role in ROLES]
    feedback_results = await asyncio.gather(*tasks)
    all_feedback.extend(feedback_results)

    return EvaluationResponse(evaluations=all_feedback)

# --- Gradio Frontend --- #

async def evaluate_content_frontend(content: str) -> str:
    if not content.strip():
        return "Please provide some educational content to evaluate."

    # In a real deployment, replace with the actual backend URL
    # For local development, assume FastAPI is running on default port
    backend_url = "http://127.0.0.1:8000/evaluate"

    try:
        # Using a simple HTTP client to interact with FastAPI. 
        # In a production setup, consider `httpx` for async requests.
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(backend_url, json={"content": content})
            response.raise_for_status()
            data = response.json()

            formatted_feedback = []
            for eval_item in data["evaluations"]:
                formatted_feedback.append(f"**{eval_item['role']}**: {eval_item['feedback']}")
            return "\n\n---\n\n".join(formatted_feedback)
    except httpx.RequestError as e:
        return f"Error connecting to the backend: {e}. Please ensure the FastAPI backend is running."
    except httpx.HTTPStatusError as e:
        return f"Backend returned an error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Gradio Interface
iface = gr.Interface(
    fn=evaluate_content_frontend,
    inputs=gr.Textbox(lines=10, label="Educational Content", placeholder="Paste your lesson plan, textbook chapter, or course module here..."),
    outputs=gr.Markdown(label="Diverse Evaluations"),
    title="EduCritique: Role-based Educational Content Evaluator",
    description="Upload your educational content and receive diverse feedback from AI agents acting as different evaluators."
)

# To run both FastAPI and Gradio in the same script, 
# we need to ensure FastAPI starts first and Gradio can connect to it.
# For simple execution, this script can be run as `python edu_critique_app.py`
# and then access Gradio at its exposed URL. 
# The FastAPI app will be running in the background for Gradio to call.

# To run FastAPI explicitly:
# uvicorn edu_critique_app:app --reload

# To run Gradio (which will call the FastAPI backend):
# if __name__ == "__main__":
#     import uvicorn
#     from threading import Thread

#     def run_fastapi():
#         uvicorn.run(app, host="127.0.0.1", port=8000)

#     # Start FastAPI in a separate thread
#     fastapi_thread = Thread(target=run_fastapi)
#     fastapi_thread.start()

#     # Start Gradio
#     iface.launch(share=True) # share=True for a public link (careful with API keys)

# For simplicity, and as per the request for a single code file, 
# we'll provide the script structure. Users would typically run FastAPI and Gradio 
# as separate processes or use a single entry point that manages both.
# The current setup allows running Gradio which then calls a FastAPI instance 
# assumed to be running independently, or one could uncomment the threading part
# for a single script run. Given the prompt constraints, I'll provide the logic 
# for both and explain how to run them, but not enforce the threading within the 
# core code output itself, as it complicates the 