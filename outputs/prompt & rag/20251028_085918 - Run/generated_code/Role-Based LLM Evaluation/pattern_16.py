
import os
import uuid
from typing import List, Dict
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# --- Pydantic Models ---
class EssaySubmission(BaseModel):
    essay_text: str

class PersonaFeedback(BaseModel):
    persona: str
    feedback: str

class EvaluationResult(BaseModel):
    essay_id: str
    essay_text: str
    individual_feedback: List[PersonaFeedback]
    overall_feedback: str
    overall_score: int

# --- EssayEvaluator Class ---
class EssayEvaluator:
    def __init__(self, llm):
        self.llm = llm
        self.personas = [
            {
                "name": "Critical Thinker",
                "system_prompt": (
                    "You are a critical thinking evaluator. Assess the logical coherence, argumentation strength, "
                    "and depth of analysis in the essay. Provide a score out of 10 and a brief explanation "
                    "of your reasoning, highlighting strengths and weaknesses in critical thought."
                ),
            },
            {
                "name": "Literary Analyst",
                "system_prompt": (
                    "You are a literary analyst. Evaluate the essay's style, tone, use of language, "
                    "creativity, and engagement. Provide a score out of 10 and a brief explanation, "
                    "focusing on the literary merits and impact."
                ),
            },
            {
                "name": "Grammar Specialist",
                "system_prompt": (
                    "You are a grammar and mechanics specialist. Evaluate the essay for grammatical errors, "
                    "spelling, punctuation, sentence structure, and clarity. Provide a score out of 10 "
                    "and a brief explanation, pointing out areas for improvement in correctness."
                ),
            },
            {
                "name": "Subject Matter Expert",
                "system_prompt": (
                    "You are a subject matter expert. Assess the essay's accuracy of information, "
                    "relevance to the topic, use of evidence, and factual correctness. "
                    "Provide a score out of 10 and a brief explanation based on your expertise."
                ),
            },
        ]

        self.synthesizer_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an overall essay feedback synthesizer. You will receive feedback from multiple evaluative personas. "
                "Your task is to consolidate this feedback into a single, comprehensive overall evaluation. "
                "Provide a final overall score out of 10, considering all perspectives, and a summary of the key strengths "
                "and areas for improvement. Format your response clearly with 'OVERALL SCORE: [score]/10' "
                "followed by 'OVERALL FEEDBACK: [summary]'."
            )),
            ("human", "Here is the individual feedback for an essay:\n\n{feedback_summary}\n\nBased on this, provide an overall score and comprehensive feedback.")
        ])

        self.synthesizer_chain = self.synthesizer_prompt | self.llm | StrOutputParser()

    def _create_persona_chain(self, persona_name: str, system_prompt: str):
        return (
            ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Evaluate the following essay: {essay_text}")
            ])
            | self.llm
            | StrOutputParser()
        )

    async def evaluate_essay(self, essay_text: str) -> Dict:
        persona_chains = {}
        for persona_data in self.personas:
            persona_name = persona_data["name"]
            system_prompt = persona_data["system_prompt"]
            persona_chains[persona_name] = self._create_persona_chain(persona_name, system_prompt)

        # Run all persona chains in parallel
        parallel_evaluations = RunnableParallel(**persona_chains)
        individual_results = await parallel_evaluations.ainvoke({"essay_text": essay_text})

        individual_feedback_list = []
        feedback_summary = ""
        for persona, feedback in individual_results.items():
            individual_feedback_list.append(PersonaFeedback(persona=persona, feedback=feedback))
            feedback_summary += f"- {persona} Feedback:\n{feedback}\n\n"

        # Synthesize overall feedback
        synthesized_output = await self.synthesizer_chain.ainvoke({"feedback_summary": feedback_summary})

        overall_score = 0
        overall_feedback = ""

        # Parse synthesizer output
        score_match = re.search(r"OVERALL SCORE: (\d+)/10", synthesized_output)
        feedback_match = re.search(r"OVERALL FEEDBACK: (.+)", synthesized_output, re.DOTALL)

        if score_match:
            overall_score = int(score_match.group(1))
        if feedback_match:
            overall_feedback = feedback_match.group(1).strip()
        else:
            # Fallback if parsing fails
            overall_feedback = synthesized_output.strip()
            overall_score = 5 # Default score if parsing fails

        return {
            "individual_feedback": individual_feedback_list,
            "overall_feedback": overall_feedback,
            "overall_score": overall_score
        }

# --- FastAPI Application ---
app = FastAPI(
    title="Multi-Perspective LLM Essay Grader",
    description="An API to evaluate essays using multiple LLM personas and provide synthesized feedback."
)

# In-memory store for essays and their evaluations
essay_store: Dict[str, EvaluationResult] = {}

# Initialize LLM (requires OPENAI_API_KEY to be set in environment variables)
try:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
except Exception as e:
    print(f"Error initializing ChatOpenAI: {e}")
    print("Please ensure OPENAI_API_KEY is set in your environment variables or .env file.")
    llm = None # Set to None to prevent further errors if not initialized

# Initialize EssayEvaluator
if llm is None:
    print("LLM not initialized. The API will not be able to process essay evaluations.")
    essay_evaluator = None
else:
    essay_evaluator = EssayEvaluator(llm)

@app.post("/submit_essay", response_model=EvaluationResult, summary="Submit an essay for multi-perspective evaluation")
async def submit_essay(essay_submission: EssaySubmission):
    if essay_evaluator is None:
        raise HTTPException(status_code=503, detail="LLM service not available. Please check API key configuration.")

    essay_id = str(uuid.uuid4())
    
    # Evaluate the essay
    evaluation_data = await essay_evaluator.evaluate_essay(essay_submission.essay_text)
    
    # Combine with essay details to form the full result
    result = EvaluationResult(
        essay_id=essay_id,
        essay_text=essay_submission.essay_text,
        individual_feedback=evaluation_data["individual_feedback"],
        overall_feedback=evaluation_data["overall_feedback"],
        overall_score=evaluation_data["overall_score"]
    )
    
    essay_store[essay_id] = result
    return result

@app.get("/get_feedback/{essay_id}", response_model=EvaluationResult, summary="Retrieve evaluation feedback for a submitted essay")
async def get_feedback(essay_id: str):
    if essay_id not in essay_store:
        raise HTTPException(status_code=404, detail=f"Essay with ID '{essay_id}' not found.")
    return essay_store[essay_id]

# To run the application:
# 1. Save this code as `main.py`
# 2. Make sure you have an `.env` file in the same directory with `OPENAI_API_KEY="your_openai_api_key"`
# 3. Install necessary libraries: `pip install fastapi "uvicorn[standard]" pydantic python-dotenv langchain-openai langchain-core`
# 4. Run from your terminal: `uvicorn main:app --reload`
# 5. Access the API documentation at http://127.0.0.1:8000/docs
