from pydantic import BaseModel
from fastapi import FastAPI
import random

class CustomerQuery(BaseModel):
    query: str

class ChatbotResponse(BaseModel):
    response: str

class EvaluationResult(BaseModel):
    query: str
    response_a: str
    response_b: str
    judgment: str

class ChatbotA:
    def generate_response(self, query: str) -> str:
        return f"Chatbot A's response to '{query}': We are looking into this for you and will get back shortly."

class ChatbotB:
    def generate_response(self, query: str) -> str:
        return f"Chatbot B's response to '{query}': Your request is being processed. Expect an update within 24 hours."

class LLMPairwiseEvaluator:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def evaluate(self, query: str, response_a: str, response_b: str) -> str:
        if "price" in query.lower() and "24 hours" in response_b.lower():
            return "Chatbot B is superior (more specific ETA for price query)."
        elif "issue" in query.lower() and "looking into" in response_a.lower():
            return "Chatbot A is superior (acknowledges issue more directly)."
        else:
            choices = ["Chatbot A is superior", "Chatbot B is superior", "Responses are equally good"]
            return random.choice(choices)

app = FastAPI()

chatbot_a = ChatbotA()
chatbot_b = ChatbotB()
llm_evaluator = LLMPairwiseEvaluator()

@app.post("/evaluate_responses", response_model=EvaluationResult)
async def evaluate_responses(customer_query: CustomerQuery):
    response_a = chatbot_a.generate_response(customer_query.query)
    response_b = chatbot_b.generate_response(customer_query.query)
    
    judgment = llm_evaluator.evaluate(customer_query.query, response_a, response_b)
    
    return EvaluationResult(
        query=customer_query.query,
        response_a=response_a,
        response_b=response_b,
        judgment=judgment
    )