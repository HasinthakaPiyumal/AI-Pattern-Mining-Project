import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

def get_llm():
    return ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL_NAME)

class QueryDecompositionService:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at breaking down complex customer queries into independent sub-problems. Provide a list of concise questions that, when answered, will fully address the original complex query. Each sub-problem should be solvable independently."),
            ("human", "{complex_query}\n\nList the sub-problems, one per line, prefixed with a hyphen:")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def decompose_query(self, complex_query: str) -> list[str]:
        response = self.chain.invoke({"complex_query": complex_query})
        sub_problems_raw = response["text"].strip().split('\n')
        sub_problems = [p.lstrip('- ').strip() for p in sub_problems_raw if p.strip()]
        return sub_problems

class SubProblemSolverService:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant tasked with providing a concise and accurate answer to a specific sub-problem related to a larger customer query."),
            ("human", "Solve the following sub-problem: {sub_problem}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    async def solve_subproblem(self, sub_problem: str) -> str:
        response = await self.chain.ainvoke({"sub_problem": sub_problem})
        return response["text"].strip()

class AnswerGenerationService:
    def __init__(self, subproblem_solver: SubProblemSolverService):
        self.subproblem_solver = subproblem_solver

    async def generate_final_answer(self, sub_problems: list[str]) -> str:
        tasks = [self.subproblem_solver.solve_subproblem(sp) for sp in sub_problems]
        individual_answers = await asyncio.gather(*tasks)
        final_answer = "\n\n".join(individual_answers)
        return final_answer

app = FastAPI()

llm = get_llm()
query_decomposition_service = QueryDecompositionService(llm)
subproblem_solver_service = SubProblemSolverService(llm)
answer_generation_service = AnswerGenerationService(subproblem_solver_service)

class QueryRequest(BaseModel):
    complex_query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    sub_problems = query_decomposition_service.decompose_query(request.complex_query)
    final_answer = await answer_generation_service.generate_final_answer(sub_problems)
    return {"answer": final_answer}