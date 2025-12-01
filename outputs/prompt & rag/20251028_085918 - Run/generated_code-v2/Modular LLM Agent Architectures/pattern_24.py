import os
from typing import Dict, List, Any
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

class WorkingMemory:
    def __init__(self):
        self.user_query: str = ""
        self.external_evidence: List[str] = []
        self.llm_candidate_responses: List[str] = []
        self.utility_scores: Dict[str, float] = {}
        self.verbalized_feedback: str = ""
        self.dialog_history: List[Dict[str, str]] = []

    def update_memory(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if key == "user_query" and value:
                self.dialog_history.append({"role": "user", "content": value})
            if key == "agent_response" and value:
                self.dialog_history.append({"role": "agent", "content": value})

    def retrieve_context(self) -> Dict[str, Any]:
        return {
            "user_query": self.user_query,
            "dialog_history": self.dialog_history,
            "external_evidence": self.external_evidence,
        }

    def clear_memory(self):
        self.user_query = ""
        self.external_evidence = []
        self.llm_candidate_responses = []
        self.utility_scores = {}
        self.verbalized_feedback = ""
        self.dialog_history = []

class LLMIntegration:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)

    def get_response(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content

class PromptEngine:
    def __init__(self):
        self.template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful customer support assistant. Maintain context from the conversation history."),
                MessagesPlaceholder(variable_name="dialog_history"),
                ("user", "{user_query}"),
            ]
        )

    def construct_prompt(self, context: Dict[str, Any]) -> List[Any]:
        formatted_history = []
        for item in context.get("dialog_history", [])[:-1]: # Exclude the current user query that is added separately
            if item["role"] == "user":
                formatted_history.append(HumanMessage(content=item["content"]))
            elif item["role"] == "agent":
                formatted_history.append(AIMessage(content=item["content"]))
        
        return self.template.format_messages(
            user_query=context["user_query"],
            dialog_history=formatted_history
        )


class PolicyModule:
    def decide_action(self, memory: WorkingMemory) -> str:
        # Simple policy: always query LLM for a response
        if memory.user_query:
            return "query_llm"
        return "no_action"

    def select_best_response(self, candidate_responses: List[str]) -> str:
        # Simple policy: just take the first candidate response
        if candidate_responses:
            return candidate_responses[0]
        return "I'm sorry, I couldn't generate a response."

class MainAgent:
    def __init__(self):
        self.memory = WorkingMemory()
        self.llm_integration = LLMIntegration()
        self.prompt_engine = PromptEngine()
        self.policy_module = PolicyModule()

    def process_query(self, user_input: str) -> str:
        # 1. Update Working Memory with new user query
        self.memory.update_memory(user_query=user_input)

        # 2. Policy Module decides action
        action = self.policy_module.decide_action(self.memory)

        final_response = ""

        if action == "query_llm":
            # 3. Construct prompt using Prompt Engine
            context = self.memory.retrieve_context()
            prompt_messages = self.prompt_engine.construct_prompt(context)
            
            # Convert list of message objects to a string for LLMIntegration if it expects a string.
            # For ChatOpenAI .invoke(), it can directly take the list of messages.
            # If LLMIntegration.get_response expects a string, we need to adapt.
            # Let's assume LLMIntegration.get_response expects a string for simplicity based on the architecture description
            # or directly call invoke on self.llm_integration.llm here.
            # Adapting LLMIntegration to accept list of messages for Langchain ChatOpenAI
            
            # A more direct approach if using Langchain's ChatOpenAI: send the list of message objects
            llm_raw_response_object = self.llm_integration.llm.invoke(prompt_messages)
            llm_raw_response_content = llm_raw_response_object.content

            # 4. Store LLM candidate responses in Working Memory
            self.memory.update_memory(llm_candidate_responses=[llm_raw_response_content])

            # 5. Policy Module selects best response
            final_response = self.policy_module.select_best_response(
                self.memory.llm_candidate_responses
            )
        else:
            final_response = "I'm not sure how to respond to that."
        
        # 6. Update Working Memory with agent's response
        self.memory.update_memory(agent_response=final_response)

        return final_response

# API Layer with FastAPI
app = FastAPI()
agent = MainAgent()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_with_agent(request: QueryRequest):
    response = agent.process_query(request.query)
    return {"response": response, "dialog_history": agent.memory.dialog_history}

@app.post("/clear_history")
async def clear_agent_history():
    agent.memory.clear_memory()
    return {"message": "Chat history cleared."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
