import os
from typing import List, Dict, Optional
from pydantic import BaseModel

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

class AgenticWorkingMemory(BaseModel):
    user_query: str = ""
    external_evidence: str = ""
    llm_candidate_response: Optional[str] = None
    utility_score: Optional[float] = None
    user_feedback: Optional[str] = None
    dialog_history: List[Dict[str, str]] = []

def simulate_knowledge_base_search(query: str) -> str:
    knowledge_base = {
        "installation issues": "To install the software, please refer to the official documentation here: [link to docs]. Ensure your system meets the minimum requirements.",
        "login problems": "If you are having login problems, try resetting your password. If that doesn't work, contact IT support.",
        "feature x not working": "Feature X requires configuration in the settings. Check the user guide for detailed steps on enabling and configuring Feature X.",
        "general support": "Please provide more details about your issue so I can assist you better."
    }
    for key, value in knowledge_base.items():
        if key in query.lower():
            return value
    return knowledge_base["general support"]

class PromptEngine:
    def create_prompt(self, memory: AgenticWorkingMemory) -> str:
        messages = []
        for item in memory.dialog_history:
            messages.append((item["speaker"], item["message"]))

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful customer support agent for a software product. Provide concise and accurate answers based on the provided context and knowledge base information."),
            *messages,
            ("user", f"User Query: {memory.user_query}"),
            ("system", f"Knowledge Base Information: {memory.external_evidence}"),
            ("system", "Please provide a helpful response to the user's query.")
        ])
        return prompt_template.format_messages(user_query=memory.user_query, external_evidence=memory.external_evidence)

class LLM_Agent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    def get_llm_response(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content

class PolicyModule:
    def decide_action(self, memory: AgenticWorkingMemory) -> str:
        if memory.utility_score is not None and memory.utility_score < 0.3:
            return "escalate_to_human_agent"
        if memory.user_feedback and "not helpful" in memory.user_feedback.lower():
            return "escalate_to_human_agent"
        if not memory.external_evidence or "general support" in memory.external_evidence.lower():
            return "request_more_info"
        if memory.llm_candidate_response:
            return "respond"
        return "continue_processing"

class CustomerSupportAgent:
    def __init__(self):
        self.memory = AgenticWorkingMemory()
        self.prompt_engine = PromptEngine()
        self.llm_agent = LLM_Agent()
        self.policy_module = PolicyModule()

    def process_user_input(self, user_input: str) -> str:
        # Update working memory with current user query
        self.memory.user_query = user_input
        self.memory.dialog_history.append({"speaker": "user", "message": user_input})

        # Simulate knowledge base search
        external_evidence = simulate_knowledge_base_search(user_input)
        self.memory.external_evidence = external_evidence

        # Create prompt for LLM
        prompt = self.prompt_engine.create_prompt(self.memory)

        # Get LLM candidate response
        llm_response = self.llm_agent.get_llm_response(prompt)
        self.memory.llm_candidate_response = llm_response

        # Simulate utility score (could be a more complex evaluation in a real system)
        self.memory.utility_score = 0.8  # Example value

        # Decide action based on policy
        action = self.policy_module.decide_action(self.memory)

        if action == "respond":
            agent_response = self.memory.llm_candidate_response
        elif action == "request_more_info":
            agent_response = "I need a bit more information to help you effectively. Could you please elaborate on your issue?"
        elif action == "escalate_to_human_agent":
            agent_response = "I apologize, but I'm unable to fully resolve your issue. I will escalate this to a human agent who will contact you shortly."
        else: # continue_processing or other states
            agent_response = self.memory.llm_candidate_response or "I'm processing your request. Please wait."

        self.memory.dialog_history.append({"speaker": "agent", "message": agent_response})
        return agent_response

if __name__ == "__main__":
    agent = CustomerSupportAgent()
    print("Customer Support Agent: Hello! How can I assist you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Customer Support Agent: Goodbye!")
            break

        response = agent.process_user_input(user_input)
        print(f"Customer Support Agent: {response}")

        # Simulate user feedback for demonstration purposes
        if "not helpful" in user_input.lower() and agent.memory.llm_candidate_response:
            agent.memory.user_feedback = "The previous response was not helpful."
            agent.memory.utility_score = 0.2 # Lower utility score to trigger escalation

        # Reset user feedback for next turn unless explicitly provided
        else:
            agent.memory.user_feedback = None
            agent.memory.utility_score = None # Reset utility score as well if not provided
