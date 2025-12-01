import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import requests
import uuid
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables (for OpenAI API key, etc.)
load_dotenv()

# --- 1. Agentic Working Memory (Core Module) ---

class WorkingMemoryState(BaseModel):
    q: Optional[str] = None  # current user query
    e: Optional[List[str]] = None  # consolidated external evidence
    o: Optional[List[str]] = None  # LLM-generated candidate responses
    u: Optional[List[float]] = None  # utility scores for responses
    f: Optional[str] = None  # verbalized feedback
    hq: List[Dict[str, str]] = []  # complete dialog history

class WorkingMemoryManager:
    def __init__(self):
        self.sessions: Dict[str, WorkingMemoryState] = {}

    def get_state(self, session_id: str) -> WorkingMemoryState:
        if session_id not in self.sessions:
            self.sessions[session_id] = WorkingMemoryState()
        return self.sessions[session_id]

    def update_state(self, session_id: str, **kwargs):
        current_state = self.get_state(session_id)
        for key, value in kwargs.items():
            setattr(current_state, key, value)
        self.sessions[session_id] = current_state

    def add_history_entry(self, session_id: str, role: str, content: str):
        state = self.get_state(session_id)
        state.hq.append({"role": role, "content": content})
        self.update_state(session_id, hq=state.hq)

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

# Global instance of WorkingMemoryManager
memory_manager = WorkingMemoryManager()

# --- 2. External Evidence Retrieval Module ---

class ExternalEvidenceRetrievalModule:
    def __init__(self):
        # Simulated product data, FAQs, return policies
        self.documents = [
            "Product A: High-quality headphones with noise cancellation. Price: $199.99. Warranty: 1 year.",
            "Product B: Ergonomic office chair, adjustable lumbar support. Price: $249.00. Assembly required.",
            "FAQ: How to return an item? Items can be returned within 30 days of purchase, unworn and in original packaging.",
            "FAQ: Shipping policy: Standard shipping takes 3-5 business days. Expedited options available.",
            "Troubleshooting: Product A not connecting? Ensure Bluetooth is on and device is charged."
        ]
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma.from_texts(texts=self.documents, embedding=self.embeddings)

    def retrieve_evidence(self, query: str, history: List[Dict[str, str]]) -> List[str]:
        # Combine query with recent history for better context
        context = " ".join([entry["content"] for entry in history[-3:]]) + " " + query if history else query
        docs = self.vectorstore.similarity_search(context, k=2)
        return [doc.page_content for doc in docs]

# Global instance of ExternalEvidenceRetrievalModule
evidence_retriever = ExternalEvidenceRetrievalModule()

# --- 3. LLM Interaction Module ---

class LLMInteractionModule:
    def __init__(self):
        # Placeholder for OpenAI or other LLM client
        pass

    def generate_response(self, prompt: str) -> str:
        # In a real application, this would call an LLM API (e.g., OpenAI, Google Gemini)
        # For this example, we'll return a placeholder or a simple canned response based on keywords
        if "return" in prompt.lower():
            return "You can return items within 30 days of purchase. Please ensure they are unworn and in original packaging."
        elif "shipping" in prompt.lower():
            return "Standard shipping usually takes 3-5 business days."
        elif "product A" in prompt.lower() or "headphones" in prompt.lower():
            return "Product A are high-quality noise-cancelling headphones with a 1-year warranty."
        elif "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! How can I assist you with your e-commerce queries today?"
        else:
            return f"I'm sorry, I need more information or am currently unable to answer that. Could you please rephrase or provide more details? (Prompt received: {prompt[:50]}...)"

# Global instance of LLMInteractionModule
llm_interactor = LLMInteractionModule()

# --- 4. Prompt Engine ---

class PromptEngine:
    def __init__(self):
        self.template = PromptTemplate(
            input_variables=["query", "evidence", "history"],
            template=(
                "You are an AI customer support agent for an e-commerce platform. "
                "Your goal is to provide helpful and accurate information based on the user's query and available evidence. "
                "Be concise and polite.\n\n"
                "Conversation History:\n{history}\n\n"
                "Relevant Information:\n{evidence}\n\n"
                "User Query: {query}\n\n"
                "Agent Response:"
            )
        )

    def construct_prompt(self, query: str, evidence: List[str], history: List[Dict[str, str]]) -> str:
        history_str = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in history])
        evidence_str = "\n".join(evidence) if evidence else "No specific external evidence found."
        return self.template.format(query=query, evidence=evidence_str, history=history_str)

# Global instance of PromptEngine
prompt_engine = PromptEngine()

# --- 5. Policy Module ---

class PolicyModule:
    def __init__(self):
        pass

    def decide_action(self, state: WorkingMemoryState) -> Dict[str, Any]:
        # Simple rule-based policy for demonstration
        if not state.q:
            return {"action": "greet", "response": "Hello! How can I help you today?"}

        if state.e and "sorry" not in state.o[0].lower() and "unable to answer" not in state.o[0].lower():
            # If evidence was found and LLM provided a decent response
            return {"action": "respond", "response": state.o[0]}
        elif not state.e and len(state.hq) > 2 and state.q.strip().endswith("?"):
            # If no evidence and it's a multi-turn conversation with a question
            return {"action": "clarify", "response": "I couldn't find specific information for that. Could you please provide more details or rephrase your question?"}
        else:
            # Default to LLM response or a general fallback
            return {"action": "respond", "response": state.o[0] if state.o else "I'm having trouble understanding. Could you please clarify?"}

# Global instance of PolicyModule
policy_module = PolicyModule()

# --- 6. Orchestration/Agent Flow ---

class AgentOrchestrator:
    def __init__(self, memory_manager: WorkingMemoryManager, 
                 evidence_retriever: ExternalEvidenceRetrievalModule, 
                 llm_interactor: LLMInteractionModule, 
                 prompt_engine: PromptEngine, 
                 policy_module: PolicyModule):
        self.memory_manager = memory_manager
        self.evidence_retriever = evidence_retriever
        self.llm_interactor = llm_interactor
        self.prompt_engine = prompt_engine
        self.policy_module = policy_module

    def process_user_query(self, session_id: str, user_query: str) -> str:
        # 1. Update Working Memory with new user query
        self.memory_manager.add_history_entry(session_id, "user", user_query)
        self.memory_manager.update_state(session_id, q=user_query, o=None, u=None, f=None)

        current_state = self.memory_manager.get_state(session_id)

        # 2. Trigger External Evidence Retrieval
        evidence = self.evidence_retriever.retrieve_evidence(user_query, current_state.hq)
        self.memory_manager.update_state(session_id, e=evidence)

        # 3. Construct Prompt
        current_state = self.memory_manager.get_state(session_id) # Re-fetch state after update
        prompt = self.prompt_engine.construct_prompt(
            query=current_state.q,
            evidence=current_state.e,
            history=current_state.hq
        )

        # 4. LLM Interaction
        llm_response = self.llm_interactor.generate_response(prompt)
        
        # For simplicity, treat LLM response as the only candidate and assign a dummy utility score
        candidate_responses = [llm_response]
        utility_scores = [0.8] # Dummy score
        self.memory_manager.update_state(session_id, o=candidate_responses, u=utility_scores)

        # 5. Policy Module Decision
        current_state = self.memory_manager.get_state(session_id) # Re-fetch state after update
        policy_decision = self.policy_module.decide_action(current_state)

        agent_response = policy_decision["response"]
        self.memory_manager.add_history_entry(session_id, "agent", agent_response)
        
        # Optional: Update feedback based on policy (e.g., if policy decided to clarify)
        if policy_decision["action"] == "clarify":
            self.memory_manager.update_state(session_id, f="Agent requested clarification.")

        return agent_response

# Global instance of AgentOrchestrator
orchestrator = AgentOrchestrator(
    memory_manager=memory_manager,
    evidence_retriever=evidence_retriever,
    llm_interactor=llm_interactor,
    prompt_engine=prompt_engine,
    policy_module=policy_module
)

# --- FastAPI API Gateway ---

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    history: List[Dict[str, str]]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    
    agent_response = orchestrator.process_user_query(session_id, request.message)
    
    current_state = memory_manager.get_state(session_id)
    
    return ChatResponse(session_id=session_id, response=agent_response, history=current_state.hq)

@app.post("/clear_session")
async def clear_session_endpoint(session_id: str):
    memory_manager.clear_session(session_id)
    return {"message": f"Session {session_id} cleared."}

# --- Streamlit UI ---

def streamlit_app():
    st.title("E-commerce Support Agent")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.sidebar.subheader("Session Info")
    st.sidebar.write(f"Session ID: {st.session_state.session_id}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about your order or our products..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call FastAPI backend
        try:
            response = requests.post(
                "http://localhost:8000/chat", 
                json={
                    "session_id": st.session_state.session_id,
                    "message": prompt
                }
            )
            response.raise_for_status() # Raise an exception for HTTP errors
            chat_response = response.json()
            agent_response_content = chat_response["response"]
            st.session_state.session_id = chat_response["session_id"]
            
            with st.chat_message("assistant"):
                st.markdown(agent_response_content)
            st.session_state.messages.append({"role": "assistant", "content": agent_response_content})
            
            # Update Streamlit's history with the full history from the backend
            # This is optional, as Streamlit's own messages list already tracks turns
            # st.session_state.messages = chat_response["history"]
            
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Please ensure it is running (run 'uvicorn ecommerce_support_agent:app --reload' in your terminal).")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

    if st.sidebar.button("Clear Chat"):
        requests.post(f"http://localhost:8000/clear_session?session_id={st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# To run this application:
# 1. Save this code as `ecommerce_support_agent.py`.
# 2. Install necessary libraries: `pip install streamlit fastapi uvicorn pydantic requests langchain-community sentence-transformers python-dotenv`
# 3. Run the FastAPI backend: `uvicorn ecommerce_support_agent:app --reload` in one terminal.
# 4. Run the Streamlit frontend: `streamlit run ecommerce_support_agent.py` in another terminal.
# Ensure you have an environment variable `OPENAI_API_KEY` set if you were to integrate with actual OpenAI. For this example, it's not strictly needed due to the mocked LLM response.

# Main entry point for Streamlit
if __name__ == "__main__":
    # This block will only run when the file is executed directly by Streamlit
    # It will not interfere with uvicorn running the FastAPI app.
    streamlit_app()
