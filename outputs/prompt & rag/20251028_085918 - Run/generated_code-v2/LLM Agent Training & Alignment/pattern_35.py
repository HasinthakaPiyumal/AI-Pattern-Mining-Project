import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import random
import time
import gradio as gr

# --- I. Data Collection & Management ---

# In-memory data stores for simplicity
demonstration_data = []
comparison_data = []

class DemonstrationRecorder:
    def __init__(self):
        self.session_id = 0

    def record_session(self, customer_query: str, expert_actions: List[Dict[str, str]], observations: List[Dict[str, str]]):
        self.session_id += 1
        session_record = {
            "session_id": self.session_id,
            "customer_query": customer_query,
            "expert_actions": expert_actions,
            "observations": observations,
            "timestamp": time.time()
        }
        demonstration_data.append(session_record)
        return session_record

# Simulate human expert interaction and data recording
def simulate_demonstration_data(num_sessions: int = 5):
    print("Simulating demonstration data collection...")
    recorder = DemonstrationRecorder()
    for i in range(num_sessions):
        query = f"Customer query {i+1}: I have an issue with my order {1000 + i}."
        actions = [
            {"type": "search_knowledge_base", "query": f"order {1000 + i}"},
            {"type": "read_article", "article_id": f"KB-00{i}"},
            {"type": "update_crm", "field": "status", "value": "investigating"}
        ]
        observations = [
            {"type": "search_results", "results": [f"Article KB-00{i}", f"FAQ-00{i}"]},
            {"type": "article_content", "content": "Solution details..."},
            {"type": "crm_update_status", "success": True}
        ]
        recorder.record_session(query, actions, observations)
    print(f"Collected {len(demonstration_data)} demonstration sessions.")

class ComparisonCollector:
    def collect_comparison(self, query: str, response_a: str, response_b: str, preferred_response: str, feedback: str = None):
        comparison_record = {
            "query": query,
            "response_a": response_a,
            "response_b": response_b,
            "preferred_response": preferred_response,
            "feedback": feedback,
            "timestamp": time.time()
        }
        comparison_data.append(comparison_record)
        return comparison_record

# Gradio interface for comparison data collection (runs separately)
def comparison_ui_function(query, response_a, response_b, preference):
    collector = ComparisonCollector()
    preferred_response = response_a if preference == "Response A" else response_b
    collector.collect_comparison(query, response_a, response_b, preferred_response)
    return f"Preference recorded for query: {query}"

comparison_if = gr.Interface(
    fn=comparison_ui_function,
    inputs=[
        gr.Textbox(label="Customer Query"),
        gr.Textbox(label="Response A"),
        gr.Textbox(label="Response B"),
        gr.Radio(["Response A", "Response B"], label="Preferred Response")
    ],
    outputs="text",
    title="Collect Human Preferences for Agent Responses",
    description="Help train the reward model by selecting the better response."
)


# --- II. Model Training Pipeline ---

# A. Behavior Cloning (BC) Module (Placeholder)
class BCPolicy(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_actions):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, num_actions)

    def forward(self, input_sequence):
        embedded = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(embedded)
        actions = self.action_head(lstm_out[:, -1, :])
        return actions

def train_bc_model(demonstrations: List[Dict]):
    print(f"Training Behavior Cloning Model with {len(demonstration_data)} demonstrations...")
    # In a real scenario, this would involve tokenization, dataset creation,
    # and a proper PyTorch training loop with an optimizer and loss function.
    # For simplicity, we'll just acknowledge the training.
    time.sleep(1)
    vocab_size = 1000
    hidden_size = 256
    num_actions = 10 # Example number of distinct actions
    bc_model = BCPolicy(vocab_size, hidden_size, num_actions)
    print("Behavior Cloning Model training simulated. Model is a placeholder.")
    return bc_model

# B. Reward Modeling (RM) Module (Placeholder)
class RewardModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1) # Output a score
        self.sigmoid = nn.Sigmoid() # To get a preference probability

    def forward(self, text_a, text_b):
        # In a real RM, you'd feed both texts and learn to predict preference
        # For this placeholder, we simulate a comparison
        # Tokenize and get embeddings, then compare
        inputs_a = self.tokenizer(text_a, return_tensors="pt", truncation=True, padding=True)
        inputs_b = self.tokenizer(text_b, return_tensors="pt", truncation=True, padding=True)
        
        # Simplified: just compare a dummy score
        score_a = self.model(**inputs_a).logits
        score_b = self.model(**inputs_b).logits
        
        # Learn to predict if score_a > score_b
        # This is a highly simplified representation of a reward model's output
        preference_logit = score_a - score_b
        return self.sigmoid(preference_logit) # Probability that A is preferred over B

def train_reward_model(comparisons: List[Dict]):
    print(f"Training Reward Model with {len(comparisons)} comparisons...")
    # This would involve creating a dataset from comparison_data,
    # fine-tuning a pre-trained language model for preference prediction.
    # For simplicity, we return a dummy model.
    time.sleep(1)
    reward_model = RewardModel()
    print("Reward Model training simulated. Model is a placeholder.")
    return reward_model

# C. Reinforcement Learning from Human Feedback (RLHF) Module (Placeholder)
def train_rlhf_agent(bc_policy, reward_model):
    print("Initiating RLHF training to refine agent policy...")
    # This is where Hugging Face TRL PPOTrainer would be used.
    # It involves generating responses with the BC policy, getting rewards from the RM,
    # and updating the policy using PPO.
    time.sleep(2)
    print("RLHF training simulated. Final Agentic LLM is a placeholder.")
    final_agent_llm = bc_policy # For demonstration, the final agent is just the BC policy
    return final_agent_llm

# --- III. Agent Deployment & Inference ---

# Mock External Systems
class CRMSystem:
    def get_customer_info(self, customer_id: str): return {"id": customer_id, "name": "John Doe", "status": "Active"}
    def update_status(self, customer_id: str, status: str): return {"success": True, "new_status": status}

class KnowledgeBase:
    def search(self, query: str): return [f"Article on '{query}'", "General FAQ"]
    def get_article(self, article_id: str): return f"Content of {article_id}: Details..."

crm_api = CRMSystem()
kb_api = KnowledgeBase()

# LangChain Tool Integration (Mock Tools)
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

class CRMLookupTool(BaseTool):
    name = "CRM Lookup"
    description = "Looks up customer information in the CRM system."

    def _run(self, customer_id: str) -> str:
        info = crm_api.get_customer_info(customer_id)
        return str(info)

    async def _arun(self, customer_id: str) -> str: return self._run(customer_id)

class KBKnowledgeSearchTool(BaseTool):
    name = "Knowledge Base Search"
    description = "Searches the knowledge base for relevant articles."

    def _run(self, query: str) -> str:
        results = kb_api.search(query)
        return str(results)

    async def _arun(self, query: str) -> str: return self._run(query)

tools = [CRMLookupTool(), KBKnowledgeSearchTool()]

# Agentic LLM Runtime (FastAPI)
app = FastAPI()

class CustomerQuery(BaseModel):
    query: str
    customer_id: str = None
    conversation_history: List[Dict[str, str]] = []

# Placeholder for the actual Agentic LLM
class AgenticLLM:
    def __init__(self, bc_policy_model, reward_model):
        self.bc_policy = bc_policy_model
        self.reward_model = reward_model
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        # Use a small T5 model for response generation as a placeholder for the fine-tuned RLHF agent
        self.llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", tokenizer=self.tokenizer)
        
        # LangChain agent setup
        self.prompt = PromptTemplate.from_template(
            """You are a helpful customer support agent. 
            You have access to the following tools: {tools}. 
            Use them to answer customer queries. Respond empathetically and accurately.
            Conversation history: {conversation_history}
            Customer ID: {customer_id}
            Question: {input}
            {agent_scratchpad}"""
        )
        self.agent = create_react_agent(self.llm_pipeline.model, tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)

    def generate_response(self, query: str, customer_id: str, conversation_history: List[Dict[str, str]]) -> str:
        # Simulate LLM decision making and tool use
        print(f"Agent received query: {query} for customer {customer_id}")
        try:
            # LangChain agent to handle complex logic and tool calls
            response = self.agent_executor.invoke({
                "input": query,
                "customer_id": customer_id,
                "conversation_history": conversation_history,
                "tools": tools # Pass tools to the prompt
            })
            return response["output"]
        except Exception as e:
            print(f"Error in agent execution: {e}")
            return "I apologize, I'm having trouble processing that request right now. Could you please rephrase or provide more details?"

agentic_llm_instance = None # Will be initialized after training

@app.post("/support_query")
async def support_query(query_data: CustomerQuery):
    if agentic_llm_instance is None:
        return {"error": "Agent is not yet trained or initialized.", "response": "Please try again later."}
    
    response = agentic_llm_instance.generate_response(
        query_data.query,
        query_data.customer_id,
        query_data.conversation_history
    )
    
    # Basic logging for monitoring
    print(f"Agent Response: {response}")
    
    return {"response": response}

# --- IV. Monitoring & Evaluation (Basic Logging) ---
def monitor_agent_performance():
    print("--- Agent Performance Monitoring ---")
    print(f"Total demonstration sessions collected: {len(demonstration_data)}")
    print(f"Total comparison preferences collected: {len(comparison_data)}")
    # In a real system, this would involve dashboards, metrics, feedback loops

# Main execution flow
if __name__ == "__main__":
    # 1. Data Collection
    simulate_demonstration_data()
    
    # You would typically run the Gradio interface in a separate process or script for actual data collection.
    # For this consolidated file, we'll just acknowledge its existence.
    print("Run 'comparison_if.launch()' in a separate cell/script to start the comparison data collection UI.")
    
    # Simulate some comparison data for training purposes
    collector = ComparisonCollector()
    collector.collect_comparison("Router not working", "Try restarting it.", "Let's troubleshoot connection issues, first check the cables and then restart your router. If it persists, we can check your service status.", "Let's troubleshoot connection issues, first check the cables and then restart your router. If it persists, we can check your service status.")
    collector.collect_comparison("Billing question", "Your bill is $50.", "I can help clarify your billing. Could you tell me what specific charge or item on your bill you'd like to understand better?", "I can help clarify your billing. Could you tell me what specific charge or item on your bill you'd like to understand better?")

    # 2. Model Training Pipeline
    bc_model = train_bc_model(demonstration_data)
    reward_model = train_reward_model(comparison_data)
    final_agent_llm_model = train_rlhf_agent(bc_model, reward_model)

    # 3. Agent Deployment & Inference
    # Initialize the global agentic_llm_instance after training
    agentic_llm_instance = AgenticLLM(final_agent_llm_model, reward_model)
    print("Agentic LLM initialized and ready for deployment via FastAPI.")
    print("To run the FastAPI server, use: uvicorn customer_support_agent:app --reload")

    # 4. Monitoring (example call)
    monitor_agent_performance()

    # You can also launch Gradio for comparison collection here if you want it to block,
    # but typically it's run separately.
    # comparison_if.launch()
