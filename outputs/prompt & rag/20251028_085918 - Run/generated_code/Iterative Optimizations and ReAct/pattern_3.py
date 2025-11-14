import os
from typing import List, Dict, Any, TypedDict
import json
import requests
from dotenv import load_dotenv

# Langchain/LangGraph imports
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langgraph.graph import StateGraph, END

# Gradio for UI
import gradio as gr

# FastAPI for API endpoint (optional, for demonstration of architecture)
from fastapi import FastAPI, Request
from pydantic import BaseModel

load_dotenv()

# --- 1. Tool Simulation (Internal and External) ---

# Simulated CRM System
class CRMSimulator:
    def __init__(self):
        self.customers = {
            "cust123": {"name": "Alice Smith", "email": "alice@example.com", "status": "Active"},
            "cust456": {"name": "Bob Johnson", "email": "bob@example.com", "status": "Inactive"},
        }

    @tool("getCustomerInfo", "Retrieve information about a customer by ID.")
    def get_customer_info(self, customer_id: str) -> Dict[str, Any]:
        """Retrieves detailed information for a given customer ID."""
        info = self.customers.get(customer_id)
        if info:
            return {"customer_id": customer_id, **info}
        return {"error": "Customer not found", "customer_id": customer_id}

    @tool("updateCustomerRecord", "Update a specific field for a customer by ID.")
    def update_customer_record(self, customer_id: str, field: str, value: Any) -> Dict[str, Any]:
        """Updates a specific field for a given customer ID with a new value."""
        if customer_id in self.customers:
            if field in self.customers[customer_id]:
                self.customers[customer_id][field] = value
                return {"status": "success", "customer_id": customer_id, "field": field, "new_value": value}
            return {"error": f"Field '{field}' not found for customer {customer_id}"}
        return {"error": "Customer not found", "customer_id": customer_id}

crm_simulator = CRMSimulator()

# Simulated Order Management System (OMS)
class OMSSimulator:
    def __init__(self):
        self.orders = {
            "ord789": {"customer_id": "cust123", "item": "Laptop", "status": "Shipped", "tracking": "TRK12345"},
            "ord101": {"customer_id": "cust456", "item": "Mouse", "status": "Processing"},
        }

    @tool("checkOrderStatus", "Check the current status of an order by ID.")
    def check_order_status(self, order_id: str) -> Dict[str, Any]:
        """Checks the current status of a given order ID."""
        status = self.orders.get(order_id)
        if status:
            return {"order_id": order_id, **status}
        return {"error": "Order not found", "order_id": order_id}

    @tool("modifyOrder", "Modify an existing order (e.g., cancel, change item).")
    def modify_order(self, order_id: str, action: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
        """Modifies an order based on the action (e.g., 'cancel', 'update_item')."""
        if order_id in self.orders:
            if action == "cancel":
                self.orders[order_id]["status"] = "Cancelled"
                return {"status": "success", "order_id": order_id, "action": "cancelled"}
            elif action == "update_item" and details and "new_item" in details:
                self.orders[order_id]["item"] = details["new_item"]
                return {"status": "success", "order_id": order_id, "action": "item_updated", "new_item": details["new_item"]}
            return {"error": "Invalid action or missing details for order modification"}
        return {"error": "Order not found", "order_id": order_id}

oms_simulator = OMSSimulator()

# Simulated External API Tool
class ExternalAPITool:
    @tool("makeHttpRequest", "Make an HTTP request to an external API (GET/POST).")
    def make_http_request(self, method: str, url: str, headers: Dict[str, str] = None, body: Dict[str, Any] = None) -> Dict[str, Any]:
        """Makes an HTTP request to a specified URL with given method, headers, and body."""
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=body)
            else:
                return {"error": "Unsupported HTTP method", "method": method}

            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return {"status": "success", "response": response.json(), "http_status": response.status_code}
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "url": url, "method": method}

external_api_tool = ExternalAPITool()

# Knowledge Base (ChromaDB)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

def initialize_knowledge_base():
    docs = [
        "Our return policy allows returns within 30 days of purchase with a valid receipt.",
        "To reset your password, visit the 'Forgot Password' link on the login page and follow the instructions.",
        "Shipping usually takes 3-5 business days for standard delivery within the country.",
        "For technical support, please visit our support portal or call us at 1-800-TECH-HELP."
    ]
    vectorstore.add_texts(docs)
    print("Knowledge base initialized with example documents.")

# Initialize KB on startup (or check if it exists)
if not os.path.exists("./chroma_db") or not vectorstore.get()['ids']:
    initialize_knowledge_base()

@tool("queryKnowledgeBase", "Query the knowledge base for information.")
def query_knowledge_base(query: str) -> Dict[str, Any]:
    """Queries the internal knowledge base for relevant information based on the input query."""
    results = vectorstore.similarity_search(query, k=2)
    return {"query": query, "results": [doc.page_content for doc in results]}

# List of all available tools
available_tools = [
    crm_simulator.get_customer_info,
    crm_simulator.update_customer_record,
    oms_simulator.check_order_status,
    oms_simulator.modify_order,
    external_api_tool.make_http_request,
    query_knowledge_base
]

# --- 2. Language Model (LLM) --- 

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# Bind tools to the LLM for tool calling capabilities
llm_with_tools = llm.bind_tools(available_tools)

# --- 3. Agent State Definition (LangGraph) ---

class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    tool_output: Any
    reflection: str
    plan: str
    escalate: bool

# --- 4. LangGraph Nodes ---

def plan_node(state: AgentState) -> Dict[str, Any]:
    """Determines the next action (tool use or response) based on the input and history."""
    print("\n--- Entering Plan Node ---")
    messages = [
        SystemMessage("You are an intelligent customer support agent. Your goal is to resolve customer inquiries efficiently by using the available tools. If you need to use a tool, specify the tool name and arguments. If you have enough information to answer, generate a response. If the query is complex and requires human intervention after multiple attempts or clearly states a need for human, set escalate to True."),
        *state["chat_history"],
        HumanMessage(content=state["input"])
    ]
    
    response = llm_with_tools.invoke(messages)
    
    # Langchain's tool calling will return a tool_call message if a tool is invoked.
    # Otherwise, it's a regular AIMessage.
    if response.tool_calls:
        print(f"Agent plans to use tool: {response.tool_calls[0].name}")
        return {"tool_output": response.tool_calls[0], "plan": f"Using tool: {response.tool_calls[0].name}"}
    else:
        print("Agent plans to generate response.")
        return {"plan": "Generating response", "tool_output": None, "chat_history": state["chat_history"] + [AIMessage(content=response.content)]}

def execute_tool_node(state: AgentState) -> Dict[str, Any]:
    """Executes the tool chosen in the planning phase."""
    print("\n--- Entering Execute Tool Node ---")
    tool_call = state["tool_output"]
    if not tool_call:
        return state # No tool to execute

    tool_name = tool_call.name
    tool_args = tool_call.args

    for t in available_tools:
        if t.name == tool_name:
            try:
                result = t.invoke(tool_args)
                print(f"Tool '{tool_name}' executed. Result: {result}")
                return {"tool_output": result, "chat_history": state["chat_history"] + [AIMessage(content=f"Tool '{tool_name}' output: {json.dumps(result)}")]}
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {str(e)}"
                print(error_message)
                return {"tool_output": {"error": error_message}, "chat_history": state["chat_history"] + [AIMessage(content=error_message)]}
    return {"tool_output": {"error": f"Tool '{tool_name}' not found or invalid."}, "chat_history": state["chat_history"] + [AIMessage(content=f"Tool '{tool_name}' not found or invalid.")]}

def observe_node(state: AgentState) -> Dict[str, Any]:
    """Observes the output from tool execution and updates chat history."""
    print("\n--- Entering Observe Node ---")
    # The tool_output is already added to chat_history in execute_tool_node for context.
    # This node primarily acts as a checkpoint before self-correction.
    return state

def self_correct_node(state: AgentState) -> Dict[str, Any]:
    """Analyzes tool output, reflects on the current state, and decides on next steps: re-plan, respond, or escalate."""
    print("\n--- Entering Self-Correction Node ---")
    current_tool_output = state["tool_output"]
    chat_history_so_far = state["chat_history"]

    # Construct a prompt for reflection
    reflection_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are an intelligent self-correction module for a customer support agent. "
            "Your task is to analyze the agent's recent actions and the tool output. "
            "Determine if the task is complete, if a re-plan is needed due to errors/insufficient info, or if escalation is required. "
            "Respond with one of 'REPLAN', 'RESPOND', or 'ESCALATE'. "
            "If REPLAN, also provide a brief reason. If RESPOND, also provide the final response to the customer. "
            "If ESCALATE, provide a reason for escalation and a summary for the human agent."
            f"Current plan: {state.get('plan', 'No specific plan yet')}"
        ),
        *chat_history_so_far,
        HumanMessage(content=f"Most recent tool output: {json.dumps(current_tool_output)}\n\nBased on this, what is the next step? (REPLAN/RESPOND/ESCALATE)")
    ])
    
    reflection_chain = reflection_prompt | llm
    reflection_result = reflection_chain.invoke({"chat_history": chat_history_so_far, "input": f"Most recent tool output: {json.dumps(current_tool_output)}"})
    reflection_text = reflection_result.content.strip()

    print(f"Reflection result: {reflection_text}")

    if reflection_text.startswith("REPLAN"):
        new_reflection = f"REPLAN: {reflection_text.split(':', 1)[1].strip() if ':' in reflection_text else 'Reason unknown.'}"
        return {"reflection": new_reflection, "tool_output": None, "plan": "Re-evaluating strategy and re-planning."}
    elif reflection_text.startswith("RESPOND"):
        final_response = reflection_text.split(':', 1)[1].strip() if ':' in reflection_text else "I have resolved your query." # Fallback
        return {"reflection": "RESPOND: Ready to provide final answer.", "chat_history": state["chat_history"] + [AIMessage(content=final_response)], "tool_output": None}
    elif reflection_text.startswith("ESCALATE"):
        escalation_reason = reflection_text.split(':', 1)[1].strip() if ':' in reflection_text else "Complex issue requiring human intervention."
        return {"escalate": True, "reflection": f"ESCALATE: {escalation_reason}"}
    else:
        # Default to re-plan if reflection is unclear
        return {"reflection": "REPLAN: Unclear reflection, re-evaluating.", "tool_output": None, "plan": "Re-evaluating strategy and re-planning."}

def generate_response_node(state: AgentState) -> Dict[str, Any]:
    """Generates the final response to the customer based on gathered information."""
    print("\n--- Entering Generate Response Node ---")
    # The final response should already be in chat_history from self_correct_node if it decided to RESPOND
    # Or the LLM can generate a new one based on the current state if needed here.
    # For simplicity, we assume self_correct_node puts the final response into chat_history for now.
    # If the last message is an AIMessage, that's our response.
    if state["chat_history"] and isinstance(state["chat_history"][-1], AIMessage):
        return state
    else:
        # Fallback if no specific response from self_correct_node, ask LLM to synthesize
        synthesis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage("Based on the conversation and tool outputs, generate a concise and helpful response for the customer."),
            *state["chat_history"],
        ])
        response_message = llm.invoke(synthesis_prompt)
        return {"chat_history": state["chat_history"] + [AIMessage(content=response_message.content)]}

def escalate_node(state: AgentState) -> Dict[str, Any]:
    """Handles the escalation to a human agent."""
    print("\n--- Entering Escalation Node ---")
    escalation_summary_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are an escalation assistant. Summarize the conversation and the reason for escalation for a human agent. "
            "Include relevant customer details and what was attempted." 
            f"Escalation Reason: {state.get('reflection', 'No specific reason provided.')}"
        ),
        *state["chat_history"],
    ])
    escalation_summary = llm.invoke(escalation_summary_prompt).content
    print(f"Escalation to human agent. Summary: {escalation_summary}")
    # In a real system, this would trigger an alert, create a ticket, etc.
    return {"chat_history": state["chat_history"] + [AIMessage(content=f"I need to escalate this to a human agent. Here's a summary: {escalation_summary}")]}


# --- 5. LangGraph Graph Definition ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("plan", plan_node)
workflow.add_node("execute_tool", execute_tool_node)
workflow.add_node("observe", observe_node)
workflow.add_node("self_correct", self_correct_node)
workflow.add_node("generate_response", generate_response_node)
workflow.add_node("escalate", escalate_node)

# Set entry point
workflow.set_entry_point("plan")

# Define edges (transitions)
workflow.add_edge("plan", "execute_tool") # Always execute a tool if planned
workflow.add_edge("execute_tool", "observe")
workflow.add_edge("observe", "self_correct")

# Conditional edges from self_correct
workflow.add_conditional_edges(
    "self_correct",
    lambda state: "escalate" if state["escalate"] else ("plan" if state["reflection"].startswith("REPLAN") else "generate_response"),
    {
        "plan": "plan",
        "generate_response": "generate_response",
        "escalate": "escalate",
    },
)

# From generate_response, we are done
workflow.add_edge("generate_response", END)
workflow.add_edge("escalate", END) # End of conversation for the agent, human takes over

# Compile the graph
app = workflow.compile()

# --- 6. User Interface (Gradio) ---

def chat_interface(message, history):
    global app # Use the compiled graph

    # Convert Gradio history to Langchain ChatHistory format
    chat_history = []
    for human_msg, ai_msg in history:
        chat_history.append(HumanMessage(content=human_msg))
        chat_history.append(AIMessage(content=ai_msg))
    
    # Invoke the LangGraph agent
    # Initial state for the graph run
    initial_state = {
        "input": message,
        "chat_history": chat_history,
        "tool_output": None,
        "reflection": "",
        "plan": "",
        "escalate": False
    }

    # LangGraph returns a list of states. We want the final one.
    final_state = None
    for s in app.stream(initial_state):
        final_state = s
    
    if final_state and isinstance(final_state, dict):
        # Get the latest AI message from the final chat_history
        if final_state["chat_history"] and isinstance(final_state["chat_history"][-1], AIMessage):
            response = final_state["chat_history"][-1].content
        else:
            response = "An unexpected error occurred or no response was generated."
        return response
    else:
        return "Error: Agent did not return a valid state."


iface = gr.ChatInterface(
    chat_interface,
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(placeholder="Ask me a question about your order, customer info, or general queries.", container=False, scale=7),
    title="Intelligent Customer Support Agent (LangGraph)",
    description="I can assist with customer inquiries using various tools and self-correction. Try asking about order status (e.g., 'What is the status of order ord789?'), customer info (e.g., 'Tell me about customer cust123.'), or general queries (e.g., 'What is your return policy?').",
    theme="soft"
)

# --- 7. FastAPI Backend (Placeholder) ---

fastapi_app = FastAPI(title="Customer Support Agent API")

class ChatRequest(BaseModel):
    message: str
    history: List[List[str]] = [] # [[user_msg, bot_msg], ...]

@fastapi_app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # This endpoint would ideally use the same LangGraph logic as the Gradio interface
    # For brevity, it's a placeholder showing how to expose it.
    # In a real scenario, you'd instantiate and run the `app` graph here.
    
    # Convert history for LangGraph
    chat_history_for_agent = []
    for human_msg, ai_msg in request.history:
        chat_history_for_agent.append(HumanMessage(content=human_msg))
        chat_history_for_agent.append(AIMessage(content=ai_msg))

    initial_state = {
        "input": request.message,
        "chat_history": chat_history_for_agent,
        "tool_output": None,
        "reflection": "",
        "plan": "",
        "escalate": False
    }

    final_state = None
    for s in app.stream(initial_state):
        final_state = s
    
    if final_state and isinstance(final_state, dict) and final_state["chat_history"]:
        response_content = final_state["chat_history"][-1].content
        return {"response": response_content}
    
    return {"response": "Error processing your request.", "status": "failed"}

# To run the Gradio app:
# if __name__ == "__main__":
#     print("Starting Gradio interface...")
#     iface.launch()

# To run the FastAPI app:
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting FastAPI server on http://127.0.0.1:8000")
#     uvicorn.run(fastapi_app, host="127.0.0.1", port=8000)

# Combined main for easier testing/demonstration. 
# You'd typically run Gradio or FastAPI separately.
if __name__ == "__main__":
    print("To run the Gradio UI, uncomment `iface.launch()`.")
    print("To run the FastAPI server, uncomment `uvicorn.run(...)` and install uvicorn.")
    print("Currently, no UI/API will start automatically. You can test the agent interactively via code.")

    # Example interactive test:
    # print("\n--- Interactive Agent Test ---")
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() == 'exit':
    #         break
    #     
    #     # For simple interactive testing, reset history each time or manage it manually
    #     # For a full conversation, pass the updated history in each turn.
    #     initial_state_test = {
    #         "input": user_input,
    #         "chat_history": [], 
    #         "tool_output": None,
    #         "reflection": "",
    #         "plan": "",
    #         "escalate": False
    #     }
    #     
    #     print("Agent thinking...")
    #     for s in app.stream(initial_state_test):
    #         pass # Just run through the states
    #     
    #     final_response_test = "Error or no response."
    #     if s and isinstance(s, dict) and s.get("chat_history"):
    #         final_response_test = s["chat_history"][-1].content
    #     print(f"Agent: {final_response_test}")




