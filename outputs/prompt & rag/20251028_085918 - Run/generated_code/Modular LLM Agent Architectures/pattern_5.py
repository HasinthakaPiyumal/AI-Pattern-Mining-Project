import os
from typing import List, Tuple, Annotated, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langgraph.graph import StateGraph, END

# --- 1. Environment Setup ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 2. Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"

class ChatResponse(BaseModel):
    response: str
    session_id: str

# --- 3. Tools (Simulated External APIs) ---
@tool
def check_order_status(order_id: str) -> str:
    """Checks the status of a customer's order using an order ID."""
    if order_id == "ORDER123":
        return "Order123 is currently in transit and expected to arrive by 2024-12-25."
    elif order_id == "ORDER456":
        return "Order456 was delivered on 2024-11-01. Please check your delivery confirmation."
    return f"Could not find order status for ID: {order_id}."

@tool
def lookup_product_info(product_name: str) -> str:
    """Looks up detailed information about a product by its name."""
    if "laptop" in product_name.lower():
        return "Our 'ProBook X' laptop features an i7 processor, 16GB RAM, and a 512GB SSD. It comes with a 1-year warranty."
    elif "headphone" in product_name.lower():
        return "The 'SoundWave' headphones offer noise cancellation and 20 hours of battery life. Available in black and white."
    return f"No detailed information found for product: {product_name}."

@tool
def update_account_details(customer_id: str, new_details: str) -> str:
    """Updates customer account details. Requires customer ID and the new details to be updated."""
    return f"Successfully updated account details for customer {customer_id} with: {new_details}."

all_tools = [check_order_status, lookup_product_info, update_account_details]

# --- 4. Vector Store (Chroma) and RAG Module ---
# In a real application, this would be populated from a larger corpus.
knowledge_base_docs = [
    "Our return policy allows returns within 30 days of purchase for a full refund.",
    "To reset your password, visit our website and click 'Forgot Password'.",
    "Shipping usually takes 3-5 business days within the continental US.",
    "For technical support, please call our hotline at 1-800-TECH-HELP."
]

embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma.from_texts(texts=knowledge_base_docs, embedding=embedding_function)
retriever = vectorstore.as_retriever()

# --- 5. LLM and Chat History ---
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# History-aware retriever for RAG
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# RAG chain for answering questions based on retrieved documents
qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Keep the answer concise and helpful.\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# --- 6. LangGraph State (Agentic Working Memory) ---
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    agent_outcome: Annotated[List[Tuple[str, str]], dict]
    intermediate_steps: Annotated[List[Tuple[str, str]], dict]
    tool_output: str
    next_step: str

# --- 7. Nodes for LangGraph ---
# Main LLM for general conversation and planning
agent_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI customer support agent. You can answer questions, look up product info, check order statuses, and update account details. Be helpful and professional. If a tool is needed, respond with a tool call. Otherwise, respond directly. Manage complex tasks by breaking them down."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
agent = create_react_agent(llm, all_tools, agent_prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=False)

def call_llm_agent(state: AgentState):
    """General LLM interaction, including planning and tool invocation decisions."""
    print("---NODE: call_llm_agent---")
    result = agent_executor.invoke({"input": state["input"], "chat_history": state["chat_history"]})
    if "output" in result:
        # Direct response from LLM
        return {"agent_outcome": result["output"], "next_step": "direct_response"}
    elif "actions" in result and result["actions"]:
        # LLM decided to use a tool
        tool_calls = []
        for action in result["actions"]:
            tool_calls.append((action.tool, action.tool_input))
        return {"intermediate_steps": result["intermediate_steps"], "next_step": "call_tool"}
    else:
        return {"agent_outcome": "I'm not sure how to proceed with that request.", "next_step": "direct_response"}

def call_tool(state: AgentState):
    """Executes a tool call based on agent's decision."""
    print("---NODE: call_tool---")
    # Assuming the tool call is extracted from intermediate_steps or a direct tool call state
    # For simplicity, we'll assume the last action in intermediate_steps is the tool to call
    if state["intermediate_steps"]:
        last_action = state["intermediate_steps"][-1]
        tool_name = last_action.tool
        tool_args = last_action.tool_input

        try:
            # Dynamically call the tool function
            tool_func = next(t for t in all_tools if t.name == tool_name)
            output = tool_func.invoke(tool_args)
            return {"tool_output": output, "next_step": "process_tool_output"}
        except Exception as e:
            return {"tool_output": f"Error executing tool {tool_name}: {e}", "next_step": "process_tool_output"}
    return {"tool_output": "No tool to call.", "next_step": "process_tool_output"}

def retrieve_knowledge(state: AgentState):
    """Retrieves information from the RAG knowledge base."""
    print("---NODE: retrieve_knowledge---")
    response = rag_chain.invoke({"input": state["input"], "chat_history": state["chat_history"]})
    return {"agent_outcome": response["answer"], "next_step": "direct_response"}

def router_decision(state: AgentState):
    """Decides the next step based on input and current state (Cognitive Load Management & Router)."""
    print("---NODE: router_decision---")
    user_input = state["input"].lower()
    
    # Check for keywords indicating a need for tools or RAG
    if any(keyword in user_input for keyword in ["order status", "track my order"]):
        return "call_llm_agent" # Agent decides if tool is needed, or asks for ID
    elif any(keyword in user_input for keyword in ["product info", "features of", "details about"]):
        return "call_llm_agent" # Agent decides if tool is needed, or asks for product name
    elif any(keyword in user_input for keyword in ["update account", "change my details"]):
        return "call_llm_agent"
    elif any(keyword in user_input for keyword in ["return policy", "refund", "shipping", "password reset", "technical support"]):
        return "retrieve_knowledge"
    else:
        return "call_llm_agent" # Default to general LLM if no specific pattern found

def final_response_node(state: AgentState):
    """Forms the final response to the user and updates chat history."""
    print("---NODE: final_response_node---")
    final_output = ""
    if state.get("tool_output"):
        final_output = f"Tool output: {state['tool_output']}"
        # After tool execution, re-invoke LLM to process tool output and generate user-friendly response
        follow_up_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("human", f"I just executed a tool and got this output: {state['tool_output']}. Based on the previous conversation and this output, please provide a helpful and concise response to the user."),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        follow_up_agent = create_react_agent(llm, all_tools, follow_up_prompt)
        follow_up_executor = AgentExecutor(agent=follow_up_agent, tools=all_tools, verbose=False)
        result = follow_up_executor.invoke({"input": state["input"], "chat_history": state["chat_history"] + [HumanMessage(content=f"Tool output: {state['tool_output']}")]})
        final_output = result.get("output", "I have processed the information.")
    elif state.get("agent_outcome"):
        final_output = state["agent_outcome"]

    # Update chat history for the next turn (Agentic Working Memory)
    # Note: In a real system, chat_history would be managed per session_id and loaded/saved.
    # For this example, we'll append to a global list for demonstration.
    return {"agent_outcome": final_output}


# --- 8. Edges and Graph Definition ---
workflow = StateGraph(AgentState)

workflow.add_node("router_decision", router_decision)
workflow.add_node("call_llm_agent", call_llm_agent)
workflow.add_node("call_tool", call_tool)
workflow.add_node("retrieve_knowledge", retrieve_knowledge)
workflow.add_node("final_response_node", final_response_node)

workflow.set_entry_point("router_decision")

workflow.add_conditional_edges(
    "router_decision",
    router_decision,
    {
        "call_llm_agent": "call_llm_agent",
        "retrieve_knowledge": "retrieve_knowledge",
    },
)

workflow.add_conditional_edges(
    "call_llm_agent",
    lambda state: state["next_step"],
    {
        "call_tool": "call_tool",
        "direct_response": "final_response_node",
        "process_tool_output": "final_response_node" # If LLM directly provides an answer after internal thought
    },
)

workflow.add_edge("call_tool", "final_response_node")
workflow.add_edge("retrieve_knowledge", "final_response_node")
workflow.add_edge("final_response_node", END)

app_agent = workflow.compile()

# --- 9. FastAPI Application ---
app = FastAPI(title="AI Customer Support Agent")

# In a real application, this would be a database or a more persistent store
session_histories = {}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global session_histories
    session_id = request.session_id
    current_history = session_histories.get(session_id, [])

    print(f"\n--- User Input for Session {session_id}: {request.message} ---")

    # Prepare initial state for LangGraph
    inputs = {"input": request.message, "chat_history": current_history, "agent_outcome": None, "intermediate_steps": [], "tool_output": None, "next_step": None}

    # Execute the graph
    # The graph will update `chat_history` implicitly within its nodes for this turn
    # For persistent memory across turns, we manually update after each invocation.
    final_state = app_agent.invoke(inputs)

    agent_response_content = final_state.get("agent_outcome", "I apologize, I encountered an issue processing your request.")

    # Update chat history for the session (Agentic Working Memory persistence)
    current_history.append(HumanMessage(content=request.message))
    current_history.append(AIMessage(content=agent_response_content))
    session_histories[session_id] = current_history

    print(f"--- Agent Response for Session {session_id}: {agent_response_content} ---")
    
    return ChatResponse(response=agent_response_content, session_id=session_id)

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    global session_histories
    history = session_histories.get(session_id, [])
    formatted_history = []
    for msg in history:
        formatted_history.append({"type": msg.type, "content": msg.content})
    return {"session_id": session_id, "history": formatted_history}


# To run this application:
# 1. Save the code as `main.py`
# 2. Install dependencies: `pip install fastapi "uvicorn[standard]" python-dotenv langchain langchain-openai langchain-community langgraph pydantic chromadb`
# 3. Create a `.env` file in the same directory with `OPENAI_API_KEY="your_openai_api_key"`
# 4. Run: `uvicorn main:app --reload`
# 5. Access the API at http://127.0.0.1:8000/docs
