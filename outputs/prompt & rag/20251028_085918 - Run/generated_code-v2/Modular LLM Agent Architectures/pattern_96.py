from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langgraph.graph import StateGraph, END
import uvicorn


class WorkingMemoryState(BaseModel):
    q: Optional[str] = None  # User query
    e: List[str] = Field(default_factory=list)  # External evidence
    o: Optional[str] = None  # LLM-generated candidate responses/observations
    u: Optional[float] = None  # Utility scores/relevance
    f: Optional[str] = None  # Verbalized feedback/internal monologue
    h: List[Dict[str, str]] = Field(default_factory=list)  # Complete dialog history
    response: Optional[str] = None  # Final response to user
    nlu_intent: Optional[str] = None
    nlu_entities: Dict[str, str] = Field(default_factory=dict)


def nlu_module(state: WorkingMemoryState) -> Dict:
    print(f"[NLU Module] Processing query: {state.q}")
    # Simulate NLU processing
    intent = "query_info"
    entities = {}
    if state.q and "product" in state.q.lower():
        entities["topic"] = "product_information"
    elif state.q and "return" in state.q.lower():
        entities["topic"] = "return_policy"

    history_entry = {"role": "user", "content": state.q}
    return {"nlu_intent": intent, "nlu_entities": entities, "h": state.h + [history_entry]}


def knowledge_retrieval_module(state: WorkingMemoryState) -> Dict:
    print(f"[Knowledge Retrieval Module] Retrieving knowledge for intent: {state.nlu_intent} and entities: {state.nlu_entities}")
    evidence = []
    if state.nlu_entities.get("topic") == "product_information":
        evidence.append("Our latest product features a long-lasting battery and a high-resolution display.")
        evidence.append("You can find detailed specifications on our product page.")
    elif state.nlu_entities.get("topic") == "return_policy":
        evidence.append("Our return policy allows returns within 30 days of purchase with a valid receipt.")
        evidence.append("Please ensure the item is in its original condition.")
    else:
        evidence.append("We are unable to find specific knowledge for this query, forwarding to general LLM.")

    return {"e": evidence}


def llm_interaction_module(state: WorkingMemoryState) -> Dict:
    print(f"[LLM Interaction Module] Generating response with query: {state.q}, evidence: {state.e}, history: {state.h}")
    # Simulate LLM call
    llm_prompt = f"User query: {state.q}\n"
    if state.e:
        llm_prompt += f"External evidence: {' '.join(state.e)}\n"
    llm_prompt += f"Dialog history: {state.h}\n"
    llm_prompt += "Based on the above, provide a helpful and concise response.\n"

    # Mock LLM response
    if state.nlu_entities.get("topic") == "product_information" and state.e:
        observation = f"The product has a long-lasting battery and high-resolution display. More details are on the product page."
    elif state.nlu_entities.get("topic") == "return_policy" and state.e:
        observation = f"You can return items within 30 days with a receipt, provided they are in original condition."
    elif state.q:
        observation = f"I'm sorry, I don't have enough specific information to answer that. Can you please provide more details?"
    else:
        observation = "Hello! How can I assist you today?"

    utility_score = 0.8 # Placeholder
    feedback = "LLM provided a relevant answer based on evidence." # Placeholder

    return {"o": observation, "u": utility_score, "f": feedback}


def generate_response_module(state: WorkingMemoryState) -> Dict:
    print(f"[Response Generation Module] Finalizing response from observation: {state.o}")
    final_response = state.o if state.o else "I'm sorry, I could not generate a response at this time."
    history_entry = {"role": "assistant", "content": final_response}
    return {"response": final_response, "h": state.h + [history_entry]}


def should_retrieve_knowledge(state: WorkingMemoryState) -> str:
    if state.nlu_intent == "query_info" and state.nlu_entities.get("topic"):
        return "retrieve_knowledge"
    return "llm_interact"


# Define the LangGraph workflow
workflow = StateGraph(WorkingMemoryState)

# Add nodes
workflow.add_node("nlu", nlu_module)
workflow.add_node("retrieve_knowledge", knowledge_retrieval_module)
workflow.add_node("llm_interact", llm_interaction_module)
workflow.add_node("generate_response", generate_response_module)

# Set entry point
workflow.set_entry_point("nlu")

# Add edges
workflow.add_conditional_edges(
    "nlu",
    should_retrieve_knowledge,
    {
        "retrieve_knowledge": "retrieve_knowledge",
        "llm_interact": "llm_interact",
    },
)
workflow.add_edge("retrieve_knowledge", "llm_interact")
workflow.add_edge("llm_interact", "generate_response")
workflow.add_edge("generate_response", END)

# Compile the graph
app_graph = workflow.compile()


# FastAPI Application
api_app = FastAPI(title="Intelligent Customer Support Agent")


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    history: List[Dict[str, str]]


@api_app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    initial_state = WorkingMemoryState(q=request.query)
    final_state = app_graph.invoke(initial_state)
    return ChatResponse(response=final_state.response, history=final_state.h)


if __name__ == "__main__":
    # Example usage for direct testing of the graph (optional)
    # print("\n--- Running graph directly ---")
    # initial_query = WorkingMemoryState(q="What are the features of your new product?")
    # final_graph_state = app_graph.invoke(initial_query)
    # print(f"Final Graph State: {final_graph_state.dict()}")
    # print(f"Agent Response: {final_graph_state.response}")

    uvicorn.run(api_app, host="0.0.0.0", port=8000)
