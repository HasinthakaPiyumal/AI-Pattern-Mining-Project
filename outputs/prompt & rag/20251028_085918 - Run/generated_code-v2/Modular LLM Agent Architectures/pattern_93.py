import os
import json
from typing import TypedDict, Annotated, List, Union, Callable
import operator

# Langchain and Langgraph imports
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation

# --- 1. Environment Setup (API keys) ---
# Please set your OPENAI_API_KEY environment variable.
# Example: export OPENAI_API_KEY="your_api_key_here"
# If not set, dummy responses will be used for tools.

# --- 2. Define Tools ---
@tool
def search_flights(destination: str, departure_date: str, return_date: str) -> str:
    """Searches for flights for a given destination and dates.
    Returns a JSON string with flight options.
    Example: search_flights(destination="Paris", departure_date="2024-07-15", return_date="2024-07-20")
    """
    print(f"DEBUG: Searching flights for {destination} from {departure_date} to {return_date}")
    if "paris" in destination.lower() and "2024-07-15" in departure_date:
        return json.dumps({"flights": [{"airline": "AirFrance", "price": "€500", "details": "Non-stop"}, {"airline": "Lufthansa", "price": "€480", "details": "1 stop"}]})
    return json.dumps({"flights": [{"airline": "Dummy Air", "price": "$300", "details": "Direct"}, {"airline": "Budget Fly", "price": "$250", "details": "1 stop"}]})

@tool
def search_hotels(destination: str, check_in_date: str, check_out_date: str, num_guests: int) -> str:
    """Searches for hotels in a given destination for specific dates and number of guests.
    Returns a JSON string with hotel options.
    Example: search_hotels(destination="Paris", check_in_date="2024-07-15", check_out_date="2024-07-18", num_guests=2)
    """
    print(f"DEBUG: Searching hotels in {destination} from {check_in_date} to {check_out_date} for {num_guests} guests")
    if "paris" in destination.lower() and "2024-07-15" in check_in_date:
        return json.dumps({"hotels": [{"name": "Hotel de Ville", "price_per_night": "€150", "stars": 4}, {"name": "Budget Stay", "price_per_night": "€80", "stars": 2}]})
    return json.dumps({"hotels": [{"name": "Grand Hotel", "price_per_night": "$120", "stars": 5}, {"name": "Comfort Inn", "price_per_night": "$70", "stars": 3}]})

@tool
def find_attractions(destination: str, date: str = None) -> str:
    """Finds popular attractions or events in a given destination on a specific date (optional).
    Returns a JSON string with attraction details.
    Example: find_attractions(destination="Paris", date="2024-07-16")
    """
    print(f"DEBUG: Finding attractions in {destination} on {date}")
    if "paris" in destination.lower():
        return json.dumps({"attractions": [{"name": "Eiffel Tower", "type": "landmark"}, {"name": "Louvre Museum", "type": "museum"}]})
    return json.dumps({"attractions": [{"name": "City Park", "type": "park"}, {"name": "Local Market", "type": "shopping"}]})

@tool
def get_weather(location: str, date: str) -> str:
    """Gets the weather forecast for a specified location and date.
    Returns a JSON string with weather information.
    Example: get_weather(location="Paris", date="2024-07-15")
    """
    print(f"DEBUG: Getting weather for {location} on {date}")
    if "paris" in location.lower() and "2024-07-15" in date:
        return json.dumps({"weather": {"condition": "Sunny", "temperature": "25C"}})
    return json.dumps({"weather": {"condition": "Cloudy", "temperature": "20C"}})

# List of all tools available to the agent
tools = [search_flights, search_hotels, find_attractions, get_weather]
tool_executor = ToolExecutor(tools)


# --- 3. Define LLM ---
# Ensure OPENAI_API_KEY is set in your environment variables.
llm = ChatOpenAI(model="gpt-4o", temperature=0) # Using gpt-4o for better tool use

# --- 4. Define Agent State (for langgraph) ---
class AgentState(TypedDict):
    """
    Represents the state of our agent in the LangGraph workflow.

    - `messages`: A list of messages detailing the conversation history.
    - `user_preferences`: A string or dictionary to store extracted user preferences.
    - `current_itinerary`: A string representing the evolving travel plan.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    user_preferences: str
    current_itinerary: str

# --- 5. Define Graph Nodes ---
def call_llm(state: AgentState) -> dict:
    """
    Node for invoking the LLM to decide the next action (tool use or final answer).
    The LLM considers the conversation history, user preferences, and available tools.
    """
    messages = state["messages"]
    system_message_content = (
        "You are a helpful AI travel planner agent. Your goal is to create personalized travel itineraries "
        "based on user preferences. You can search for flights, hotels, attractions, and weather. "
        "Respond with a comprehensive itinerary once you have gathered enough information and made decisions. "
        "Always use the tools available to gather necessary information before making recommendations." 
        "Be concise and directly address the user's request. If you need more information, ask for it." 
        f"Current user preferences: {state.get('user_preferences', 'None specified.')}"
    )
    
    # Prepend the system message to the current messages list for the LLM context
    full_messages = [SystemMessage(content=system_message_content)] + messages
    
    llm_tools = [convert_to_openai_tool(t) for t in tools]
    llm_with_tools = llm.bind_tools(llm_tools)
    response = llm_with_tools.invoke(full_messages)
    
    return {"messages": [response]}

def call_tool(state: AgentState) -> dict:
    """
    Node for executing tools identified by the LLM in its previous response.
    It extracts tool calls, executes them using the ToolExecutor, and adds the results to the state.
    """
    last_message = state["messages"][-1]
    
    # Extract tool calls from the LLM's response
    tool_calls = last_message.tool_calls
    
    # List to store results of tool calls
    tool_results = []
    for tool_call in tool_calls:
        # Construct ToolInvocation from the tool_call for the ToolExecutor
        tool_invocation = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        print(f"DEBUG: Executing tool: {tool_invocation.tool} with input {tool_invocation.tool_input}")
        try:
            output = tool_executor.invoke(tool_invocation)
            print(f"DEBUG: Tool output: {output}")
            tool_results.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
        except Exception as e:
            error_message = f"Error executing tool {tool_invocation.tool}: {e}"
            print(f"ERROR: {error_message}")
            tool_results.append(ToolMessage(content=error_message, tool_call_id=tool_call["id"]))
    
    # Add tool results to the messages, which will be passed back to the LLM
    return {"messages": tool_results}

# --- 6. Define Graph Logic ---
def should_continue(state: AgentState) -> str:
    """
    Determines if the agent should continue by calling tools or end the conversation.
    It checks the last message from the LLM for any tool calls.
    """
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        # If there are no tool calls, it means the LLM provided a final answer or completed its thought process.
        return "end"
    else:
        # Otherwise, the LLM wants to use tools, so continue the cycle.
        return "continue"

# --- 7. Define Graph ---
# Initialize the StateGraph with our AgentState
workflow = StateGraph(AgentState)

# Add the nodes to the workflow, mapping names to functions
workflow.add_node("llm", call_llm)
workflow.add_node("tool", call_tool)

# Set the entry point for the graph - the process always starts with the LLM
workflow.set_entry_point("llm")

# Define conditional edges: from the 'llm' node, decide whether to call tools or end
workflow.add_conditional_edges(
    "llm", # From LLM node
    should_continue, # Use the function to decide the next step
    {
        "continue": "tool", # If 'should_continue' returns "continue", go to the 'tool' node
        "end": END           # If 'should_continue' returns "end", terminate the graph
    }
)

# Define a regular edge: after executing tools, always go back to the LLM for further reasoning/response
workflow.add_edge("tool", "llm") 

# Compile the workflow into a runnable LangGraph application
app = workflow.compile()

# --- 8. Main execution loop (for demonstration) ---
def run_travel_planner(user_query: str):
    """
    Runs the travel planner agent with a given user query and prints the execution steps.
    """
    # Initialize the input state for the agent with the user's query
    inputs = {"messages": [HumanMessage(content=user_query)], "user_preferences": "", "current_itinerary": ""}
    
    print(f"\n--- User Query: {user_query} ---")
    # Stream the execution of the graph to see intermediate steps
    for s in app.stream(inputs):
        if "__end__" not in s:
            print(s)
            # In a real application, you might parse LLM's natural language output
            # to update user_preferences or current_itinerary in the state object
            # For this example, we simply show the raw state updates.
        else:
            final_state = s["__end__"]
            # The final response from the agent will be the last message in the state
            final_message = final_state["messages"][-1]
            print(f"\n--- Final Itinerary/Response ---")
            print(final_message.content)
            print("--- End of planning ---")
            return final_message.content

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("\nWARNING: OPENAI_API_KEY environment variable not set. Using dummy responses for tools.")
        print("For actual LLM interaction and dynamic planning, please set OPENAI_API_KEY.")
    else:
        print("\nOPENAI_API_KEY is set. Agent will use real LLM interaction and tool calls.")

    print("\nWelcome to the Personalized Travel Itinerary Planner Agent!\n")
    print("Type 'exit' to quit at any time (currently, runs predefined examples).\n")

    # Example 1: Simple flight search
    run_travel_planner("I want to find flights to Paris from July 15th to July 20th, 2024.")

    # Example 2: More complex plan requiring multiple tools and steps
    run_travel_planner("Plan a 3-day trip to Paris for me and my partner starting on July 15th, 2024. I like museums and good food. Also check the weather for the first day.")

    # Example 3: Hotel search
    run_travel_planner("Can you find hotels in Paris for two people from July 15th to July 18th, 2024?")

    # Example 4: Attraction search
    run_travel_planner("What are some popular attractions in Paris?")

