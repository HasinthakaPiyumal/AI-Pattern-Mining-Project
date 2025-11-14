import os
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.prompts import PromptTemplate

# Set your OpenAI API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# --- Mock Tools Implementation ---

def knowledge_base_lookup(query: str) -> str:
    """Simulates searching an internal knowledge base for FAQs, troubleshooting guides, or product information."""
    print(f"\n[TOOL CALL] Knowledge Base Lookup for: '{query}'")
    if "faulty product" in query.lower() or "troubleshoot" in query.lower():
        return "Found troubleshooting steps for common product issues: 1. Restart device. 2. Check connections. 3. Update firmware. If issues persist, consider a return."
    elif "return policy" in query.lower():
        return "Our return policy allows returns within 30 days of purchase with original receipt. Please ensure the item is in its original packaging."
    elif "shipping status" in query.lower():
        return "For shipping status, please use the 'External API Tool' with the order ID."
    return "No direct answer found in knowledge base for that query. Please try rephrasing or consider other tools."

def crm_system_action(action: str) -> str:
    """Simulates interacting with a CRM system to retrieve customer history, update tickets, or initiate actions like returns/exchanges. Input should describe the action, e.g., 'initiate return for order 12345' or 'lookup customer history for John Doe'."""
    print(f"\n[TOOL CALL] CRM System Action: '{action}'")
    if "initiate return for order" in action.lower():
        order_id = action.split("order ")[-1].strip()
        return f"Return process initiated for order {order_id}. A return label has been sent to the customer's email."
    elif "lookup customer history for" in action.lower():
        customer_name = action.split("for ")[-1].strip()
        return f"Retrieved customer history for {customer_name}: Last purchase on 2023-10-26, previously contacted about billing inquiry on 2023-09-15."
    elif "create ticket" in action.lower():
        return f"New support ticket created for the current issue."
    return f"CRM action '{action}' completed with no specific details to return or action failed."

def external_api_call(query: str) -> str:
    """Simulates calling external APIs for specific data, e.g., order status, shipping information. Input should describe the query, e.g., 'order status for 12345'."""
    print(f"\n[TOOL CALL] External API Call for: '{query}'")
    if "order status for" in query.lower():
        order_id = query.split("for ")[-1].strip()
        if order_id == "12345":
            return "Order 12345 status: Shipped, estimated delivery 2 days. Tracking: TRK123456789"
        elif order_id == "67890":
            return "Order 67890 status: Processing."
        return f"Order {order_id} not found."
    return f"External API could not process query: '{query}'. Please ensure correct format."

# --- Langchain Tools Setup ---

tools = [
    Tool(
        name="KnowledgeBaseLookup",
        func=knowledge_base_lookup,
        description="Useful for finding information in the internal knowledge base, FAQs, and troubleshooting guides. Input should be a question or keywords."
    ),
    Tool(
        name="CRMSystemAction",
        func=crm_system_action,
        description="Useful for interacting with the CRM system to manage customer data, tickets, returns, or exchanges. Input should be a clear action description, e.g., 'initiate return for order 12345' or 'lookup customer history for John Doe'."
    ),
    Tool(
        name="ExternalAPICall",
        func=external_api_call,
        description="Useful for querying external APIs for real-time data like order status or shipping details. Input should be a specific query, e.g., 'order status for 12345'."
    ),
]

# --- LLM and Agent Setup ---

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Use Langchain Hub for a suitable ReAct style prompt
# The prompt will guide the agent to think, use tools, process feedback, and refine its approach.
# We'll adapt a standard ReAct prompt to include self-refinement instructions.

# Fetch the ReAct prompt from Langchain Hub
base_prompt = hub.pull("hwchase17/react")

# Customizing the prompt to include self-refinement and adaptive behavior
custom_prompt_template = """You are an Adaptive Agentic Customer Support AI. Your goal is to resolve complex customer inquiries by dynamically planning, using integrated tools, and continuously refining your understanding based on tool execution results and simulated user feedback. 

**Self-Refinement Strategy:**
1.  **Analyze Request:** Understand the core problem and identify potential sub-tasks. 
2.  **Plan:** Formulate a step-by-step plan to address the request, considering which tools might be useful. 
3.  **Execute Tool:** Use the most appropriate tool(s) for the current step. 
4.  **Process Feedback:** Evaluate the tool's output. If the output is not satisfactory or doesn't fully resolve the issue, adapt your plan. Consider if a different tool is needed, if the query should be rephrased, or if the problem needs escalation.
5.  **Iterate/Refine:** Based on feedback, update your internal understanding, self-correct any errors in reasoning, and adjust subsequent actions. Continuously strive for the best resolution. If the user expresses dissatisfaction, re-evaluate and try a different approach.
6.  **Conclude:** Once the issue is resolved to the best of your ability, provide a clear summary and ask if anything else is needed. If you cannot resolve it, suggest escalation to a human agent with a concise summary of your attempts.

Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

# Create a PromptTemplate instance
custom_prompt = PromptTemplate.from_template(custom_prompt_template)

# Combine the custom prompt with the base prompt structure
# We need to explicitly pass tools and tool_names to the prompt's partial variables
final_prompt = custom_prompt.partial(tools=base_prompt.partial_variables["tools"], tool_names=base_prompt.partial_variables["tool_names"])


# Create the ReAct agent
agent = create_react_agent(llm, tools, final_prompt)

# Create the AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Example Interactions ---

def run_customer_interaction(query: str):
    print(f"\n--- Customer Query: {query} ---")
    try:
        response = agent_executor.invoke({"input": query})
        print(f"\n--- Agent Final Response ---")
        print(response["output"])
    except Exception as e:
        print(f"\n--- An error occurred during interaction: {e} ---")

if __name__ == "__main__":
    # Scenario 1: Troubleshooting a faulty product
    run_customer_interaction("My new laptop is not turning on. What should I do?")

    # Scenario 2: Initiating a return and then checking the status of another order
    run_customer_interaction("I need to return my recent purchase, order number 12345. Also, what is the status of order 67890?")

    # Scenario 3: Query that requires initial knowledge base lookup, then potentially escalation or further action
    run_customer_interaction("I'm having trouble understanding my bill. Can you help me?")

    # Scenario 4: A more complex scenario where initial tool use might lead to dissatisfaction and refinement
    print("\n--- Simulating a complex scenario with potential refinement ---")
    # Initial attempt to resolve
    run_customer_interaction("My internet speed is very slow, I've already restarted my router, and it didn't help. What next?")
    print("\n--- (Simulated User Feedback: That didn't solve my problem. I need more advanced help.) ---")
    # Agent should ideally 'refine' its approach here if it were truly interactive, but for this single run, 
    # we simulate the refinement by a subsequent, more direct query or a prompt that implies dissatisfaction.
    run_customer_interaction("The previous steps didn't work. Can you check my account details and escalate this to a technical specialist?")
