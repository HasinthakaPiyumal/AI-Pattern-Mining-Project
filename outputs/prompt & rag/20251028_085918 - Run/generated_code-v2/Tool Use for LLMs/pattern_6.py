import autogen

# --- Simulated Tool Functions ---
def basic_knowledge_base(query: str) -> str:
    if "common issue" in query.lower():
        return "Solution for common issue: Restart your device."
    return "No direct FAQ found for your query in basic knowledge base."

def tech_kb(query: str) -> str:
    if "internet disconnecting" in query.lower():
        return "Technical KB: Check router connection, update drivers, contact ISP if persistent."
    return "No technical solution found in TechKB."

def troubleshooting_tool(issue: str) -> str:
    if "internet connection" in issue.lower():
        return "Running diagnostics for internet connection... Result: Potential DNS issue. Try flushing DNS."
    return "Running general diagnostics... Result: No immediate solution found."

def billing_system_api(action: str, details: dict) -> str:
    if action == "check_charge":
        charge_id = details.get("charge_id", "unknown")
        return f"Billing System: Investigating charge {charge_id}. Found it is a premium subscription renewal."
    elif action == "modify_subscription":
        new_plan = details.get("new_plan")
        return f"Billing System: Subscription updated to {new_plan} plan."
    return "Billing System: Invalid action."

def product_db(query: str) -> str:
    if "product features" in query.lower():
        return "Product DB: Our premium product features include cloud storage and advanced analytics."
    return "No product details found."

def user_manual_lookup(product: str, topic: str) -> str:
    if "internet router" in product.lower() and "setup" in topic.lower():
        return "User Manual: For router setup, refer to page 5 of the quick start guide."
    return "User Manual: No specific topic found."

# --- Agent Configuration ---
llm_config = {
    "config_list": [
        {
            "model": "gpt-4", 
            "api_key": "YOUR_OPENAI_API_KEY" # Replace with your actual OpenAI API key
        }
    ],
    "temperature": 0.7
}

# --- Define Agents ---
user_proxy = autogen.UserProxyAgent(
    name="Customer",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("EXIT"),
    code_execution_config={"last_n_messages": 3, "work_dir": "customer_support_workspace"},
    llm_config=llm_config,
    system_message="A human customer interacting with the support system. Provide the initial query and approve agent actions."
)

triage_agent = autogen.AssistantAgent(
    name="Triage_Agent",
    llm_config=llm_config,
    system_message="You are a Triage Agent. Your role is to understand the customer's problem, identify the relevant domain(s) (technical, billing, product), and route the inquiry to the appropriate specialized agent(s). You can use a basic knowledge base for simple FAQs."
)
triage_agent.register_function(function_map={"basic_knowledge_base": basic_knowledge_base})

technical_support_agent = autogen.AssistantAgent(
    name="Technical_Support_Agent",
    llm_config=llm_config,
    system_message="You are a Technical Support Agent. You handle technical issues, troubleshooting, and provide solutions for product malfunctions or software problems. Use your TechKB and TroubleshootingTool."
)
technical_support_agent.register_function(function_map={
    "tech_kb": tech_kb,
    "troubleshooting_tool": troubleshooting_tool
})

billing_agent = autogen.AssistantAgent(
    name="Billing_Agent",
    llm_config=llm_config,
    system_message="You are a Billing Agent. You address billing inquiries, subscription changes, payment issues, and refund requests. Use the BillingSystemAPI."
)
billing_agent.register_function(function_map={"billing_system_api": billing_system_api})

product_expert_agent = autogen.AssistantAgent(
    name="Product_Expert_Agent",
    llm_config=llm_config,
    system_message="You are a Product Expert Agent. You provide in-depth information about product features, usage, and advanced configurations. Use the ProductDB and UserManualLookup."
)
product_expert_agent.register_function(function_map={
    "product_db": product_db,
    "user_manual_lookup": user_manual_lookup
})

# --- Group Chat Setup ---
groupchat = autogen.GroupChat(
    agents=[user_proxy, triage_agent, technical_support_agent, billing_agent, product_expert_agent],
    messages=[],
    max_round=12
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# --- Initiate Conversation ---
user_proxy.initiate_chat(
    manager,
    message="My internet keeps disconnecting, and I also see an unfamiliar charge on my last bill. Can you help me with both?"
)
