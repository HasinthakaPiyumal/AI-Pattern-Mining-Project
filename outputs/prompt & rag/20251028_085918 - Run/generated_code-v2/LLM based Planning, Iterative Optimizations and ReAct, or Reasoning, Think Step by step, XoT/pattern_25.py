import os
from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory


os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Simulated Tools
knowledge_base = {
    "shipping_policy": "Our standard shipping takes 5-7 business days. Expedited options are available.",
    "return_policy": "You can return items within 30 days of purchase with a valid receipt.",
    "account_reset": "To reset your account password, please visit our website and click 'Forgot Password'.",
    "common_issue_1": "Try restarting your device for connectivity problems."
}

def knowledge_base_search(query: str) -> str:
    for key, value in knowledge_base.items():
        if query.lower() in key.lower() or query.lower() in value.lower():
            return f"Found information: {value}"
    return "No relevant information found in the knowledge base."

support_tickets = []
def create_support_ticket(issue_description: str, customer_name: str = "Unknown") -> str:
    ticket_id = len(support_tickets) + 1
    ticket = {"id": ticket_id, "description": issue_description, "customer": customer_name, "status": "Open"}
    support_tickets.append(ticket)
    return f"Support ticket #{ticket_id} created successfully for {customer_name}. Issue: {issue_description}"

product_data = {
    "P100": {"name": "Wireless Headphones", "price": "$99.99", "warranty": "1 year"},
    "P101": {"name": "Smartwatch", "price": "$199.99", "warranty": "2 years"}
}

def product_information(product_id: str) -> str:
    product = product_data.get(product_id.upper())
    if product:
        return f"Product {product_id}: Name - {product['name']}, Price - {product['price']}, Warranty - {product['warranty']}"
    return f"Product with ID {product_id} not found."

# Initialize LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Create LangChain Tools
tools = [
    Tool(
        name="KnowledgeBaseSearch",
        func=knowledge_base_search,
        description="Useful for searching the internal knowledge base for articles, FAQs, or troubleshooting steps."
    ),
    Tool(
        name="CreateSupportTicket",
        func=create_support_ticket,
        description="Useful for creating a new support ticket in the CRM system when an issue cannot be resolved directly. Input should be a detailed issue description and optionally the customer's name."
    ),
    Tool(
        name="ProductInformation",
        func=product_information,
        description="Useful for retrieving details about a specific product, given its product ID."
    )
]

# Initialize Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the ReAct Agent
agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True # To handle potential parsing errors gracefully
)

print("Intelligent Customer Support Agent initialized. Type 'quit' to exit.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'quit':
        break
    try:
        response = agent_executor.run(input=user_input)
        print(f"Agent: {response}")
    except Exception as e:
        print(f"Agent Error: {e}")
        print("Agent: I encountered an error while processing your request. Please try again or rephrase.")
