import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
import re

PRODUCT_CATALOG = {
    "P101": {"name": "Laptop Pro", "price": 1200, "description": "High-performance laptop for professionals.", "category": "Electronics", "image_url": "http://example.com/laptop_pro.jpg"},
    "P102": {"name": "Wireless Mouse", "price": 35, "description": "Ergonomic wireless mouse with long battery life.", "category": "Accessories", "image_url": "http://example.com/wireless_mouse.jpg"},
    "P103": {"name": "Mechanical Keyboard", "price": 99, "description": "RGB mechanical keyboard for gamers.", "category": "Accessories", "image_url": "http://example.com/mechanical_keyboard.jpg"},
    "P201": {"name": "Smartwatch X", "price": 250, "description": "Fitness tracking and smart notifications on your wrist.", "category": "Wearables", "image_url": "http://example.com/smartwatch_x.jpg"},
}

USER_PROFILE_DB = {}

def recommend_products(query: str) -> str:
    if "laptop" in query.lower() or "computer" in query.lower():
        return "Based on your interest in laptops, I recommend 'Laptop Pro' (ID: P101). It's a high-performance device for professionals."
    elif "accessory" in query.lower() or "keyboard" in query.lower() or "mouse" in query.lower():
        return "Looking for accessories? Consider the 'Wireless Mouse' (ID: P102) or the 'Mechanical Keyboard' (ID: P103)."
    elif "wearable" in query.lower() or "watch" in query.lower():
        return "For wearables, I suggest the 'Smartwatch X' (ID: P201)."
    return "I can recommend various electronics and accessories. Please be more specific about what you are looking for."

def get_product_details(product_id: str) -> str:
    product_info = PRODUCT_CATALOG.get(product_id)
    if product_info:
        return (
            f"Product Name: {product_info['name']}\n"
            f"Price: ${product_info['price']}\n"
            f"Description: {product_info['description']}\n"
            f"Category: {product_info['category']}\n"
            f"Image URL: {product_info['image_url']}"
        )
    return f"Sorry, I couldn't find details for product ID: {product_id}."

tools = [
    Tool(
        name="RecommendProducts",
        func=recommend_products,
        description="Useful for recommending products based on user query or preferences. Input should be a concise query string."
    ),
    Tool(
        name="GetProductDetails",
        func=get_product_details,
        description="Useful for getting detailed information about a specific product. Input should be a product ID (e.g., 'P101')."
    ),
]

llm = ChatOpenAI(temperature=0)

system_template = """You are an E-commerce Conversational Shopping Assistant.\nYour goal is to help users find products, provide product details, and offer personalized recommendations.\nYou have access to the following tools: {tools}\n\nYou should always:\n1. Try to understand the user's intent.\n2. Use the available tools when necessary to fulfill the user's request.\n3. If the user expresses a preference or gives a fact about themselves, state it clearly for me to remember, for example: "I will remember that you like 'laptops'."\n4. When making recommendations, always provide the product ID so the user can ask for details.\n5. If you need more information to use a tool, ask clarifying questions.\n\nCurrent user ID: {user_id}\nKnown facts about the user: {user_facts}\n"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

agent = create_react_agent(llm, tools, prompt)

store = {}
def get_session_history(session_id: str) -> ConversationBufferWindowMemory:
    if session_id not in store:
        store[session_id] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )
    return store[session_id]

def store_user_fact(user_id: str, fact: str):
    if user_id not in USER_PROFILE_DB:
        USER_PROFILE_DB[user_id] = []
    if fact not in USER_PROFILE_DB[user_id]:
        USER_PROFILE_DB[user_id].append(fact)

def retrieve_user_facts(user_id: str) -> str:
    facts = USER_PROFILE_DB.get(user_id, [])
    if facts:
        return "Known facts: " + "; ".join(facts) + "."
    return "No specific facts stored about you yet."

def extract_and_store_fact_from_response(user_id: str, response: str):
    match = re.search(r"I will remember that you (like|prefer) '([^']+)'", response)
    if match:
        fact_type = match.group(1)
        item = match.group(2)
        store_user_fact(user_id, f"{fact_type} {item}")

if __name__ == "__main__":
    user_id = "user123"

    print("Welcome to the E-commerce Conversational Shopping Assistant!")
    print("Type 'exit' to end the conversation.")

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
    )

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Assistant: Goodbye!")
            break

        current_user_facts = retrieve_user_facts(user_id)
        
        input_data = {
            "input": user_input,
            "user_id": user_id,
            "user_facts": current_user_facts
        }

        try:
            response = agent_with_chat_history.invoke(
                input_data,
                config={"configurable": {"session_id": user_id}}
            )
            assistant_response = response["output"]
            print(f"Assistant: {assistant_response}")
            extract_and_store_fact_from_response(user_id, assistant_response)
        except Exception as e:
            print(f"Assistant: An error occurred: {e}. Please try again.")
