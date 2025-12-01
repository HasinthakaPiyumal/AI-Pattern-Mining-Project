from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate

import json

# Mock E-commerce Backend APIs
def search_product_by_keyword_mock(keyword: str) -> str:
    products = {
        "laptop": ["Dell XPS 15", "MacBook Air M2", "HP Spectre x360"],
        "smartphone": ["iPhone 15 Pro", "Samsung Galaxy S24", "Google Pixel 8"],
        "headphone": ["Sony WH-1000XM5", "Bose QuietComfort Ultra", "AirPods Max"],
        "t-shirt": ["Cotton Crewneck T-Shirt", "Graphic Tee", "V-Neck T-Shirt"]
    }
    results = products.get(keyword.lower(), [])
    if results:
        return f"Found products for '{keyword}': {', '.join(results)}."
    return f"No products found for '{keyword}'."

def get_product_details_mock(product_name: str) -> str:
    details = {
        "Dell XPS 15": "High-performance laptop with 15-inch display, Intel i9, 32GB RAM.",
        "iPhone 15 Pro": "Latest smartphone with A17 Bionic chip, pro camera system.",
        "Sony WH-1000XM5": "Industry-leading noise-canceling headphones with excellent sound quality."
    }
    detail = details.get(product_name, "Details not available.")
    return f"Details for {product_name}: {detail}"

def get_personalized_recommendations_mock(user_id: str) -> str:
    if user_id == "user123": # Example user
        return "Based on your history, we recommend: Gaming Mouse, Mechanical Keyboard, Ultra-wide Monitor."
    return "Please log in for personalized recommendations."

def get_popular_products_mock() -> str:
    return "Today's popular products are: Wireless Earbuds, Smartwatch, Robot Vacuum Cleaner."

# Wrap mock functions as Langchain tools
@tool
def search_products(keyword: str) -> str:
    return search_product_by_keyword_mock(keyword)

@tool
def get_details(product_name: str) -> str:
    return get_product_details_mock(product_name)

@tool
def get_personalized_recs(user_id: str = "user123") -> str:
    return get_personalized_recommendations_mock(user_id)

@tool
def get_popular_recs() -> str:
    return get_popular_products_mock()

# Memory Management
class ConversationMemory:
    def __init__(self):
        self.history = []
        self.user_facts = {}

    def add_interaction(self, user_query: str, assistant_response: str):
        self.history.append({"user": user_query, "assistant": assistant_response})

    def extract_and_store_facts(self, conversation_turn: str, llm):
        prompt_template = PromptTemplate(
            input_variables=["conversation_turn"],
            template="""Extract key user preferences, constraints, or facts from the following conversation turn. 
            If no facts are present, return 'No facts'. Otherwise, return a JSON object with key-value pairs.
            Example: {\"favorite_color\": \"blue\", \"budget\": \"500\"}
            Conversation: {conversation_turn}
            Facts:"""
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.run(conversation_turn=conversation_turn)
        if response and response != "No facts":
            try:
                extracted_facts = json.loads(response)
                self.user_facts.update(extracted_facts)
            except json.JSONDecodeError:
                pass

    def retrieve_context(self, current_query: str) -> str:
        context_parts = []
        if self.user_facts:
            context_parts.append(f"User Facts: {json.dumps(self.user_facts)}")
        
        # Simple retrieval of last few interactions
        for interaction in self.history[-3:]:
            context_parts.append(f"User: {interaction['user']}")
            context_parts.append(f"Assistant: {interaction['assistant']}")
        
        return "\n".join(context_parts)

# Initialize LLM (replace with your actual API key and model)
llm = ChatOpenAI(temperature=0, model_name="gpt-4-0613") # or gpt-3.5-turbo-0613

# Define the tools the LLM can use
tools = [
    search_products,
    get_details,
    get_personalized_recs,
    get_popular_recs
]

# Define the system prompt for the agent
system_prompt = """You are an AI-powered Conversational Shopping Assistant. 
Your goal is to help users find products and get recommendations based on their needs. 
You can use the following tools:

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

Begin! Remember to be helpful and conversational. If you need more information, ask clarifying questions. 
Always consider the conversation history and user facts to provide better assistance.
"""

# Create the agent prompt template
agent_prompt = ChatPromptTemplate.from_messages(
    [ ("system", system_prompt),
      ("user", "{input}\n{agent_scratchpad}\nContext: {context}")
    ]
)

# Create the agent
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

# Initialize memory
memory = ConversationMemory()

def run_shopping_assistant(user_input: str) -> str:
    # Retrieve relevant context from memory
    retrieved_context = memory.retrieve_context(user_input)
    
    # Prepare the input for the agent
    agent_input = {
        "input": user_input,
        "tool_names": [tool.name for tool in tools],
        "tools": tools,
        "context": retrieved_context
    }

    # Run the agent
    response = agent_executor.invoke(agent_input)
    assistant_response = response['output']

    # Update memory with current interaction
    memory.add_interaction(user_input, assistant_response)
    memory.extract_and_store_facts(f"User: {user_input} Assistant: {assistant_response}", llm)

    return assistant_response

if __name__ == "__main__":
    print("Welcome to your AI Shopping Assistant! Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break
        
        response = run_shopping_assistant(user_query)
        print(f"Assistant: {response}")
        print("---")
        # print("Current User Facts:", memory.user_facts) # For debugging
        # print("Conversation History:", memory.history) # For debugging

