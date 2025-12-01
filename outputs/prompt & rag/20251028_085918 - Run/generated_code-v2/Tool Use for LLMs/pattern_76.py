import streamlit as st
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM

load_dotenv()

class AmazonProductSearchInput(BaseModel):
    product_keyword: str = Field(description="the keyword to search for products on Amazon")

class AmazonProductSearchTool(BaseTool):
    name: str = "amazon_product_search"
    description: str = "Searches for products on Amazon. Use this tool when you need to find products based on a keyword. Input should be a string representing the product keyword."
    args_schema: type[BaseModel] = AmazonProductSearchInput

    def _run(self, product_keyword: str) -> str:
        if "laptop" in product_keyword.lower():
            return f"Found 'Gaming Laptop XYZ' for $1200, 'UltraBook ABC' for $900 on Amazon for keyword '{product_keyword}'."
        elif "headphone" in product_keyword.lower():
            return f"Found 'Noise-Cancelling Headphones' for $250, 'Wireless Earbuds' for $80 on Amazon for keyword '{product_keyword}'."
        return f"No specific products found for '{product_keyword}' on Amazon. Try searching for 'Gaming Laptop XYZ'."

class eBayPriceComparisonInput(BaseModel):
    product_name: str = Field(description="the name of the product for which to compare prices on eBay")

class eBayPriceComparisonTool(BaseTool):
    name: str = "ebay_price_comparison"
    description: str = "Compares prices for a specific product on eBay. Use this tool when you need to compare prices for an item. Input should be a string representing the product name."
    args_schema: type[BaseModel] = eBayPriceComparisonInput

    def _run(self, product_name: str) -> str:
        if "gaming laptop xyz" in product_name.lower():
            return f"eBay prices for 'Gaming Laptop XYZ': New from $1150, Used from $950. Best deal: Refurbished at $1050."
        elif "noise-cancelling headphones" in product_name.lower():
            return f"eBay prices for 'Noise-Cancelling Headphones': New from $240, Used from $180. Best deal: Open-box at $200."
        return f"No specific price comparison available for '{product_name}' on eBay. Try 'Gaming Laptop XYZ'."

tools = [
    AmazonProductSearchTool(),
    eBayPriceComparisonTool()
]

# Placeholder LLM for demonstration
llm = FakeListLLM(responses=[
    "Action: amazon_product_search\nAction Input: Gaming Laptop XYZ",
    "Found 'Gaming Laptop XYZ' for $1200, 'UltraBook ABC' for $900 on Amazon for keyword 'Gaming Laptop XYZ'.",
    "Action: ebay_price_comparison\nAction Input: Gaming Laptop XYZ",
    "eBay prices for 'Gaming Laptop XYZ': New from $1150, Used from $950. Best deal: Refurbished at $1050.",
    "Okay, I found 'Gaming Laptop XYZ' on Amazon for $1200 and on eBay, new ones start from $1150. Would you like more details or another search?",
    "Action: amazon_product_search\nAction Input: best noise cancelling headphones",
    "Found 'Noise-Cancelling Headphones' for $250, 'Wireless Earbuds' for $80 on Amazon for keyword 'best noise cancelling headphones'.",
    "Action: ebay_price_comparison\nAction Input: Noise-Cancelling Headphones",
    "eBay prices for 'Noise-Cancelling Headphones': New from $240, Used from $180. Best deal: Open-box at $200.",
    "I found Noise-Cancelling Headphones on Amazon for $250. On eBay, new ones are available from $240, and open-box at $200. Is there anything else I can help you with?",
    "I'm sorry, I can only help with product searches and price comparisons. Please ask me about products or prices."
])

# Define the agent prompt
# The system message instructs the AI on its persona and capabilities.
# It emphasizes using tools effectively.
# The `tools` variable is explicitly referenced in the prompt to ensure the LLM understands available tools.
# The agent_scratchpad is crucial for the ReAct pattern.
# The Human message is where the user's query comes in.

agent_prompt_template = PromptTemplate.from_template(
    """You are an AI e-commerce assistant. Your goal is to help users find products and compare prices across different online retailers.
    You have access to the following tools:

    {tools}

    To use a tool, please use the following format:

    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action

    When you have a final answer, respond in a natural conversational tone to the user.

    Begin!

    {agent_scratchpad}
    Human: {input}
    """
)

agent = create_react_agent(llm, tools, agent_prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

st.set_page_config(page_title="AI E-commerce Assistant")
st.title("🛍️ AI E-commerce Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What product are you looking for or want to compare prices for?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent_executor.invoke({"input": prompt})
                st.markdown(response["output"])
                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
            except Exception as e:
                error_message = f"I apologize, but I encountered an error: {e}. Please try again."
                st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
