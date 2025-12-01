import streamlit as st
import io
import contextlib
import os

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

# Set your OpenAI API key
# It's recommended to set this as an environment variable for production
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@tool
def python_interpreter(code: str) -> str:
    """Executes Python code and returns its output. Be careful with security.
    Input should be valid Python code as a string.
    """
    old_stdout = io.StringIO()
    redirect_stdout = contextlib.redirect_stdout(old_stdout)

    try:
        with redirect_stdout:
            exec(code, {'__builtins__': {}})
        output = old_stdout.getvalue()
        return f"Execution successful. Output:\n{output}"
    except Exception as e:
        return f"Execution failed. Error:\n{e}"


# Initialize LLM for code generation and general reasoning
llm_code_gen = ChatOpenAI(model="gpt-4o", temperature=0.0)

# Define the tools available to the agent
tools = [python_interpreter]

# Define the agent's prompt
prompt_template = PromptTemplate.from_template(
    """You are a helpful financial advisor. Your goal is to assist users with complex financial calculations and advice.
    You have access to a Python interpreter tool to perform calculations accurately.
    When a user asks a question that requires numerical computation, generate Python code to solve it using the `python_interpreter` tool.
    Present the final answer clearly and concisely, incorporating the results from the Python execution.
    If the question is not financial, respond appropriately.

    Here are the tools you have:
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
)

# Create the ReAct agent
agent = create_react_agent(llm_code_gen, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

st.set_page_config(page_title="PAL Personal Finance Advisor", layout="centered")
st.title("💰 PAL Personal Finance Advisor")

st.markdown("Ask me anything about your personal finance, and I'll use my computational skills to help you!")

user_query = st.text_area("Enter your financial question:", "What is the future value of an investment of $10,000 at an annual interest rate of 5% compounded monthly for 10 years?")

if st.button("Get Financial Advice"):
    if user_query:
        with st.spinner("Calculating and generating advice..."):
            try:
                response = agent_executor.invoke({"input": user_query})
                st.subheader("Your Financial Advice:")
                st.write(response["output"])
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("Please try rephrasing your question or contact support if the issue persists.")
    else:
        st.warning("Please enter a financial question to get advice.")

st.markdown("---")
st.markdown("**Note on Security**: This demonstration uses a simplified `exec` for Python code execution. In a production environment, a robust and secure sandboxed execution environment is crucial to prevent malicious code injection.")