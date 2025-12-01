import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import PromptTemplate
import numpy as np
import numpy_financial as npf
import io
import contextlib
import sys

def execute_python_code(code: str) -> str:
    local_vars = {"np": np, "npf": npf}
    global_vars = {
        "__builtins__": {
            "print": print, "len": len, "str": str, "int": int, "float": float,
            "dict": dict, "list": list, "tuple": tuple, "set": set,
            "sum": sum, "min": min, "max": max, "range": range,
            "abs": abs, "round": round, "bool": bool,
            "True": True, "False": False, "None": None,
            "ArithmeticError": ArithmeticError, "AssertionError": AssertionError,
            "AttributeError": AttributeError, "Exception": Exception,
            "IndexError": IndexError, "KeyError": KeyError, "NameError": NameError,
            "OverflowError": OverflowError, "RuntimeError": RuntimeError,
            "StopIteration": StopIteration, "SyntaxError": SyntaxError,
            "TypeError": TypeError, "ValueError": ValueError, "ZeroDivisionError": ZeroDivisionError
        }
    }
    
    output_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_capture):
            exec(code, global_vars, local_vars)
        return output_capture.getvalue()
    except Exception as e:
        return f"Error during code execution: {e}\nCaptured output before error:\n{output_capture.getvalue()}"

llm = ChatOpenAI(temperature=0, model="gpt-4")

tools = [
    Tool(
        name="Python_Code_Executor",
        func=execute_python_code,
        description="Executes Python code. Input should be valid Python code. Useful for financial calculations, data manipulation, and scientific computing. Available libraries: numpy (as np), numpy_financial (as npf)."
    )
]

prompt_template = PromptTemplate.from_template("""
You are an expert financial analyst assistant. Your goal is to help users with complex financial calculations, risk assessments, and portfolio optimizations.
When a user asks a question that requires precise numerical computation, generate and execute Python code using the 'Python_Code_Executor' tool.
After executing the code, synthesize the numerical output into a clear, understandable financial analysis or recommendation in natural language.
If the question does not require code execution, answer directly.

Available libraries in the Python environment:
- numpy as np
- numpy_financial as npf

For example:
Question: "Calculate the NPV for an initial investment of -100,000, and cash flows of 20,000, 30,000, 40,000, 50,000 over 4 years with a discount rate of 10%."
Thought: The user is asking for Net Present Value (NPV). This requires numerical computation, so I should generate and execute Python code using numpy_financial.
Action: Python_Code_Executor
Action Input:
cash_flows = [-100000, 20000, 30000, 40000, 50000]
discount_rate = 0.10
npv = npf.npv(discount_rate, cash_flows)
print(f"The Net Present Value (NPV) is: {npv:.2f}")

After execution, you will see the output. Use it to formulate your final answer.

Always prioritize precise numerical results obtained from code execution when applicable.

Question: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

st.title("Intelligent Financial Analyst Assistant")
st.write("Ask me complex financial questions, and I'll use Python to compute the answers!")

user_query = st.text_area("Enter your financial query:")

if st.button("Analyze"):
    if user_query:
        with st.spinner("Analyzing..."):
            try:
                response = agent_executor.invoke({"input": user_query})
                st.subheader("Analysis Result:")
                st.write(response["output"])
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
    else:
        st.warning("Please enter a financial query.")
