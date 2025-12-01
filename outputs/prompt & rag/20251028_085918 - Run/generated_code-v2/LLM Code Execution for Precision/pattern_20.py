import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import io
import contextlib
import numpy_financial as npf
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0)

code_generation_template = """You are an expert financial assistant. Based on the user's financial query, generate a Python code snippet that performs the necessary calculations.
The code should be executable and print the final result.
Use `numpy_financial` for standard financial functions if applicable (e.g., npf.pv, npf.fv, npf.pmt).
Do NOT include any explanations or extra text, just the Python code.

Example Query: "Calculate the future value of an investment of $1000 with an annual interest rate of 5% compounded annually for 10 years."
Example Code:
import numpy_financial as npf
principal = 1000
rate = 0.05
nper = 10
fv = npf.fv(rate, nper, 0, -principal)
print(f"The future value is: ${fv:.2f}")

User Query: {query}
Python Code:
"""
code_generation_prompt = PromptTemplate.from_template(code_generation_template)
code_generation_chain = code_generation_prompt | llm | StrOutputParser()

explanation_template = """You are a financial advisory assistant. The user asked the following financial question:
"{original_query}"

You executed a Python script, and here is its output:
"{code_output}"

Based on this output, provide a clear, concise, and helpful natural language answer to the user's original question.
Explain the result in an easy-to-understand manner.
"""
explanation_prompt = PromptTemplate.from_template(explanation_template)
explanation_chain = explanation_prompt | llm | StrOutputParser()

def execute_code(code_string):
    redirected_output = io.StringIO()
    exec_error = None

    # In a real application, this would be a much more carefully constructed or sandboxed environment.
    # For this demonstration, we are making numpy_financial available directly.
    restricted_globals = {
        "__builtins__": {},
        "print": print,
        "npf": npf,
        "__name__": "__main__",
    }
    restricted_locals = {}

    try:
        with contextlib.redirect_stdout(redirected_output):
            exec(code_string, restricted_globals, restricted_locals)
        result = redirected_output.getvalue()
    except Exception as e:
        exec_error = str(e)

    if exec_error:
        return None, exec_error
    else:
        return result, None

st.set_page_config(page_title="AI Financial Advisory Assistant", layout="centered")

st.title("💰 AI Financial Advisory Assistant")
st.markdown("Ask complex financial questions and get precise answers powered by AI-generated code execution.")

user_query = st.text_area("Enter your financial question:", height=100, placeholder="e.g., What is the monthly payment for a $300,000 mortgage at 3.5% interest over 30 years?")

if st.button("Calculate"):
    if user_query:
        with st.spinner("Thinking and generating code..."):
            try:
                generated_code = code_generation_chain.invoke({"query": user_query})
                st.subheader("Generated Python Code:")
                st.code(generated_code, language="python")

                st.subheader("Code Execution Output:")
                execution_output, execution_error = execute_code(generated_code)

                if execution_error:
                    st.error(f"Error during code execution: {execution_error}")
                    st.write("Please refine your query or contact support if the issue persists.")
                else:
                    st.text(execution_output if execution_output else "No output from code execution.")

                    with st.spinner("Generating explanation..."):
                        final_answer = explanation_chain.invoke({
                            "original_query": user_query,
                            "code_output": execution_output if execution_output else "No output was produced by the code."
                        })
                        st.subheader("AI Assistant's Answer:")
                        st.write(final_answer)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.write("Please try again or simplify your query.")
    else:
        st.warning("Please enter a financial question to proceed.")

st.markdown("---")
st.info("💡 **Disclaimer:** This is an AI assistant for informational purposes only and should not be considered financial advice. Always consult with a qualified financial professional for personalized guidance.")