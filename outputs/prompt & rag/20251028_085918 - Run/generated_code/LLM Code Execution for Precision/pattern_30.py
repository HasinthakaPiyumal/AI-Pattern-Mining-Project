import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import numpy_financial as npf

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- LLM and LangChain Setup ---
llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)

# Prompt for code generation
code_gen_template = """
You are an expert financial analyst assistant. Your task is to generate Python code to answer financial questions. 
Use numpy_financial for common functions like PV, FV, NPV, IRR. For other calculations, use standard Python. 
The code should print the final numerical result(s) clearly. 
Do not include any explanations or extra text, just the executable Python code.

Example:
User Query: What is the future value of an investment of $1000 at 5% annual interest compounded annually for 10 years?
Generated Code:
print(npf.fv(0.05, 10, 0, -1000))

User Query: {query}
Generated Code:
"""
code_gen_prompt = PromptTemplate(input_variables=["query"], template=code_gen_template)
code_gen_chain = LLMChain(llm=llm, prompt=code_gen_prompt)

# Prompt for advice generation based on code output
advice_gen_template = """
You are a helpful financial advisor. Based on the following user query and the numerical result from a calculation, provide a clear, concise, and actionable financial advice or report.

User Query: {query}
Calculation Result: {result}

Financial Advice:
"""
advice_gen_prompt = PromptTemplate(input_variables=["query", "result"], template=advice_gen_template)
advice_gen_chain = LLMChain(llm=llm, prompt=advice_gen_prompt)

# --- Code Executor --- (Simple and illustrative - needs robust sandboxing for production)
def execute_code(code: str):
    try:
        # Capture stdout
        import io
        import sys
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        exec(code, {'npf': npf})
        
        sys.stdout = old_stdout # Restore stdout
        return redirected_output.getvalue().strip()
    except Exception as e:
        return f"Error during code execution: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="AI Financial Analyst", layout="centered")
st.title("🧠 AI Financial Analyst")
st.markdown("Enter your financial queries and get intelligent analysis and advice.")

user_query = st.text_area("Ask me a financial question:", height=100)

if st.button("Get Financial Advice"):
    if user_query:
        with st.spinner("Generating code for calculation..."):
            generated_code = code_gen_chain.run(query=user_query)
            st.subheader("Generated Code (for verification):")
            st.code(generated_code, language="python")

        with st.spinner("Executing code and analyzing results..."):
            calculation_result = execute_code(generated_code)
            st.subheader("Calculation Result:")
            st.write(calculation_result)
            
            if "Error" not in calculation_result:
                financial_advice = advice_gen_chain.run(query=user_query, result=calculation_result)
                st.subheader("Financial Advice:")
                st.write(financial_advice)
            else:
                st.error("Could not provide financial advice due to an error in calculation.")
    else:
        st.warning("Please enter a financial question.")

st.markdown("""
---
**Disclaimer:** This tool is for illustrative purposes only and should not be used for actual financial decisions. Always consult with a qualified financial professional.
""")