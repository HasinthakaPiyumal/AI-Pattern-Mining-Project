import streamlit as st
import io
import contextlib
import re

# --- Mock LLM and Code Execution Functions ---

def mock_llm_generate_code(user_query: str) -> str:
    """Simulates an LLM generating Python code based on the user's query.
    In a real application, this would be an actual API call to an LLM.
    """
    # Simplified parsing and code generation for demonstration purposes.
    # A real LLM would use more sophisticated reasoning and tool integration.
    if "future value of an investment" in user_query.lower() or "calculate fv" in user_query.lower():
        # Example: "Calculate the future value of an investment of $1000 at 5% annual interest compounded annually for 10 years."
        # Example with contribution: "Calculate the future value of an investment of $5000 at 7% annual interest compounded annually for 15 years, with an additional annual contribution of $200."

        principal_match = re.search(r'\$(\d+\.?\d*)', user_query)
        rate_match = re.search(r'(\d+\.?\d*)\%', user_query)
        years_match = re.search(r'for (\d+) years', user_query)
        contribution_match = re.search(r'additional annual contribution of \$(\d+\.?\d*)', user_query)

        principal = float(principal_match.group(1)) if principal_match else 0.0
        rate = float(rate_match.group(1)) / 100 if rate_match else 0.0
        nper = int(years_match.group(1)) if years_match else 0
        pmt = -float(contribution_match.group(1)) if contribution_match else 0.0 # Negative for outflow

        # Present Value (pv) is typically negative as it's an outflow from the investor's perspective
        pv = -principal if principal > 0 else 0.0

        code = f"""
import numpy_financial as npf

rate = {rate} # annual interest rate
nper = {nper} # number of periods
pmt = {pmt} # payment (additional contribution), negative for outflow
pv = {pv} # present value (initial investment), negative for outflow

# Calculate future value
fv = npf.fv(rate, nper, pmt, pv)
print(f"The future value of the investment is: ${fv:,.2f}")
"""
        return code

    # Add more financial calculation cases here in a real LLM setup

    return "print(\"Sorry, I can only calculate future value for now. Please try a different query.\")"

def execute_python_code(code: str) -> str:
    """Executes Python code in a restricted environment and captures its output."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            with contextlib.redirect_stderr(stderr_capture):
                # A safer approach for real-world scenarios would involve sandboxing
                exec(code, {'npf': None}) # Pass a restricted global scope. npf will be imported inside the code.
        output = stdout_capture.getvalue()
        error = stderr_capture.getvalue()
        if error:
            return f"Execution Error:\n{error}"
        return output
    except Exception as e:
        return f"An error occurred during code execution: {e}"

def mock_llm_formulate_answer(user_query: str, code_output: str) -> str:
    """Simulates an LLM formulating a natural language answer based on code output."""
    if "future value" in user_query.lower() and "The future value" in code_output:
        fv_match = re.search(r'\$([\d,\.]+)', code_output)
        if fv_match:
            future_value = fv_match.group(0)
            return f"Based on your investment details, the calculated future value is {future_value}. This represents the total value of your investment after considering the principal, interest, and any additional contributions over the specified period. It's important to remember that these calculations are theoretical and actual returns may vary due to market conditions, fees, and taxes."
    
    if "Execution Error" in code_output or "An error occurred" in code_output:
        return f"I encountered an issue while trying to process your request. Here are the details of the error: {code_output} Please review your input or try a different query."

    return f"I have processed your request, and the computation resulted in: {code_output.strip()} If you need further clarification or different analysis, please let me know."

# --- Streamlit Application ---
st.set_page_config(layout="wide", page_title="Smart Financial Advisor PAL")
st.title("Smart Financial Advisor (PAL Demo)")
st.markdown("Ask complex financial questions and let our AI assist with precise calculations using Program-Aided Language Models.")

user_query = st.text_area("Enter your financial query here:", 
                          "Calculate the future value of an investment of $10000 at 6% annual interest compounded annually for 20 years, with an additional annual contribution of $500.")

if st.button("Get Financial Advice"):
    if user_query:
        st.subheader("1. LLM Generates Code")
        generated_code = mock_llm_generate_code(user_query)
        st.code(generated_code, language="python")

        st.subheader("2. Code Execution Output")
        execution_output = execute_python_code(generated_code)
        st.text(execution_output)

        st.subheader("3. LLM Formulates Final Answer")
        final_answer = mock_llm_formulate_answer(user_query, execution_output)
        st.info(final_answer)
    else:
        st.warning("Please enter a financial query.")

st.markdown("---")
st.markdown("**Note:** This is a simplified demonstration of the PAL pattern. The LLM's code generation and answer formulation are mocked for illustrative purposes.")
