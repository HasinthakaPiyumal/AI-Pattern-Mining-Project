import streamlit as st
import io
import contextlib

def mock_llm_generate_code(query: str) -> str:
    """Simulates an LLM generating Python code based on a financial query."""
    if "compound interest" in query.lower() or "future value" in query.lower():
        if "$10,000 with a 7% annual return for 15 years" in query:
            return """
principal = 10000
annual_rate_1 = 0.07
time_1 = 15
future_value_1 = principal * (1 + annual_rate_1)**time_1
print(f"Future Value (7% for 15 years): ${future_value_1:.2f}")

annual_rate_2 = 0.08
time_2 = 10
future_value_2 = principal * (1 + annual_rate_2)**time_2
print(f"Future Value (8% for 10 years): ${future_value_2:.2f}")

if future_value_1 > future_value_2:
    print(f"Investing for 15 years at 7% yields a higher future value by ${future_value_1 - future_value_2:.2f}.")
elif future_value_2 > future_value_1:
    print(f"Investing for 10 years at 8% yields a higher future value by ${future_value_2 - future_value_1:.2f}.")
else:
    print("Both investment scenarios yield the same future value.")
            """
        else:
            return """
# Generic compound interest calculation. Please specify principal, rate, and time.
principal = 1000  # Example value
annual_rate = 0.05 # Example value (5%)
time = 10 # Example value (10 years)
future_value = principal * (1 + annual_rate)**time
print(f"Calculated Future Value: ${future_value:.2f}")
            """
    elif "mortgage payment" in query.lower():
        return """
# Mortgage payment calculation. Please specify loan amount, interest rate, and loan term.
loan_amount = 200000 # Example value
annual_interest_rate = 0.04 # Example value (4%)
loan_term_years = 30 # Example value (30 years)

monthly_interest_rate = annual_interest_rate / 12
number_of_payments = loan_term_years * 12

if monthly_interest_rate > 0:
    monthly_payment = loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate)**number_of_payments) / (((1 + monthly_interest_rate)**number_of_payments) - 1)
else:
    monthly_payment = loan_amount / number_of_payments

print(f"Estimated Monthly Mortgage Payment: ${monthly_payment:.2f}")
        """
    else:
        return "print(\"No specific financial calculation code generated for this query.\")"

def execute_code(code: str) -> str:
    """Executes Python code and captures its stdout."""
    old_stdout = io.StringIO()
    redirect_stdout = contextlib.redirect_stdout(old_stdout)
    with redirect_stdout:
        try:
            exec(code, globals())
        except Exception as e:
            return f"Error during code execution: {e}"
    return old_stdout.getvalue()

def mock_llm_formulate_answer(query: str, code_output: str) -> str:
    """Simulates an LLM formulating a natural language answer based on the query and code output."""
    if "compound interest" in query.lower() or "future value" in query.lower():
        if "$10,000 with a 7% annual return for 15 years" in query:
            lines = code_output.strip().split('\n')
            fv1 = "N/A"
            fv2 = "N/A"
            comparison = ""
            for line in lines:
                if "Future Value (7% for 15 years)" in line:
                    fv1 = line.split(": ")[1]
                elif "Future Value (8% for 10 years)" in line:
                    fv2 = line.split(": ")[1]
                elif "yields a higher future value" in line or "same future value" in line:
                    comparison = line
            return (f"Based on your investment query:\n"\
                    f"- The future value of investing $10,000 at 7% annual return for 15 years is {fv1}.\n"\
                    f"- The future value of investing $10,000 at 8% annual return for 10 years is {fv2}.\n"\
                    f"Comparison: {comparison}\n"\
                    f"This analysis indicates the specific outcomes for each scenario, helping you compare potential returns.")
        else:
            return f"Here's the financial analysis based on the executed code:\n\n{code_output}\n\nPlease provide specific values for a more detailed calculation and advice."
    elif "mortgage payment" in query.lower():
        return f"Your mortgage payment estimate is:\n\n{code_output}\n\nThis calculation provides an approximate monthly payment based on the provided loan details."
    else:
        return f"I have processed your request. Here's the output from the generated code:\n\n{code_output}\n\nIf you have a specific financial calculation in mind, please be more precise with your query."

st.set_page_config(layout="wide", page_title="Smart Financial Analyst")
st.title("Smart Financial Analyst (PAL Prompting Demo)")

st.markdown("This application demonstrates how a Language Model (simulated) can generate and execute code to perform precise financial calculations.")

user_query = st.text_area("Enter your financial query:", 
                            value="If I invest $10,000 with a 7% annual return for 15 years, what will be the future value, and how does it compare to investing for 10 years at 8%?", 
                            height=100)

if st.button("Get Financial Analysis"):
    if user_query:
        st.subheader("1. LLM Generates Code")
        generated_code = mock_llm_generate_code(user_query)
        st.code(generated_code, language="python")

        st.subheader("2. Executing Code")
        st.info("Executing the generated Python code...")
        code_execution_output = execute_code(generated_code)
        st.text_area("Code Execution Output:", code_execution_output, height=150)

        st.subheader("3. LLM Formulates Answer")
        final_answer = mock_llm_formulate_answer(user_query, code_execution_output)
        st.success("Final Financial Analysis and Advice:")
        st.write(final_answer)
    else:
        st.warning("Please enter a financial query.")
