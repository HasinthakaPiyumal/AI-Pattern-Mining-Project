import streamlit as st
import sys
import io
import contextlib

# Placeholder for LLM interaction and code execution logic
# In a real application, this would interact with an actual LLM API
# and a secure code execution environment.

# --- Simulated LLM and Code Execution Services ---
class CodeExecutor:
    """Simulates a secure environment for executing Python code."""
    def execute_python_code(self, code_string: str) -> str:
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        try:
            # Define a dictionary to hold variables during execution
            exec_globals = {}
            exec(code_string, exec_globals)
            output = redirected_output.getvalue()
            return output
        except Exception as e:
            return f"Error during code execution: {e}"
        finally:
            sys.stdout = old_stdout # Restore original stdout

class LLMService:
    """Simulates an LLM for code generation and natural language explanation.
    In a real application, this would make API calls to an actual LLM."""
    def __init__(self, executor: CodeExecutor):
        self.executor = executor

    def generate_code_and_explanation(self, user_query: str) -> tuple[str, str, str]:
        # --- Step 1: LLM generates code based on user query ---
        # This is a highly simplified simulation. A real LLM would parse
        # the query and generate appropriate financial calculation code.
        st.info(f"Simulating LLM analyzing query and generating code for: '{user_query}'")
        generated_code = self._simulate_code_generation(user_query)
        st.code(generated_code, language="python")

        # --- Step 2: Execute the generated code ---
        st.info("Executing generated Python code...")
        execution_output = self.executor.execute_python_code(generated_code)
        st.text_area("Code Execution Output", value=execution_output, height=150)

        # --- Step 3: LLM interprets results and generates natural language explanation ---
        st.info("Simulating LLM interpreting results and generating explanation...")
        final_explanation = self._simulate_explanation_generation(user_query, execution_output)

        return generated_code, execution_output, final_explanation

    def _simulate_code_generation(self, user_query: str) -> str:
        # Simple rule-based simulation for a few specific financial queries
        if "future value" in user_query.lower() and "compare" in user_query.lower():
            return """
# Financial calculation for two investments
import math

def future_value(principal, annual_rate, compounding_periods_per_year, years):
    return principal * (1 + annual_rate / compounding_periods_per_year)**(compounding_periods_per_year * years)

# Example based on typical user query for comparison
# Investment 1: $10,000 at 5% annual interest compounded monthly for 10 years
P1 = 10000
r1 = 0.05
n1 = 12 # monthly
t1 = 10

# Investment 2: $8,000 at 6% annual interest compounded annually for 12 years
P2 = 8000
r2 = 0.06
n2 = 1 # annually
t2 = 12

fv1 = future_value(P1, r1, n1, t1)
fv2 = future_value(P2, r2, n2, t2)

print(f"\nFuture Value of Investment 1 ($10,000, 5% monthly for 10 years): ${fv1:,.2f}")
print(f"Future Value of Investment 2 ($8,000, 6% annually for 12 years): ${fv2:,.2f}")

if fv1 > fv2:
    print(f"Investment 1 yields more by: ${fv1 - fv2:,.2f}")
elif fv2 > fv1:
    print(f"Investment 2 yields more by: ${fv2 - fv1:,.2f}")
else:
    print("Both investments yield the same amount.")
"""
        elif "loan payment" in user_query.lower() or "mortgage" in user_query.lower():
            return """
# Loan Payment Calculation
import math

def calculate_monthly_payment(principal, annual_rate, loan_term_years):
    monthly_rate = annual_rate / 12
    number_of_payments = loan_term_years * 12
    if monthly_rate == 0:
        return principal / number_of_payments
    payment = principal * (monthly_rate * (1 + monthly_rate)**number_of_payments) / ((1 + monthly_rate)**number_of_payments - 1)
    return payment

# Example: $200,000 loan, 4.5% annual interest, 30 years
principal_loan = 200000
annual_rate_loan = 0.045
loan_term_years = 30

monthly_payment = calculate_monthly_payment(principal_loan, annual_rate_loan, loan_term_years)

print(f"\nLoan Principal: ${principal_loan:,.2f}")
print(f"Annual Interest Rate: {annual_rate_loan * 100:.2f}%")
print(f"Loan Term: {loan_term_years} years")
print(f"Calculated Monthly Payment: ${monthly_payment:,.2f}")
"""
        else:
            return """
# Default code for simple arithmetic if specific financial query not recognized
result = 0
try:
    # Attempt to evaluate a simple mathematical expression from the query
    # WARNING: Direct eval() of user input is highly dangerous in a real app.
    # This is for demonstration purposes only within a simulated context.
    import re
    math_expression = re.search(r'calculate (.*)', user_query.lower())
    if math_expression:
        expression_str = math_expression.group(1).replace('plus', '+').replace('minus', '-')\
                                       .replace('times', '*').replace('divided by', '/')
        result = eval(expression_str) # UNSAFE for production without sandboxing
    else:
        result = "Could not parse a simple calculation from your query."
except Exception as e:
    result = f"Error in simple calculation: {e}"

print(f"\nCalculated result: {result}")
"""

    def _simulate_explanation_generation(self, user_query: str, execution_output: str) -> str:
        # Simple rule-based explanation based on the query and output
        if "future value" in user_query.lower() and "compare" in user_query.lower():
            explanation_prefix = "Based on your query regarding future value comparison, the calculations show:"
            return f"{explanation_prefix}\n\n{execution_output}\n\nThis detailed breakdown allows you to clearly see the performance of each investment over the specified periods. Investment 1 leverages monthly compounding over 10 years, while Investment 2 benefits from a higher annual rate over a longer term. The results indicate which option is financially more advantageous based on the provided parameters."
        elif "loan payment" in user_query.lower() or "mortgage" in user_query.lower():
            explanation_prefix = "Regarding your loan payment inquiry, the following details and monthly payment were calculated:"
            return f"{explanation_prefix}\n\n{execution_output}\n\nThis calculation provides your estimated monthly payment based on the principal amount, annual interest rate, and loan term. Understanding this helps in financial planning and budgeting for your loan."
        else:
            explanation_prefix = "Here is the result of your computational query:"
            return f"{explanation_prefix}\n\n{execution_output}\n\nIf you have a more specific financial question, please provide details such as principal, rates, and terms."


# --- Streamlit UI --- 
st.set_page_config(layout="wide", page_title="Intelligent Financial Advisor (PAL)")

st.title("🧠 Intelligent Financial Advisor (PAL Prompting)")
st.markdown("This AI assistant leverages Program-Aided Language Models (PAL) to perform accurate financial calculations and provide insightful explanations.")
st.markdown("\n")

# Initialize services
executor = CodeExecutor()
llm_service = LLMService(executor)

user_query = st.text_area(
    "Ask a financial question (e.g., 'Calculate the future value of $10,000 at 5% annual interest compounded monthly for 10 years and compare it with $8,000 at 6% annually for 12 years.' or 'What is the monthly payment for a $200,000 loan at 4.5% over 30 years?')",
    height=100,
    value="Calculate the future value of $10,000 at 5% annual interest compounded monthly for 10 years and compare it with $8,000 at 6% annually for 12 years."
)

if st.button("Get Financial Advice") and user_query:
    st.subheader("Processing your request...")
    
    generated_code, execution_output, final_explanation = llm_service.generate_code_and_explanation(user_query)

    st.subheader("Final Financial Advice")
    st.success(final_explanation)

    with st.expander("View Raw Details (Generated Code & Execution Output)"):
        st.write("**Generated Python Code:**")
        st.code(generated_code, language="python")
        st.write("**Code Execution Output:**")
        st.text_area("Output", value=execution_output, height=150, disabled=True)

elif not user_query:
    st.warning("Please enter a financial question to get advice.")

st.markdown("\n---")
st.markdown("**Note:** This is a simulated demonstration. In a real-world PAL application, the LLM would dynamically generate code and a secure, sandboxed environment would be used for execution.")
