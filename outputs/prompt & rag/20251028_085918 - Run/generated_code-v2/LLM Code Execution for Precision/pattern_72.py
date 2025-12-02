import streamlit as st
import io
import contextlib

def generate_code_with_llm_simulation(query: str) -> str:
    if "loan amortization" in query.lower() or "monthly payment" in query.lower():
        try:
            parts = query.lower().replace(",", "").replace("$", "").split()
            principal_index = parts.index("loan") - 1 if "loan" in parts else None
            rate_index = parts.index("interest") - 1 if "interest" in parts else None
            years_index = parts.index("years") - 1 if "years" in parts else None

            principal = float(parts[principal_index]) if principal_index is not None and principal_index >= 0 else 200000
            rate_str = parts[rate_index].replace("%", "") if rate_index is not None and rate_index >= 0 else "4.5"
            annual_interest_rate = float(rate_str) / 100
            term_years = float(parts[years_index]) if years_index is not None and years_index >= 0 else 30

        except (ValueError, IndexError):
            principal = 200000
            annual_interest_rate = 0.045
            term_years = 30
            st.warning("Could not parse loan details from your query. Using default values for demonstration.")

        return f"""
def calculate_monthly_payment(principal, annual_interest_rate, term_years):
    monthly_interest_rate = annual_interest_rate / 12
    number_of_payments = term_years * 12
    if monthly_interest_rate == 0:
        monthly_payment = principal / number_of_payments
    else:
        monthly_payment = (principal * monthly_interest_rate) / (1 - (1 + monthly_interest_rate)**(-number_of_payments))
    return round(monthly_payment, 2)

principal = {principal}
annual_interest_rate = {annual_interest_rate}
term_years = {term_years}
result = calculate_monthly_payment(principal, annual_interest_rate, term_years)
print(f"The calculated monthly payment is: ${{result}}")
"""
    elif "investment growth" in query.lower() or "future value" in query.lower():
        try:
            parts = query.lower().replace(",", "").replace("$", "").split()
            initial_index = parts.index("investment") - 1 if "investment" in parts else None
            return_index = parts.index("return") - 1 if "return" in parts else None
            years_index = parts.index("years") - 1 if "years" in parts else None

            initial_investment = float(parts[initial_index]) if initial_index is not None and initial_index >= 0 else 10000
            return_rate_str = parts[return_index].replace("%", "") if return_index is not None and return_index >= 0 else "7"
            annual_return_rate = float(return_rate_str) / 100
            years = float(parts[years_index]) if years_index is not None and years_index >= 0 else 10
        except (ValueError, IndexError):
            initial_investment = 10000
            annual_return_rate = 0.07
            years = 10
            st.warning("Could not parse investment details from your query. Using default values for demonstration.")

        return f"""
def calculate_future_value(initial_investment, annual_return_rate, years):
    future_value = initial_investment * (1 + annual_return_rate)**years
    return round(future_value, 2)

initial_investment = {initial_investment}
annual_return_rate = {annual_return_rate}
years = {years}
result = calculate_future_value(initial_investment, annual_return_rate, years)
print(f"The future value of your investment will be: ${{result}}")
"""
    else:
        return """
print("I'm sorry, I can only calculate loan amortization and investment growth at the moment. Please try a different query.")
"""

def execute_python_code(code_string: str) -> str:
    old_stdout = io.StringIO()
    redirect_stdout = contextlib.redirect_stdout(old_stdout)
    with redirect_stdout:
        try:
            exec(code_string)
        except Exception as e:
            return f"Error during code execution: {e}"
    return old_stdout.getvalue().strip()

def generate_final_answer_with_llm_simulation(original_query: str, code_output: str) -> str:
    if "loan amortization" in original_query.lower():
        if "Error" in code_output:
            return f"I encountered an error calculating your loan payment: {code_output}. Please check your input."
        return f"Based on my calculations:\n{code_output}\n\nThis is your estimated monthly payment for the loan details provided. Please note that this is an estimation and may not include taxes, insurance, or other fees."
    elif "investment growth" in original_query.lower():
        if "Error" in code_output:
            return f"I encountered an error calculating your investment growth: {code_output}. Please check your input."
        return f"Based on my calculations:\n{code_output}\n\nThis is the projected future value of your investment. Remember that actual returns can vary and are not guaranteed."
    else:
        return code_output

st.set_page_config(page_title="PAL Finance Advisor", layout="wide")

st.title("💰 Personal Finance Advisor (PAL Prompting Demo) 💰")
st.markdown(
    """
    This application demonstrates the **Program-Aided Language Models (PAL) Prompting** pattern.
    It simulates an AI assistant that, for complex financial calculations,
    generates and executes Python code to ensure precision, then uses the
    results to formulate a natural language answer.
    """
)

user_query = st.text_area(
    "Ask me about personal finance (e.g., loan amortization, investment growth):",
    "What is the monthly payment for a $250,000 loan at 3.8% interest over 20 years?"
)

if st.button("Get Financial Advice"):
    if user_query:
        st.subheader("Thinking Process:")
        st.info("1. AI receives your query.")

        with st.spinner("AI generating code for calculation..."):
            generated_code = generate_code_with_llm_simulation(user_query)
            st.code(generated_code, language="python")
            st.success("2. AI generated Python code based on your request.")

        with st.spinner("Executing generated code..."):
            execution_output = execute_python_code(generated_code)
            st.text_area("Code Execution Output:", execution_output, height=100)
            st.success("3. Code executed and results obtained.")

        with st.spinner("AI formulating final answer..."):
            final_answer = generate_final_answer_with_llm_simulation(user_query, execution_output)
            st.success("4. AI formulated the final answer.")

        st.subheader("Your Financial Advice:")
        st.write(final_answer)

        st.markdown("""
            ---
            **Important Security Note**: In a real-world production system, directly executing
            arbitrary code generated by an untrusted source (like an LLM) using `exec()` is a
            **severe security vulnerability**. A robust solution would require a highly
            sandboxed execution environment (e.g., a dedicated microservice, containerized
            execution, or a strictly controlled environment with whitelisted functions) to
            prevent malicious code injection or system compromise. This demonstration uses
            `exec()` for simplicity in illustrating the pattern, but it is not
            production-ready from a security perspective.
            """)
    else:
        st.warning("Please enter a financial query.")
