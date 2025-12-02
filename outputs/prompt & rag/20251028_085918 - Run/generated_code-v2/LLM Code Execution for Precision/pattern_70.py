import streamlit as st
import math
import re

def simulate_llm_code_generation(query: str) -> str:
    """Simulates an LLM generating Python code based on a financial query."""
    code_template = """
def calculate_future_value(principal, annual_rate, years, compound_frequency):
    # Convert percentage rate to decimal
    rate = annual_rate / 100.0
    # Calculate future value
    future_value = principal * (1 + rate / compound_frequency)**(years * compound_frequency)
    return future_value

_calculation_result = calculate_future_value({principal}, {rate}, {years}, {compound_frequency})
"""

    principal = None
    rate = None
    years = None
    compound_frequency = None

    # Example parsing for 'future value' query
    principal_match = re.search(r"investment of (\$?\d+\.?\d*)", query, re.IGNORECASE)
    rate_match = re.search(r"rate of (\d+\.?\d*)\%?", query, re.IGNORECASE)
    years_match = re.search(r"over (\d+\.?\d*) years", query, re.IGNORECASE)
    compound_match = re.search(r"compounded (\d+) times a year", query, re.IGNORECASE)

    if principal_match: principal = float(principal_match.group(1).replace('$', ''))
    if rate_match: rate = float(rate_match.group(1))
    if years_match: years = float(years_match.group(1))
    if compound_match: compound_frequency = int(compound_match.group(1))

    if all([principal, rate, years, compound_frequency]):
        return code_template.format(principal=principal, rate=rate, years=years, compound_frequency=compound_frequency)
    else:
        return "# Error: Could not parse all required parameters for future value calculation."

def execute_code(code_string: str):
    """Safely executes the generated Python code and returns the result."""
    local_vars = {}
    try:
        exec(code_string, {}, local_vars)
        return local_vars.get("_calculation_result")
    except Exception as e:
        return f"Error during code execution: {e}"

st.set_page_config(layout="wide")
st.title("Financial Calculator Assistant (PAL Prompting Simulation)")

st.write("Ask me a financial question and I'll generate and execute Python code to find the answer.")
st.write("Example: `What is the future value of an investment of $10000 with a rate of 5% over 10 years compounded 4 times a year?`")

user_query = st.text_input("Your Financial Query:", "What is the future value of an investment of $10000 with a rate of 5% over 10 years compounded 4 times a year?")

if st.button("Calculate") and user_query:
    st.subheader("1. Original Query")
    st.code(user_query)

    st.subheader("2. Simulated LLM Code Generation")
    generated_code = simulate_llm_code_generation(user_query)
    st.code(generated_code, language="python")

    if "# Error" not in generated_code:
        st.subheader("3. Code Execution")
        calculation_result = execute_code(generated_code)

        if isinstance(calculation_result, (int, float)):
            st.success(f"Calculation Result: {calculation_result:,.2f}")
            st.subheader("4. Result Formulation")
            st.markdown(f"Based on your query, an investment of **${principal_match.group(1).replace('$', '')}** at an annual rate of **{rate_match.group(1)}%** compounded **{compound_match.group(1)} times per year** for **{years_match.group(1)} years** would be worth approximately **${calculation_result:,.2f}**.")
        else:
            st.error(f"Error during calculation: {calculation_result}")
            st.subheader("4. Result Formulation")
            st.write("Could not formulate a precise answer due to calculation errors.")
    else:
        st.error(generated_code.replace('# Error: ', ''))
        st.subheader("3. Code Execution")
        st.write("Skipped due to code generation error.")
        st.subheader("4. Result Formulation")
        st.write("Could not formulate an answer due to an inability to generate the calculation code.")
