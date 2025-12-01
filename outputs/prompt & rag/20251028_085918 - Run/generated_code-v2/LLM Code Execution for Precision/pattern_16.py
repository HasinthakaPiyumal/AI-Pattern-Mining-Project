import streamlit as st
import math
import pandas as pd

# 1. Financial Calculation Utilities
def calculate_future_value(principal, annual_rate, years, compound_per_year=1):
    return principal * (1 + (annual_rate / compound_per_year)) ** (years * compound_per_year)

def calculate_compound_interest(principal, annual_rate, years, compound_per_year=1):
    return calculate_future_value(principal, annual_rate, years, compound_per_year) - principal

def calculate_portfolio_std_dev(returns):
    if not isinstance(returns, (pd.Series, list)):
        raise TypeError("Returns must be a pandas Series or a list.")
    if isinstance(returns, list):
        returns = pd.Series(returns)
    return returns.std()

# A dictionary of available financial functions for the LLM to use
FINANCIAL_FUNCTIONS = {
    "calculate_future_value": calculate_future_value,
    "calculate_compound_interest": calculate_compound_interest,
    "calculate_portfolio_std_dev": calculate_portfolio_std_dev,
}

# 2. Code Execution Environment (Simulated Sandbox)
def execute_generated_code(code_string, available_functions):
    try:
        # Create a limited execution environment
        exec_globals = {"__builtins__": None, "pd": pd, "math": math}
        exec_globals.update(available_functions)
        
        # Use a local dictionary to capture results from exec
        exec_locals = {}
        exec(code_string, exec_globals, exec_locals)
        
        # Assuming the generated code will set a 'result' variable
        return exec_locals.get("result", "No explicit result variable set by the code.")
    except Exception as e:
        return f"Error during code execution: {e}"

# 3. Simulated LLM Integration (Orchestration Layer component)
# In a real application, this would call an actual LLM API
def get_llm_response(prompt, user_context=None):
    # This is a simplified simulation of an LLM's behavior.
    # It tries to detect if a calculation is needed and generates simple Python.
    
    response_text = ""
    generated_code = None
    
    if "future value" in prompt.lower() and "principal" in prompt.lower() and "rate" in prompt.lower() and "years" in prompt.lower():
        # Example: Extract numbers from a simple prompt for demonstration
        # In a real LLM, this extraction would be more robust.
        try:
            principal = float(st.session_state.get('principal_input', 0))
            rate = float(st.session_state.get('rate_input', 0)) / 100
            years = int(st.session_state.get('years_input', 0))
            
            generated_code = f"result = calculate_future_value({principal}, {rate}, {years})"
            response_text = "I will calculate the future value of your investment." 
        except ValueError:
            response_text = "Please provide valid numbers for principal, rate, and years for future value calculation."
            
    elif "compound interest" in prompt.lower() and "principal" in prompt.lower() and "rate" in prompt.lower() and "years" in prompt.lower():
        try:
            principal = float(st.session_state.get('principal_input', 0))
            rate = float(st.session_state.get('rate_input', 0)) / 100
            years = int(st.session_state.get('years_input', 0))
            
            generated_code = f"result = calculate_compound_interest({principal}, {rate}, {years})"
            response_text = "Calculating your compound interest..."
        except ValueError:
            response_text = "Please provide valid numbers for principal, rate, and years for compound interest calculation."
            
    elif "portfolio risk" in prompt.lower() and "returns" in prompt.lower():
        returns_str = st.session_state.get('returns_input', '')
        if returns_str:
            try:
                returns_list = [float(x.strip()) for x in returns_str.split(',') if x.strip()]
                if returns_list:
                    generated_code = f"result = calculate_portfolio_std_dev({returns_list})"
                    response_text = "Analyzing your portfolio risk..."
                else:
                    response_text = "Please provide a comma-separated list of numerical returns."
            except ValueError:
                response_text = "Please provide a valid comma-separated list of numerical returns."
        else:
            response_text = "Please provide a list of returns for portfolio risk assessment."

    else:
        response_text = f"Hello! How can I help you with your financial planning today? (Simulated LLM response for: '{prompt}')"
        
    return response_text, generated_code

# 4. Streamlit UI (Orchestration Layer)
st.title("Financial Advisor AI (PAL Prompting Demo)")
st.write("This AI assists with financial calculations by generating and executing Python code.")

# User Inputs
st.header("Financial Inputs")
principal_input = st.number_input("Principal Amount ($)", min_value=0.0, value=1000.0, format="%.2f", key='principal_input')
rate_input = st.number_input("Annual Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0, format="%.2f", key='rate_input')
years_input = st.number_input("Number of Years", min_value=0, value=10, key='years_input')
returns_input = st.text_input("Comma-separated list of historical returns (e.g., 0.05, -0.02, 0.10)", key='returns_input')

user_query = st.text_area("Ask your financial question:", "What is the future value of $1000 at 5% interest over 10 years?")

if st.button("Get Financial Advice"):
    if user_query:
        st.subheader("AI's Thinking Process:")
        
        # Simulate LLM interaction
        llm_initial_response, generated_code = get_llm_response(user_query)
        st.info(f"LLM Initial Thought: {llm_initial_response}")
        
        calculation_result = None
        if generated_code:
            st.success(f"LLM Generated Code:\n```python\n{generated_code}\n```")
            
            # Execute the generated code in the sandbox
            calculation_result = execute_generated_code(generated_code, FINANCIAL_FUNCTIONS)
            st.write(f"Code Execution Result: {calculation_result}")
            
            # LLM re-integrates result to form final answer
            if "future value" in user_query.lower():
                final_advice = f"Based on my calculations, the future value of your investment will be approximately ${calculation_result:,.2f}. This is calculated from your principal of ${principal_input:,.2f}, an annual rate of {rate_input:.2f}%, over {years_input} years."
            elif "compound interest" in user_query.lower():
                final_advice = f"The compound interest earned on your investment will be approximately ${calculation_result:,.2f}. This accounts for your principal of ${principal_input:,.2f}, an annual rate of {rate_input:.2f}%, over {years_input} years."
            elif "portfolio risk" in user_query.lower():
                final_advice = f"Your portfolio's standard deviation (a measure of risk) is approximately {calculation_result:.4f}. This indicates the volatility of your returns {returns_input}."
            else:
                final_advice = f"I have completed the calculation and the result is {calculation_result}. How else can I assist?"
        else:
            final_advice = llm_initial_response # If no code was generated, initial response is the final
            
        st.subheader("Financial Advice:")
        st.success(final_advice)
    else:
        st.warning("Please enter a financial question.")
