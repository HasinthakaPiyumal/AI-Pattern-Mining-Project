import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def simulate_llm_code_generation(financial_goals, risk_tolerance):
    """
    Simulates an LLM generating Python code for portfolio optimization.
    In a real system, this would be an actual LLM API call.
    """
    # Placeholder asset data for demonstration
    # In a real application, this data would be fetched from external APIs
    # or provided by the LLM based on user queries/market context.
    code = f"""
import numpy as np
from scipy.optimize import minimize

# --- Asset Data (simulated for demonstration) ---
# For a real system, these would come from market data or a financial API
assets = ['Tech Stock A', 'Bond Fund B', 'Real Estate ETF C']
expected_returns = np.array([0.15, 0.04, 0.08]) # Annual expected returns
# Covariance matrix (example, needs to be positive definite)
cov_matrix = np.array([
    [0.040, 0.005, 0.010],
    [0.005, 0.003, 0.002],
    [0.010, 0.002, 0.015]
])

num_assets = len(assets)

# --- Portfolio Functions ---
def portfolio_return(weights, returns):
    return np.sum(returns * weights)

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# --- Optimization Objective (Negative Sharpe Ratio for Maximization) ---
# Assuming a fixed risk-free rate for demonstration
risk_free_rate = 0.01

def neg_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    p_return = portfolio_return(weights, returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    if p_volatility == 0:
        return np.inf # Avoid division by zero
    return - (p_return - risk_free_rate) / p_volatility

# --- Constraints and Bounds ---
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Sum of weights = 1
bounds = tuple((0, 1) for asset in range(num_assets)) # Weights between 0 and 1

# Initial guess (equal weighting)
initial_weights = num_assets * [1./num_assets,]

# Determine optimization strategy based on risk tolerance
objective_function = None
if "{risk_tolerance}" == "Low":
    # Minimize volatility for low risk tolerance
    objective_function = lambda w: portfolio_volatility(w, cov_matrix)
elif "{risk_tolerance}" == "High":
    # Maximize Sharpe ratio for high risk tolerance
    objective_function = lambda w: neg_sharpe_ratio(w, expected_returns, cov_matrix, risk_free_rate)
else: # Medium risk tolerance, maximize Sharpe ratio
    objective_function = lambda w: neg_sharpe_ratio(w, expected_returns, cov_matrix, risk_free_rate)

# Perform optimization
result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x

# Calculate optimal portfolio's return and volatility
optimal_portfolio_return = portfolio_return(optimal_weights, expected_returns)
optimal_portfolio_volatility = portfolio_volatility(optimal_weights, cov_matrix)

# Store results in a dictionary that can be captured by the calling environment
results_output = {{
    "optimal_weights": optimal_weights.tolist(),
    "optimal_portfolio_return": optimal_portfolio_return,
    "optimal_portfolio_volatility": optimal_portfolio_volatility,
    "assets": assets
}}
    """
    return code

def execute_generated_code(code_string):
    """
    Executes the generated Python code in a controlled environment
    and captures its output.

    WARNING: Using exec() with untrusted input is extremely dangerous.
    For a production system, this must be run in a secure, isolated sandbox.
    """
    # Define a custom dictionary for the execution environment to capture outputs
    # and limit access to built-ins.
    local_vars = {
        'np': np,
        'minimize': minimize,
        'results_output': {} # This will be populated by the executed code
    }
    try:
        # Execute the code. __builtins__ is set to None to restrict access
        # to built-in functions, mitigating some (but not all) risks.
        exec(code_string, {'__builtins__': None}, local_vars)
        return local_vars['results_output']
    except Exception as e:
        return {"error": str(e)}

def simulate_llm_interpretation(financial_goals, risk_tolerance, execution_results):
    """
    Simulates an LLM interpreting the numerical results and providing natural language advice.
    In a real system, this would be an actual LLM API call, processing the numerical data
    and generating a natural language response.
    """
    if "error" in execution_results:
        return f"An error occurred during financial calculation: {execution_results['error']}. Please review your inputs or try again."

    optimal_weights = execution_results.get("optimal_weights", [])
    optimal_return = execution_results.get("optimal_portfolio_return", 0) * 100 # Convert to percentage
    optimal_volatility = execution_results.get("optimal_portfolio_volatility", 0) * 100 # Convert to percentage
    assets = execution_results.get("assets", [])

    advice_parts = [
        f"Based on your financial goals ('{financial_goals}') and a '{risk_tolerance}' risk tolerance, here's a personalized portfolio strategy:",
        "\n**Optimal Portfolio Allocation:**"
    ]

    if assets and optimal_weights:
        for i, asset in enumerate(assets):
            if i < len(optimal_weights):
                advice_parts.append(f"- {asset}: {optimal_weights[i]:.2%} of your portfolio")
    else:
        advice_parts.append("- No specific asset allocation could be determined.")

    advice_parts.append(f"\n**Projected Annual Performance:**")
    advice_parts.append(f"- Expected Return: {optimal_return:.2f}%")
    advice_parts.append(f"- Expected Volatility (Risk): {optimal_volatility:.2f}%")

    if risk_tolerance == "Low":
        advice_parts.append("\nThis portfolio emphasizes minimizing risk while aiming for a stable return, suitable for investors with a low risk tolerance.")
    elif risk_tolerance == "Medium":
        advice_parts.append("\nThis strategy strikes a balance between potential returns and managing risk, aligning with a medium risk tolerance.")
    else: # High
        advice_parts.append("\nThis aggressive portfolio targets higher returns, accepting a greater level of volatility, consistent with your high risk tolerance.")

    advice_parts.append("\n*Disclaimer: This is simulated financial advice based on hypothetical data and models. For real investment decisions, always consult a qualified financial advisor and conduct thorough due diligence.*")

    return "\n".join(advice_parts)

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="PAL Financial Advisor")

st.title("💰 Program-Aided Language Model (PAL) Financial Advisor")
st.write("Leveraging AI to generate and execute Python code for precise financial calculations and personalized investment advice.")

with st.sidebar:
    st.header("Your Financial Profile")
    financial_goals = st.text_area("What are your financial goals?", "Save for retirement in 20 years with moderate growth.")
    risk_tolerance = st.select_slider(
        "What is your risk tolerance?",
        options=["Low", "Medium", "High"],
        value="Medium"
    )
    analyze_button = st.button("Get Financial Advice")

st.subheader("How it Works:")
st.markdown("""
1.  **Input:** You provide your financial goals and risk tolerance.
2.  **Code Generation (AI):** Our AI (simulated LLM) generates Python code tailored to your inputs, incorporating financial algorithms like portfolio optimization.
3.  **Code Execution:** This generated code is then executed to perform precise numerical calculations.
4.  **Result Interpretation (AI):** The AI interprets the numerical output and translates it into easy-to-understand financial advice and portfolio recommendations.
""")

if analyze_button:
    st.subheader("1. AI Generates Financial Code:")
    generated_code = simulate_llm_code_generation(financial_goals, risk_tolerance)
    st.code(generated_code, language="python")

    st.subheader("2. Executing Code and Analyzing Results:")
    st.warning("🚨 **SECURITY WARNING:** Executing LLM-generated code (`exec()`) directly in a production environment carries significant security risks. This demonstration uses it for illustrative purposes. A real-world application would require robust sandboxing and security measures to protect against malicious code execution.")

    execution_results = execute_generated_code(generated_code)

    if "error" in execution_results:
        st.error(f"An error occurred during code execution: {execution_results['error']}")
    else:
        st.json(execution_results)

        st.subheader("3. AI Interprets Results and Provides Advice:")
        financial_advice = simulate_llm_interpretation(financial_goals, risk_tolerance, execution_results)
        st.success("Analysis Complete!")
        st.markdown(financial_advice)

st.sidebar.markdown("---")
st.sidebar.info("This application demonstrates the 'Program-Aided Language Models (PAL) Prompting' pattern. The LLM interactions are simulated for this demo, focusing on the architectural flow.")
