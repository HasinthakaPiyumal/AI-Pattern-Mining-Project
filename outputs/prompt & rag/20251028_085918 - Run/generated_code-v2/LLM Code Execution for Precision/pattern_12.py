
import gradio as gr
import io
import contextlib
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --- 1. Simulate LLM for Code Generation ---
def simulate_llm_generate_code(user_query: str) -> str:
    """
    Simulates an LLM generating Python code for portfolio optimization based on a user query.
    In a real scenario, a sophisticated LLM would parse the query and construct
    the optimization logic dynamically. Here, we use simple regex and a template.
    """
    # Simple parsing to extract assets
    assets_match = re.search(r"stocks\s+([\w,\s]+)", user_query, re.IGNORECASE)
    assets_str = assets_match.group(1).strip() if assets_match else "AAPL,GOOG,MSFT"
    assets = [a.strip().upper() for a in assets_str.split(',') if a.strip()]

    # Template for the portfolio optimization code
    # This code includes Markowitz optimization using scipy.optimize
    generated_code_template = f"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Mock function to get historical data for demonstration
# In a real application, this would fetch real market data (e.g., using yfinance)
def get_mock_daily_returns(assets):
    data = {{
        'AAPL': [0.001, 0.002, -0.001, 0.003, 0.0005, 0.002, 0.001, -0.0005, 0.003, 0.001],
        'GOOG': [0.002, 0.001, 0.000, 0.0015, 0.002, 0.0015, 0.0005, 0.001, 0.0025, 0.001],
        'MSFT': [0.0005, 0.0015, -0.0005, 0.002, 0.001, 0.001, 0.000, 0.0015, 0.001, 0.002],
        'AMZN': [0.003, 0.001, 0.001, 0.0025, 0.0015, 0.002, 0.001, 0.0005, 0.001, 0.0025],
        'TSLA': [0.005, -0.002, 0.003, 0.001, 0.004, 0.003, -0.001, 0.002, 0.001, 0.004]
    }}
    selected_data = {{asset: data[asset][:10] for asset in assets if asset in data}}
    return pd.DataFrame(selected_data)

def calculate_portfolio_performance(weights, mean_returns, cov_matrix, annualization_factor=252):
    returns = np.sum(mean_returns * weights) * annualization_factor
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(annualization_factor)
    return returns, std

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate, annualization_factor=252):
    p_returns, p_std = calculate_portfolio_performance(weights, mean_returns, cov_matrix, annualization_factor)
    if p_std == 0:
        return np.inf # Avoid division by zero
    return -(p_returns - risk_free_rate) / p_std

def get_optimal_portfolio_weights(mean_returns, cov_matrix, risk_free_rate=0.01):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    constraints = ({{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]

    # Minimize the negative Sharpe Ratio to find the portfolio with the highest Sharpe Ratio
    optimal_weights_result = minimize(neg_sharpe_ratio, initial_weights, args=args,
                                      method='SLSQP', bounds=bounds, constraints=constraints)

    return optimal_weights_result.x

def main():
    assets = {assets} # Assets parsed from user query
    
    if not assets:
        print("Error: No assets specified or found.")
        return

    mock_returns_df = get_mock_daily_returns(assets)
    if mock_returns_df.empty or mock_returns_df.shape[1] < len(assets):
        print(f"Error: Could not get mock data for all specified assets: {{assets}}. Available: {{list(mock_returns_df.columns)}}.")
        return

    mean_daily_returns = mock_returns_df.mean()
    cov_matrix = mock_returns_df.cov()

    # User input parameters (could be parsed from LLM, hardcoded for demo)
    user_risk_free_rate = 0.01 

    optimal_weights = get_optimal_portfolio_weights(mean_daily_returns, cov_matrix, user_risk_free_rate)
    optimal_returns, optimal_std = calculate_portfolio_performance(optimal_weights, mean_daily_returns, cov_matrix)
    
    sharpe_ratio = (optimal_returns - user_risk_free_rate) / optimal_std if optimal_std != 0 else 0.0

    print("--- Portfolio Optimization Results ---")
    print("Assets considered:", assets)
    print("Optimal Weights:")
    for i, asset in enumerate(assets):
        print(f"  {{asset}}: {{optimal_weights[i]:.4f}}")
    print(f"Expected Annual Return: {{optimal_returns:.4f}}")
    print(f"Expected Annual Volatility (Std Dev): {{optimal_std:.4f}}")
    print(f"Sharpe Ratio (assuming risk-free rate of {{user_risk_free_rate}}): {{sharpe_ratio:.4f}}")

if __name__ == "__main__":
    main()
"""
    return generated_code_template


# --- 2. Code Execution Environment ---
def execute_python_code(code_string: str) -> str:
    """
    Executes the given Python code in a sandboxed environment and captures its output.
    WARNING: Executing arbitrary code from an LLM can be a security risk.
    This is for demonstration purposes. In production, use a secure sandbox.
    """
    old_stdout = io.StringIO()
    redirected_stdout = contextlib.redirect_stdout(old_stdout)

    try:
        # Ensure numpy, pandas, scipy.optimize are available in the exec context
        # by passing them into the globals/locals dict
        exec_globals = {
            '__builtins__': __builtins__,
            'np': np,
            'pd': pd,
            'minimize': minimize,
            'io': io,
            'contextlib': contextlib
        }
        with redirected_stdout:
            exec(code_string, exec_globals, exec_globals)
        return old_stdout.getvalue()
    except Exception as e:
        return f"Code Execution Error: {e}\n{old_stdout.getvalue()}"


# --- 3. Simulate LLM for Result Explanation ---
def simulate_llm_explain_results(optimization_output: str, user_query: str) -> str:
    """
    Simulates an LLM interpreting the numerical output from the code execution
    and generating a natural language explanation for the user.
    """
    explanation_parts = [
        "Based on your request and the computed optimization:",
        "The following optimal portfolio allocation has been determined to maximize the Sharpe Ratio (return per unit of risk)." if "Optimal Weights" in optimization_output else "The system encountered an issue during optimization or could not find a suitable portfolio.",
    ]

    # Parse key metrics from the optimization_output string
    weights_match = re.findall(r"  (\w+): ([\d.]+)", optimization_output)
    returns_match = re.search(r"Expected Annual Return: ([\d.]+)", optimization_output)
    volatility_match = re.search(r"Expected Annual Volatility \(Std Dev\): ([\d.]+)", optimization_output)
    sharpe_match = re.search(r"Sharpe Ratio \(assuming risk-free rate of [\d.]+\): ([\d.]+)", optimization_output)
    error_match = re.search(r"Error: (.+)", optimization_output)

    if error_match:
        explanation_parts.append(f"\n**Error Encountered:** {error_match.group(1)}")
        explanation_parts.append("Please check your input query or the availability of data for the specified assets.")
    elif weights_match:
        explanation_parts.append("\n**Optimal Asset Weights:**")
        for asset, weight in weights_match:
            explanation_parts.append(f"- **{asset}**: {float(weight)*100:.2f}%")
    
        if returns_match and volatility_match:
            expected_return = float(returns_match.group(1))
            expected_volatility = float(volatility_match.group(1))
            explanation_parts.append(f"\nThis portfolio is projected to have an **annual return of {expected_return*100:.2f}%** with an **annual volatility (risk) of {expected_volatility*100:.2f}%**.")

        if sharpe_match:
            sharpe_ratio = float(sharpe_match.group(1))
            explanation_parts.append(f"The calculated **Sharpe Ratio is {sharpe_ratio:.2f}**, indicating a good risk-adjusted return.")
            
        explanation_parts.append("\nThis optimization aims to provide the best possible return for the given level of risk, considering historical performance (mock data in this demo). Remember that past performance is not indicative of future results and investment decisions should always be made with careful consideration and professional advice.")
    else:
        explanation_parts.append("\nNo specific optimization results were found, possibly due to an error in the generated code or its execution.")

    explanation_parts.append(f"\nOriginal Query: '{{user_query}}'")

    return "\n".join(explanation_parts)

# --- Main Application Flow (using Gradio) ---
def run_financial_advisor(user_query: str):
    """
    Orchestrates the PAL prompting flow for the financial advisor.
    """
    if not user_query:
        return "Please enter a query.", "No code generated.", "No output.", "Please provide a query to get financial advice."

    # Step 1: LLM generates code
    generated_code = simulate_llm_generate_code(user_query)
    
    # Step 2: Execute the generated code
    code_output = execute_python_code(generated_code)

    # Step 3: LLM interprets output and generates natural language advice
    llm_explanation = simulate_llm_explain_results(code_output, user_query)

    return generated_code, code_output, llm_explanation

# Gradio Interface
iface = gr.Interface(
    fn=run_financial_advisor,
    inputs=gr.Textbox(lines=3, label="Your Financial Query", placeholder="e.g., Optimize my portfolio for maximum return given stocks AAPL, GOOG, MSFT, AMZN, TSLA"),
    outputs=[
        gr.Code(label="Generated Python Code (from LLM)", language="python", lines=20),
        gr.Textbox(label="Code Execution Output"),
        gr.Markdown(label="Financial Advice (from LLM)")
    ],
    title="PAL Financial Advisor: Program-Aided Portfolio Optimization",
    description="This system demonstrates Program-Aided Language Models (PAL) prompting. The LLM generates Python code to perform complex financial calculations (portfolio optimization), executes it, and then uses the results to provide natural language advice."
)

if __name__ == "__main__":
    iface.launch()