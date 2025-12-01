
import pandas as pd
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import io
import contextlib
import re
import json

# --- Step 1: Simulate LLM's understanding and code generation ---
def generate_optimization_script(user_query: str) -> str:
    """
    Simulates an LLM generating a Python script for portfolio optimization
    based on a natural language query. This is a simplified example.
    In a real PAL system, the LLM would be much more sophisticated at parsing
    and generating dynamic, robust code.
    """
    portfolio_value = 100000  # Default
    risk_tolerance = "moderate"  # Default
    target_return = 0.08  # Default (8%)

    # Simple parsing using regex to extract parameters
    value_match = re.search(r'\$?([\d,]+\.?\d*)\s*(?:portfolio|investment|capital)', user_query, re.IGNORECASE)
    if value_match:
        portfolio_value = float(value_match.group(1).replace(',', ''))

    risk_match = re.search(r'(low|moderate|high)\s*risk', user_query, re.IGNORECASE)
    if risk_match:
        risk_tolerance = risk_match.group(1).lower()

    return_match = re.search(r'(\d+\.?\d*)\%\s*(?:annual\s*)?return', user_query, re.IGNORECASE)
    if return_match:
        target_return = float(return_match.group(1)) / 100

    # Map risk tolerance to a factor for a simple proxy (e.g., gamma for PyPortfolioOpt)
    risk_gamma = {
        "low": 0.5,
        "moderate": 1.0,
        "high": 2.0
    }.get(risk_tolerance, 1.0)

    # Define a set of diverse assets for demonstration
    assets = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "JPM", "VWO", "GLD"]

    # Generate the Python script that performs the financial calculation
    # The script is designed to be executed and print a JSON string to stdout.
    script = f"""
import pandas as pd
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import json
import sys

def run_portfolio_optimization():
    assets = {assets}
    portfolio_value = {portfolio_value}
    target_return = {target_return}
    risk_gamma = {risk_gamma} # Gamma parameter for L2 regularization

    # Fetch historical data
    try:
        # Limiting data range for quicker demo execution and to avoid very large files
        data = yf.download(assets, start="2018-01-01", end="2023-01-01")["Adj Close"]
    except Exception as e:
        return {{"error": str(e), "message": "Failed to download market data. Please check asset symbols or internet connection."}}

    if data.empty:
        return {{"error": "No data fetched", "message": "No historical data found for the specified assets."}}

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)

    # Optimize for maximal Sharpe ratio (a common approach for risk-adjusted returns)
    # For simplicity, we'll use max_sharpe and then check if it broadly meets target_return expectations
    # A more complex LLM could generate code for efficient_return directly with constraints.
    ef = EfficientFrontier(mu, S)
    
    try:
        weights = ef.max_sharpe(risk_free_rate=0.02) # Using a typical risk-free rate
    except Exception as e:
        return {{"error": str(e), "message": "Portfolio optimization failed. This could be due to ill-conditioned data or unachievable targets. Details: {{str(e)}}"}}

    cleaned_weights = ef.clean_weights()

    # Calculate portfolio performance with the optimized weights
    latest_returns, latest_volatility, _ = ef.portfolio_performance(verbose=False)

    results = {{
        "optimal_allocation": cleaned_weights,
        "projected_annual_return": latest_returns,
        "projected_annual_volatility": latest_volatility,
        "portfolio_value": portfolio_value
    }}
    return json.dumps(results)

if __name__ == '__main__':
    print(run_portfolio_optimization())
"""
    return script

# --- Step 2: Simulate external code execution ---
def execute_financial_script(script_code: str) -> dict:
    """
    Simulates the execution of a Python script in a sandboxed environment
    and captures its JSON output from stdout.

    WARNING: Executing arbitrary code from untrusted sources is a security risk.
    This function is for demonstration purposes only. In a production system,
    a secure, isolated execution environment (e.g., a containerized service)
    with strict resource limits and input validation would be essential.
    """
    old_stdout = io.StringIO()
    with contextlib.redirect_stdout(old_stdout):
        try:
            # The script is expected to print a JSON string.
            # We use a restricted global and local dictionary for `exec`
            # to limit what the executed script can access.
            exec(script_code, {"__builtins__": None, "pd": pd, "yf": yf, 
                               "EfficientFrontier": EfficientFrontier, "risk_models": risk_models,
                               "expected_returns": expected_returns, "json": json, "sys": sys}, {})
            output = old_stdout.getvalue()
            # Try to parse the output as JSON
            return json.loads(output)
        except json.JSONDecodeError as e:
            return {"error": "JSON_DECODE_ERROR", "message": f"Script output was not valid JSON: {e}. Raw output: {output}"}
        except Exception as e:
            return {"error": "SCRIPT_EXECUTION_ERROR", "message": f"Script execution failed: {e}"}

# --- Step 3: Simulate LLM's response generation ---
def format_llm_response(optimization_results: dict) -> str:
    """
    Simulates an LLM generating a natural language response based on the
    numerical output from the financial script.
    """
    if optimization_results.get("error"):
        return f"I encountered an error during financial analysis: {optimization_results['message']}. Please check the input or try again."

    portfolio_value = optimization_results.get("portfolio_value", 0)
    allocations = optimization_results.get("optimal_allocation", {})
    annual_return = optimization_results.get("projected_annual_return", 0)
    annual_volatility = optimization_results.get("projected_annual_volatility", 0)

    response_parts = [
        f"Based on your request for a ${portfolio_value:,.2f} portfolio, here is the recommended asset allocation:"
    ]

    if not allocations:
        response_parts.append("Unfortunately, I couldn't determine an optimal allocation with the given parameters. Please try again with different constraints or a broader asset selection.")
        return "\n".join(response_parts)

    # Filter out assets with negligible weights and format percentages
    active_allocations = {asset: weight for asset, weight in allocations.items() if weight > 0.005} # > 0.5%
    if not active_allocations:
        response_parts.append("The optimization resulted in no significant asset allocations. This might happen with very restrictive conditions or unsuitable assets.")
    else:
        for asset, weight in active_allocations.items():
            response_parts.append(f"- {asset}: {weight:.2%}")

    response_parts.append(
        f"\nThis portfolio is projected to yield an annual return of approximately {annual_return:.2%} " # Two decimal places for percentage
        f"with an annual volatility of {annual_volatility:.2%}. "
        "Please remember that past performance is not indicative of future results, and this advice is for informational purposes only. Consult with a professional financial advisor before making investment decisions."
    )

    return "\n".join(response_parts)

# --- Main application flow ---
def main():
    # Example user query for the financial advisor
    user_query = "What's the optimal asset allocation for my $150,000 investment with a high risk tolerance targeting a 10% annual return?"
    print(f"User Query: {user_query}\n")

    print("Step 1: LLM (simulated) generates a Python script for financial computation...")
    financial_script = generate_optimization_script(user_query)
    # Optionally, print the generated script to see what the LLM produced:
    # print("\n--- Generated Script ---\n")
    # print(financial_script)
    # print("\n------------------------\n")

    print("Step 2: External interpreter (simulated) executes the generated script...")
    optimization_results = execute_financial_script(financial_script)
    # Optionally, print the raw output from the script execution:
    # print("\n--- Script Execution Output ---\n")
    # print(json.dumps(optimization_results, indent=2))
    # print("\n-------------------------------\n")

    print("Step 3: LLM (simulated) formats the numerical results into a natural language response...")
    final_response = format_llm_response(optimization_results)
    print("\n--- Final LLM Response ---\n")
    print(final_response)
    print("\n--------------------------\n")

if __name__ == '__main__':
    main()
