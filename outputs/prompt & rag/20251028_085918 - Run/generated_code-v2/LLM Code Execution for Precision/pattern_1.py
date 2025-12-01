import pandas as pd
import numpy as np
import io

def get_mock_market_data():
    """Generates mock market data for demonstration purposes."""
    dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=252, freq='B')) # Approx 1 year of business days
    data = {
        'AAPL': np.random.rand(252) * 100 + 150,
        'MSFT': np.random.rand(252) * 80 + 200,
        'GOOG': np.random.rand(252) * 120 + 100,
        'AMZN': np.random.rand(252) * 50 + 120
    }
    df = pd.DataFrame(data, index=dates)
    return df

def optimize_portfolio_simple(market_data, user_risk_tolerance):
    """Simulates a simple portfolio optimization based on mock data and risk tolerance.
    In a real system, this would involve complex financial models.
    """
    returns = market_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_assets = len(mean_returns)
    
    # Example: Simple allocation strategy based on risk tolerance
    # Higher risk tolerance means more allocated to 'higher' expected return assets
    # This is a very simplistic heuristic for demonstration.
    if user_risk_tolerance <= 0.3: # Low risk
        weights = np.array([0.3, 0.3, 0.2, 0.2]) # More balanced/conservative
    elif user_risk_tolerance <= 0.7: # Medium risk
        weights = np.array([0.25, 0.25, 0.25, 0.25]) # Equal weighting
    else: # High risk
        weights = np.array([0.15, 0.15, 0.35, 0.35]) # More aggressive (towards GOOG/AMZN assumed higher growth)

    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)

    portfolio_return = np.sum(weights * mean_returns) * 252 # Annualized return
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) # Annualized volatility

    return {
        "portfolio_return": portfolio_return,
        "portfolio_std_dev": portfolio_std_dev,
        "assets": list(market_data.columns),
        "weights": {asset: round(weight, 4) for asset, weight in zip(market_data.columns, weights)}
    }

def simulate_llm_code_generation(user_financial_goal, user_risk_tolerance_str):
    """Simulates an LLM generating Python code based on user input.
    In a real application, an actual LLM would generate this string.
    """
    # A simple mapping for demonstration. An LLM would parse natural language.
    risk_map = {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8
    }
    numeric_risk_tolerance = risk_map.get(user_risk_tolerance_str.lower(), 0.5)

    generated_code = f"""
import pandas as pd
import numpy as np

# --- BEGIN GENERATED CODE --- #

def _get_market_data_for_execution():
    # This data would typically be fetched from an API in a real scenario
    # For this simulation, we use the mock data function defined in main context
    return get_mock_market_data()

def _run_optimization_for_llm(market_data, risk_tol):
    # This calls the pre-defined optimization logic in the main context
    return optimize_portfolio_simple(market_data, risk_tol)

market_data_for_opt = _get_market_data_for_execution()
optimization_results = _run_optimization_for_llm(market_data_for_opt, {numeric_risk_tolerance})

# The LLM expects a 'results' variable to be available after execution
results = optimization_results
# --- END GENERATED CODE --- #
"""
    return generated_code

def execute_generated_code(code_string, global_context):
    """Executes the generated Python code in a controlled environment
    and captures its output (specifically the 'results' variable).
    """
    local_vars = {}
    # Inject necessary functions into the execution context
    exec(code_string, global_context, local_vars)
    return local_vars.get('results')

def simulate_llm_recommendation_formulation(optimization_results, user_financial_goal, user_risk_tolerance_str):
    """Simulates an LLM formulating a natural language recommendation
    based on optimization results and original user input.
    """
    portfolio_return_annual = optimization_results['portfolio_return'] * 100
    portfolio_std_dev_annual = optimization_results['portfolio_std_dev'] * 100
    assets_weights = ', '.join([f" {asset}: {weight*100:.2f}%" for asset, weight in optimization_results['weights'].items()])

    recommendation = (
        f"Based on your goal of '{user_financial_goal}' and a {user_risk_tolerance_str} risk tolerance, "
        f"I recommend the following portfolio allocation:\n"
        f"  {assets_weights}\n"
        f"This portfolio is projected to have an annualized return of approximately "
        f"{portfolio_return_annual:.2f}% with an annualized volatility of {portfolio_std_dev_annual:.2f}%. "
        f"Please remember that past performance is not indicative of future results, and this is a simplified model."
    )
    return recommendation

if __name__ == "__main__":
    print("\n--- Financial Portfolio Optimization and Recommendation System ---\n")

    # 1. User Input (simulated)
    user_goal = input("What are your financial goals? (e.g., 'long-term growth', 'capital preservation'): ")
    user_risk = input("What is your risk tolerance? (e.g., 'low', 'medium', 'high'): ")

    print(f"\nUser Input: Goal = '{user_goal}', Risk = '{user_risk}'")

    # 2. LLM Generates Code
    print("\n--- LLM Generating Code for Financial Calculations ---")
    generated_python_code = simulate_llm_code_generation(user_goal, user_risk)
    print("Generated Code:\n" + generated_python_code)

    # 3. Execute Generated Code via External Interpreter
    print("\n--- Executing Generated Code ---")
    # We pass the current globals so that get_mock_market_data and optimize_portfolio_simple are available
    # in the scope where the generated code runs.
    execution_context = globals().copy()
    optimization_output = execute_generated_code(generated_python_code, execution_context)

    if optimization_output:
        print("Optimization Results (from executed code):\n", optimization_output)
        # 4. LLM Formulates Recommendation
        print("\n--- LLM Formulating Natural Language Recommendation ---")
        final_recommendation = simulate_llm_recommendation_formulation(optimization_output, user_goal, user_risk)
        print("\nFinal Portfolio Recommendation:\n")
        print(final_recommendation)
    else:
        print("Error: Could not retrieve optimization results.")

    print("\n------------------------------------------------------")