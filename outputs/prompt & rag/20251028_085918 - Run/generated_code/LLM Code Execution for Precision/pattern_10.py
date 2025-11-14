import sys
import io
import numpy as np

def simulate_llm_code_generation(user_preferences: dict, market_data: dict) -> str:
    """
    Simulates an LLM generating Python code for financial portfolio optimization.
    For this demo, it returns a predefined script.
    """
    # In a real scenario, the LLM would dynamically generate this code
    # based on user_preferences, market_data, and sophisticated financial models.
    # Here, we use a simplified Monte Carlo simulation for demonstration.
    num_assets = len(market_data['assets'])
    asset_returns_str = f"[{', '.join(map(str, market_data['returns']))}]"
    asset_volatilities_str = f"[{', '.join(map(str, market_data['volatilities']))}]"
    
    generated_code = f"""
import numpy as np

# --- Simulated Portfolio Optimization Code ---
# This code simulates a Monte Carlo approach to find an 'optimal' portfolio
# based on basic expected returns and volatilities.

# Input parameters (simulated from LLM's understanding of market_data)
expected_returns = np.array({asset_returns_str})
expected_volatilities = np.array({asset_volatilities_str})
num_assets = {num_assets}
num_portfolios = 10000 # Number of random portfolios to generate

results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    weights_record.append(weights)

    portfolio_return = np.sum(expected_returns * weights)
    # Simplified portfolio volatility (not using covariance matrix for simplicity)
    # In a real scenario, a covariance matrix would be crucial.
    portfolio_volatility = np.sqrt(np.sum(weights**2 * expected_volatilities**2))
    
    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = portfolio_return / portfolio_volatility # Simple Sharpe ratio proxy

# Find the portfolio with the highest 'Sharpe Ratio proxy' for this demo
max_sharpe_idx = np.argmax(results[2])
optimal_weights = weights_record[max_sharpe_idx]
optimal_return = results[0, max_sharpe_idx]
optimal_volatility = results[1, max_sharpe_idx]

print(f"Optimal Weights: {{list(np.round(optimal_weights, 4))}}")
print(f"Expected Return: {{optimal_return:.4f}}")
print(f"Expected Volatility: {{optimal_volatility:.4f}}")
# --- End of Simulated Portfolio Optimization Code ---
    """
    return generated_code

def execute_generated_code(code_string: str) -> str:
    """
    Executes the given Python code string and captures its stdout.
    """
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        # It's important to be cautious with exec() in a real application
        # as it can pose security risks. A sandboxed environment would be preferred.
        exec(code_string, {'np': np}) # Pass numpy to the executed code's global scope
    except Exception as e:
        return f"Error during code execution: {e}"
    finally:
        sys.stdout = old_stdout # Restore stdout
    return redirected_output.getvalue()

def simulate_llm_interpretation(calculation_results: str) -> str:
    """
    Simulates an LLM interpreting the numerical results from the executed code
    and generating natural language recommendations.
    """
    # In a real scenario, an LLM would process the results more intelligently.
    # Here, we parse the specific output format from our demo code.
    
    optimal_weights = []
    expected_return = 0.0
    expected_volatility = 0.0

    for line in calculation_results.split('\n'):
        if "Optimal Weights:" in line:
            try:
                # Extract list of weights, e.g., '[0.25, 0.35, 0.40]'
                weights_str = line.split("Optimal Weights:")[1].strip()
                optimal_weights = eval(weights_str) # Using eval for parsing list string
            except Exception as e:
                print(f"Error parsing weights: {e}")
        elif "Expected Return:" in line:
            try:
                expected_return = float(line.split("Expected Return:")[1].strip())
            except Exception as e:
                print(f"Error parsing return: {e}")
        elif "Expected Volatility:" in line:
            try:
                expected_volatility = float(line.split("Expected Volatility:")[1].strip())
            except Exception as e:
                print(f"Error parsing volatility: {e}")
    
    if not optimal_weights and expected_return == 0.0 and expected_volatility == 0.0:
        return f"Could not interpret financial results. Raw output: \n{calculation_results}"

    recommendations = []
    recommendations.append(f"Based on our analysis, the recommended asset allocation is as follows:")
    for i, weight in enumerate(optimal_weights):
        recommendations.append(f"  - Asset {i+1}: {weight:.2%} (representing {weight*100:.2f}% of your portfolio)")
    
    recommendations.append(f"This optimized portfolio is projected to have an annual expected return of {expected_return:.2%} ")
    recommendations.append(f"with an expected annual volatility (risk) of {expected_volatility:.2%}.")
    
    if expected_return / expected_volatility > 1.5: # Simple heuristic for good performance
        recommendations.append("This indicates a relatively strong risk-adjusted return profile.")
    elif expected_return / expected_volatility > 0.5:
        recommendations.append("This indicates a balanced risk-adjusted return profile.")
    else:
        recommendations.append("You might consider reviewing your risk tolerance or exploring alternative investments if this profile doesn't meet your goals.")
        
    recommendations.append("\nRemember that past performance is not indicative of future results, and market conditions can change rapidly.")
    recommendations.append("It is always advisable to consult with a professional financial advisor before making investment decisions.")
    
    return "\n".join(recommendations)

def main():
    print("--- AI-powered Financial Portfolio Optimizer (CAR Pattern Demo) ---")

    # 1. Simulate User Input and Market Data
    user_preferences = {
        "risk_tolerance": "medium",
        "investment_horizon_years": 5,
        "investment_goal": "growth"
    }
    market_data = {
        "assets": ["Stock A", "Stock B", "Bond C"],
        "returns": [0.10, 0.15, 0.04], # Example expected annual returns
        "volatilities": [0.20, 0.30, 0.05], # Example annual volatilities
        "correlation_matrix": [] # Placeholder for more complex models
    }
    print("\nSimulated User Preferences:", user_preferences)
    print("Simulated Market Data (Simplified):")
    for i, asset in enumerate(market_data['assets']):
        print(f"  - {asset}: Return={market_data['returns'][i]:.2%}, Volatility={market_data['volatilities'][i]:.2%}")

    # 2. LLM Generates Code for Calculation
    print("\n--- LLM Generating Code ---")
    optimization_code = simulate_llm_code_generation(user_preferences, market_data)
    # print("\nGenerated Python Code:\n" + optimization_code)

    # 3. Execute the Generated Code
    print("\n--- Executing Generated Code ---")
    calculation_results = execute_generated_code(optimization_code)
    print("Raw Calculation Results from Code Execution:\n" + calculation_results.strip())

    # 4. LLM Interprets Results and Provides Recommendations
    print("\n--- LLM Interpreting Results and Formulating Recommendations ---")
    final_recommendations = simulate_llm_interpretation(calculation_results)
    print("\n--- Final Financial Recommendations ---")
    print(final_recommendations)

if __name__ == "__main__":
    main()
