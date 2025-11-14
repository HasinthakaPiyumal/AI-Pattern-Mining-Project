import pandas as pd
import numpy as np
from scipy.optimize import minimize
import io
import contextlib

# 1. Simulated LLM: Generates Python code for financial calculations
def generate_financial_code(user_data):
    # In a real application, this would be an actual LLM call.
    # For this example, we'll return a predefined string of Python code.
    # The code will take 'portfolio_data' and 'risk_free_rate' as inputs
    # and calculate optimal weights and portfolio metrics.

    code_template = """
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# User-provided data (simulated from LLM's understanding of user input)
assets = {user_data['asset_data']} # Example: {'AAPL': {'returns': [0.01, 0.02, 0.005]}, 'MSFT': {'returns': [0.015, 0.01, 0.02]}}
risk_free_rate = {user_data['risk_free_rate']} # Example: 0.02

# --- Start of Generated Financial Calculation Logic ---

# Convert asset data to a DataFrame of returns
data = {{asset: item['returns'] for asset, item in assets.items()}}
returns_df = pd.DataFrame(data)

# Calculate daily returns (if not already provided as such)
# For simplicity, assuming 'returns' are already daily or period returns
# If raw prices were given, we'd do: returns_df = prices_df.pct_change().dropna()

num_assets = len(returns_df.columns)
mean_returns = returns_df.mean()
cov_matrix = returns_df.cov()

# Function to calculate portfolio volatility
def portfolio_volatility(weights, mean_returns, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) # Annualized

# Function to calculate portfolio return
def portfolio_return(weights, mean_returns):
    return np.sum(mean_returns * weights) * 252 # Annualized

# Objective function for Markowitz optimization (minimize negative Sharpe Ratio)
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret = portfolio_return(weights, mean_returns)
    p_vol = portfolio_volatility(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_vol

# Constraints for optimization: weights sum to 1
constraints = ({{"type': 'eq', 'fun': lambda x: np.sum(x) - 1}})

# Bounds for optimization: each weight between 0 and 1
bounds = tuple((0, 1) for asset in range(num_assets))

# Initial guess (equal weighting)
init_guess = num_assets * [1. / num_assets,]

# Perform optimization to find the portfolio with the maximum Sharpe Ratio
optimized_results = minimize(negative_sharpe_ratio, init_guess, args=(mean_returns, cov_matrix, risk_free_rate,),
                            method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = optimized_results.x
max_sharpe_return = portfolio_return(optimal_weights, mean_returns)
max_sharpe_volatility = portfolio_volatility(optimal_weights, mean_returns, cov_matrix)
max_sharpe = (max_sharpe_return - risk_free_rate) / max_sharpe_volatility

# Print results in a structured format for the LLM to consume
print("--- Optimal Portfolio Details ---")
print(f"Optimal Asset Weights: {{dict(zip(returns_df.columns, np.round(optimal_weights, 4)))}}")
print(f"Expected Annual Return: {{max_sharpe_return:.2%}}")
print(f"Expected Annual Volatility: {{max_sharpe_volatility:.2%}}")
print(f"Max Sharpe Ratio: {{max_sharpe:.2f}}")

# --- End of Generated Financial Calculation Logic ---
"""
    # Replace placeholders with actual user data (simplified for this example)
    asset_data_str = "{" + ", ".join([f"'{k}': {{'returns': {v}}}" for k,v in user_data['asset_data'].items()]) + "}"
    
    # Format the code string with user data
    formatted_code = code_template.format(
        user_data={
            'asset_data': asset_data_str,
            'risk_free_rate': user_data['risk_free_rate']
        }
    )
    return formatted_code

# 2. Code Execution Function
@contextlib.contextmanager
def stdout_redirect(buffer):
    import sys
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        yield
    finally:
        sys.stdout = old_stdout

def execute_python_code(code_string):
    output_buffer = io.StringIO()
    try:
        with stdout_redirect(output_buffer):
            exec(code_string)
        return output_buffer.getvalue()
    except Exception as e:
        return f"Error during code execution: {e}\n{output_buffer.getvalue()}"

# 3. Simulated LLM: Interprets results and generates a report
def interpret_financial_results(execution_output, user_goals):
    # In a real application, this would be an actual LLM call
    # to process the execution output and user_goals into a natural language report.
    # For this example, we'll do a basic parsing and formatting.

    report = f"--- Financial Portfolio Analysis Report ---\n\n"
    report += f"Based on your investment goals (e.g., {user_goals['goals']}) and risk tolerance (e.g., {user_goals['risk_tolerance']}), "
    report += f"we have performed an optimization using a Code-Assisted Reasoning approach.\n\n"

    # Extract key information from the execution_output
    if "Error" in execution_output:
        report += f"An error occurred during the financial calculations:\n{execution_output}\n"
        report += "Please review your inputs or consult a financial advisor."
        return report

    # Simple parsing (could be more robust with regex or specific print formats)
    output_lines = execution_output.split('\n')
    parsed_data = {}
    for line in output_lines:
        if "Optimal Asset Weights:" in line:
            weights_str = line.split("Optimal Asset Weights:")[1].strip()
            # Clean up string for eval
            weights_str = weights_str.replace("'", "\"") # Use double quotes for dict keys
            weights_str = weights_str.replace("{", "").replace("}", "")
            parsed_weights = {}
            for item in weights_str.split(", "):
                if ":" in item:
                    key, value = item.split(":", 1)
                    parsed_weights[key.strip().strip("'").strip('"')] = float(value.strip())
            parsed_data['Optimal Asset Weights'] = parsed_weights
        elif "Expected Annual Return:" in line:
            parsed_data['Expected Annual Return'] = line.split(":")[1].strip()
        elif "Expected Annual Volatility:" in line:
            parsed_data['Expected Annual Volatility'] = line.split(":")[1].strip()
        elif "Max Sharpe Ratio:" in line:
            parsed_data['Max Sharpe Ratio'] = line.split(":")[1].strip()

    if parsed_data:
        report += "--- Optimal Portfolio Recommendations ---\n"
        report += "The Code-Assisted Reasoning engine suggests the following optimal portfolio:\n\n"
        report += f"  Optimal Asset Allocation:\n"
        for asset, weight in parsed_data.get('Optimal Asset Weights', {}).items():
            report += f"    - {asset}: {weight:.2%}\n"
        report += f"\n  Projected Performance:\n"
        report += f"    - Expected Annual Return: {parsed_data.get('Expected Annual Return', 'N/A')}\n"
        report += f"    - Expected Annual Volatility: {parsed_data.get('Expected Annual Volatility', 'N/A')}\n"
        report += f"    - Max Sharpe Ratio: {parsed_data.get('Max Sharpe Ratio', 'N/A')}\n\n"
        report += "These recommendations aim to maximize your risk-adjusted returns based on the provided inputs.\n"
        report += "Please remember that past performance is not indicative of future results, and market conditions can change."
    else:
        report += "Could not parse detailed financial results from the code execution output. Raw output:\n"
        report += execution_output

    return report

# Main application function simulating the CAR pattern
def financial_car_tool(investment_goals, risk_tolerance, current_portfolio_holdings):
    print("Step 1: LLM analyzes user input and plans code generation.")
    # Simulate LLM extracting relevant data for the code generation
    user_data_for_code = {
        'asset_data': {
            'AAPL': [0.001, 0.002, -0.001, 0.003, 0.002], # Example daily returns
            'MSFT': [0.0015, 0.001, 0.002, 0.001, 0.0015],
            'GOOG': [0.0005, 0.0015, 0.001, 0.002, 0.0025]
        },
        'risk_free_rate': 0.015 # Example annual risk-free rate
    }

    # If current_portfolio_holdings were used, the LLM would likely transform this
    # into a format suitable for the financial code (e.g., historical returns data).
    # For this example, we use static example data for simplicity.

    print("Step 2: LLM generates Python code for precise financial calculations.")
    generated_code = generate_financial_code(user_data_for_code)
    # print("\n--- Generated Code ---\n", generated_code, "\n-----------------------\n") # For debugging

    print("Step 3: Executing the generated Python code...")
    execution_output = execute_python_code(generated_code)
    print("Step 4: Code execution complete. Output received by LLM for interpretation.")
    # print("\n--- Execution Output ---\n", execution_output, "\n-------------------------\n") # For debugging

    user_goals_summary = {
        'goals': investment_goals,
        'risk_tolerance': risk_tolerance
    }
    final_report = interpret_financial_results(execution_output, user_goals_summary)

    return final_report

# Example Usage:
if __name__ == "__main__":
    investment_goals = "long-term growth, moderate income"
    risk_tolerance = "medium"
    current_portfolio_holdings = {
        "AAPL": 100,
        "MSFT": 50,
        "GOOG": 20
    }

    report = financial_car_tool(investment_goals, risk_tolerance, current_portfolio_holdings)
    print("\n", report)

    print("\n--- Example with different asset data (simulating a new user input) ---")
    # Simulate changing user data (e.g., adding a new asset or different returns)
    def financial_car_tool_with_custom_data(investment_goals, risk_tolerance, current_portfolio_holdings, custom_asset_data, custom_risk_free_rate):
        print("Step 1: LLM analyzes user input and plans code generation.")
        user_data_for_code = {
            'asset_data': custom_asset_data,
            'risk_free_rate': custom_risk_free_rate
        }

        print("Step 2: LLM generates Python code for precise financial calculations.")
        generated_code = generate_financial_code(user_data_for_code)
        # print("\n--- Generated Code ---\n", generated_code, "\n-----------------------\n") # For debugging

        print("Step 3: Executing the generated Python code...")
        execution_output = execute_python_code(generated_code)
        print("Step 4: Code execution complete. Output received by LLM for interpretation.")
        # print("\n--- Execution Output ---\n", execution_output, "\n-------------------------\n") # For debugging

        user_goals_summary = {
            'goals': investment_goals,
            'risk_tolerance': risk_tolerance
        }
        final_report = interpret_financial_results(execution_output, user_goals_summary)
        return final_report

    custom_assets = {
        'TSLA': [0.005, -0.003, 0.008, 0.001, -0.002],
        'AMZN': [0.002, 0.003, 0.001, 0.004, 0.003],
        'NFLX': [0.001, 0.004, -0.002, 0.005, 0.001]
    }
    custom_risk_free = 0.02

    report_custom = financial_car_tool_with_custom_data(
        "aggressive growth",
        "high",
        {"TSLA": 50, "AMZN": 30, "NFLX": 20},
        custom_assets,
        custom_risk_free
    )
    print("\n", report_custom)