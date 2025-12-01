import streamlit as st
import subprocess
import tempfile
import os
import json
import numpy as np
from scipy.optimize import minimize

# --- Simulated LLM Orchestrator: Code Generation --- 
# In a real application, an LLM would generate this code dynamically.
# For this example, we provide a template and inject user-defined assets.
def generate_financial_script(asset_names, investment_goal):
    # Dummy data for expected returns and covariance matrix
    # In a real scenario, this would come from historical data fetched by the LLM or a tool.
    num_assets = len(asset_names)
    # Create somewhat diverse but consistent dummy data
    base_returns = np.array([0.08 + i * 0.02 for i in range(num_assets)])
    base_volatility = np.array([0.15 + i * 0.01 for i in range(num_assets)])

    # Generate a simple covariance matrix (diagonal with some off-diagonal for correlation)
    cov_matrix = np.diag(base_volatility**2)
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            # Introduce some correlation
            correlation = 0.3 # Example correlation
            cov_val = correlation * base_volatility[i] * base_volatility[j]
            cov_matrix[i, j] = cov_val
            cov_matrix[j, i] = cov_val

    # Convert numpy arrays to lists for embedding in the script string
    expected_returns_list = base_returns.tolist()
    cov_matrix_list = cov_matrix.tolist()

    script_template = f"""
import numpy as np
from scipy.optimize import minimize
import json

asset_names = {json.dumps(asset_names)}
expected_returns = np.array({expected_returns_list})
cov_matrix = np.array({cov_matrix_list})

num_assets = len(asset_names)

def portfolio_return(weights):
    return np.sum(weights * expected_returns)

def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def neg_sharpe_ratio(weights):
    risk_free_rate = 0.02 # Example risk-free rate
    return -(portfolio_return(weights) - risk_free_rate) / portfolio_volatility(weights)

def minimize_volatility(weights):
    return portfolio_volatility(weights)

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(num_assets))
initial_weights = num_assets * [1. / num_assets,]

results = {{}}

if "maximize_sharpe" in "{investment_goal}".lower():
    sharpe_results = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_sharpe_weights = sharpe_results.x
    optimal_sharpe_return = portfolio_return(optimal_sharpe_weights)
    optimal_sharpe_volatility = portfolio_volatility(optimal_sharpe_weights)
    
    results["sharpe_portfolio"] = {{
        "weights": {{asset_names[i]: float(optimal_sharpe_weights[i]) for i in range(num_assets)}},
        "expected_return": float(optimal_sharpe_return),
        "volatility": float(optimal_sharpe_volatility)
    }}

if "minimize_risk" in "{investment_goal}".lower() or "minimize_volatility" in "{investment_goal}".lower():
    min_vol_results = minimize(minimize_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_min_vol_weights = min_vol_results.x
    optimal_min_vol_return = portfolio_return(optimal_min_vol_weights)
    optimal_min_vol_volatility = portfolio_volatility(optimal_min_vol_weights)

    results["min_volatility_portfolio"] = {{
        "weights": {{asset_names[i]: float(optimal_min_vol_weights[i]) for i in range(num_assets)}},
        "expected_return": float(optimal_min_vol_return),
        "volatility": float(optimal_min_vol_volatility)
    }}

print(json.dumps(results))
"""
    return script_template

# --- Code Execution Environment --- 
def execute_python_code(code_string):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(code_string)
        temp_file_path = temp_file.name
    
    try:
        # Execute the script using a subprocess
        process = subprocess.run(['python', temp_file_path], capture_output=True, text=True, check=True)
        return process.stdout
    except subprocess.CalledProcessError as e:
        return f"Error during script execution: {e.stderr}"
    finally:
        os.remove(temp_file_path)

# --- Simulated LLM Orchestrator: Result Interpretation ---
def interpret_results(raw_output, investment_goal):
    try:
        results = json.loads(raw_output)
        interpretation = []

        if "sharpe_portfolio" in results:
            sharpe_port = results["sharpe_portfolio"]
            interpretation.append(f"### Portfolio for Maximizing Sharpe Ratio (Risk-Adjusted Return):")
            interpretation.append(f"- **Expected Return:** {sharpe_port['expected_return']:.2%}")
            interpretation.append(f"- **Volatility (Risk):** {sharpe_port['volatility']:.2%}")
            interpretation.append(f"- **Optimal Weights:**")
            for asset, weight in sharpe_port['weights'].items():
                interpretation.append(f"    - {asset}: {weight:.2%}")
            interpretation.append("")

        if "min_volatility_portfolio" in results:
            min_vol_port = results["min_volatility_portfolio"]
            interpretation.append(f"### Portfolio for Minimum Volatility (Lowest Risk):")
            interpretation.append(f"- **Expected Return:** {min_vol_port['expected_return']:.2%}")
            interpretation.append(f"- **Volatility (Risk):** {min_vol_port['volatility']:.2%}")
            interpretation.append(f"- **Optimal Weights:**")
            for asset, weight in min_vol_port['weights'].items():
                interpretation.append(f"    - {asset}: {weight:.2%}")
            interpretation.append("")

        if not interpretation:
            interpretation.append("No specific optimization results found based on your goal. Try 'maximize sharpe' or 'minimize risk'.")

        return "\n".join(interpretation)

    except json.JSONDecodeError:
        return f"Error parsing results from calculation script: {raw_output}"
    except Exception as e:
        return f"An error occurred during result interpretation: {e}"


# --- Streamlit UI --- 
st.set_page_config(layout="wide", page_title="PAL Portfolio Optimizer")

st.title("📈 Program-Aided Language Model (PAL) Financial Portfolio Optimizer")
st.markdown("--- Generates and executes Python code for portfolio optimization based on your goals ---")

st.header("Your Investment Portfolio Details")

# Input for assets
asset_input = st.text_area(
    "Enter your asset names, one per line (e.g., Apple, Google, Microsoft)",
    "Apple Inc.\nGoogle LLC\nMicrosoft Corp."
)
asset_names_list = [name.strip() for name in asset_input.split('\n') if name.strip()]

# Input for investment goal
investment_goal = st.selectbox(
    "Select your primary investment goal:",
    ["Maximize Sharpe Ratio (Risk-Adjusted Return)", "Minimize Portfolio Risk (Volatility)", "Custom Goal (e.g., 'achieve 10% return')"]
)

if "Custom Goal" in investment_goal:
    custom_goal_text = st.text_input(
        "Describe your custom investment goal (e.g., 'achieve 10% return with moderate risk')",
        "maximize sharpe"
    )
    actual_goal_for_llm = custom_goal_text
elif "Maximize Sharpe" in investment_goal:
    actual_goal_for_llm = "maximize sharpe"
elif "Minimize Portfolio Risk" in investment_goal:
    actual_goal_for_llm = "minimize risk"
else:
    actual_goal_for_llm = investment_goal # Fallback, though selectbox should handle this

st.subheader("Run Optimization")
if st.button("Optimize Portfolio"):
    if not asset_names_list:
        st.error("Please enter at least one asset name.")
    else:
        st.info(f"Optimizing portfolio for {len(asset_names_list)} assets with goal: '{investment_goal}'...")

        # 1. LLM Orchestrator: Generate Code
        generated_code = generate_financial_script(asset_names_list, actual_goal_for_llm)
        st.subheader("Generated Python Code (by LLM)")
        st.code(generated_code, language='python')

        # 2. Code Execution Environment: Execute Code
        st.subheader("Executing Code...")
        calculation_output = execute_python_code(generated_code)

        if "Error during script execution" in calculation_output:
            st.error(f"An error occurred during financial calculations: {calculation_output}")
        else:
            st.success("Financial calculations completed successfully!")
            st.subheader("Raw Calculation Output")
            st.code(calculation_output, language='json')

            # 3. LLM Orchestrator: Interpret Results
            st.subheader("LLM Interpretation & Recommendations")
            final_recommendations = interpret_results(calculation_output, actual_goal_for_llm)
            st.markdown(final_recommendations)

st.markdown("""
--- 
**Note:** This is a simulated demonstration of the Program-Aided Language Model (PAL) pattern. 
In a real application, the 'Generated Python Code' would be produced by an actual LLM 
and the financial models would be more sophisticated (e.g., fetching real-time data).
""")
