import streamlit as st
import pandas as pd
import numpy as np
import io
import contextlib

# 1. LLM Simulation (Generates Code)
def simulate_llm_code_generation(user_prompt, assets, weights):
    # In a real scenario, an LLM would generate this code based on user_prompt
    # Here, we're hardcoding a simple portfolio calculation for demonstration.
    # The LLM's role would be to interpret user intent and create the *correct* code.

    asset_list = ', '.join([f"'{a}'" for a in assets])
    weights_list = ', '.join([str(w) for w in weights])

    code = f"""
import numpy as np
import pandas as pd

def calculate_portfolio_metrics(asset_names, weights):
    # Placeholder for actual data fetching/simulation
    # For demonstration, we'll use dummy expected returns and covariance matrix
    # In a real app, 'yfinance' or a similar library would fetch historical data
    # and calculate these.

    num_assets = len(asset_names)

    # Dummy expected annual returns (e.g., from historical data or expert estimates)
    # Ensure this aligns with the number of assets
    expected_returns_data = {{
        'AssetA': 0.10, 'AssetB': 0.15, 'AssetC': 0.08
    }} # Example data, extend as needed

    # Dummy annual covariance matrix
    # Ensure this is a symmetric positive semi-definite matrix
    # For simplicity, we'll create a diagonal matrix for uncorrelated assets,
    # or a slightly correlated one if we have more assets.
    # The size must be num_assets x num_assets
    if num_assets == 3:
        covariance_matrix_data = pd.DataFrame([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.003],
            [0.005, 0.003, 0.0225]
        ], index=asset_names, columns=asset_names)
    else: # Fallback for different number of assets
        # Simple diagonal covariance for demonstration
        std_devs = np.array([0.2, 0.3, 0.15][:num_assets]) # Example std devs
        covariance_matrix_data = np.diag(std_devs**2)
        covariance_matrix_data = pd.DataFrame(covariance_matrix_data, index=asset_names, columns=asset_names)

    # Map asset names to their dummy expected returns and covariance
    expected_returns = np.array([expected_returns_data.get(asset, 0.05) for asset in asset_names])
    weights_arr = np.array(weights)

    # Ensure weights sum to 1
    weights_arr = weights_arr / np.sum(weights_arr)

    portfolio_return = np.sum(expected_returns * weights_arr)

    # Ensure covariance_matrix_data is a numpy array for dot product
    covariance_matrix_np = covariance_matrix_data.to_numpy()

    portfolio_variance = np.dot(weights_arr.T, np.dot(covariance_matrix_np, weights_arr))
    portfolio_std_dev = np.sqrt(portfolio_variance)

    return portfolio_return, portfolio_std_dev

# Example Usage:
# asset_names = [{asset_list}]
# weights = [{weights_list}]
# port_ret, port_std = calculate_portfolio_metrics(asset_names, weights)
# print(f"Portfolio Expected Return: {{port_ret:.2%}}")
# print(f"Portfolio Standard Deviation: {{port_std:.2%}}")
"""
    return code

# 2. Code Execution (Safe Execution)
@contextlib.contextmanager
def stdout_redirector(stream):
    import sys
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout

def execute_generated_code(code_string, assets, weights):
    output_buffer = io.StringIO()
    try:
        # Create a dictionary to hold the execution environment
        exec_globals = {}
        exec_locals = {}

        # Execute the function definition
        exec(code_string, exec_globals, exec_locals)

        # Now call the function from the executed code
        # Ensure the function is in the exec_globals or exec_locals
        if 'calculate_portfolio_metrics' in exec_locals:
            calculate_portfolio_metrics_func = exec_locals['calculate_portfolio_metrics']
        elif 'calculate_portfolio_metrics' in exec_globals:
            calculate_portfolio_metrics_func = exec_globals['calculate_portfolio_metrics']
        else:
            raise ValueError("The generated code did not define 'calculate_portfolio_metrics' function.")

        with stdout_redirector(output_buffer):
            port_ret, port_std = calculate_portfolio_metrics_func(assets, weights)
            print(f"Portfolio Expected Return: {port_ret:.2%}")
            print(f"Portfolio Standard Deviation: {port_std:.2%}")

        return output_buffer.getvalue(), None
    except Exception as e:
        return output_buffer.getvalue(), str(e)

# 3. Streamlit UI
st.title("💰 Financial Portfolio Optimization Assistant (PAL Prompting Demo)")
st.markdown("""
This application demonstrates the **Program-Aided Language Models (PAL) Prompting** pattern.
An LLM (simulated here) generates Python code for financial calculations, which is then executed,
and the results are used to provide insights.
""")

st.header("Define Your Portfolio")

# Example assets and weights for initial population
default_assets = ["AssetA", "AssetB", "AssetC"]
default_weights = [0.4, 0.3, 0.3]

num_assets = st.number_input("Number of Assets in Portfolio", min_value=1, value=len(default_assets), step=1)

assets_input = []
weights_input = []

for i in range(int(num_assets)):
    col1, col2 = st.columns(2)
    with col1:
        asset_name = st.text_input(f"Asset {i+1} Name", value=default_assets[i] if i < len(default_assets) else f"Asset{i+1}")
        assets_input.append(asset_name)
    with col2:
        weight = st.number_input(f"Weight for {asset_name}", min_value=0.0, max_value=1.0, value=default_weights[i] if i < len(default_weights) else 0.0, step=0.01, format="%.2f")
        weights_input.append(weight)

# Normalize weights
sum_weights = sum(weights_input)
if sum_weights > 0:
    weights_input_normalized = [w / sum_weights for w in weights_input]
    st.info(f"Weights normalized to sum to 1: {', '.join([f'{w:.2f}' for w in weights_input_normalized])}")
else:
    weights_input_normalized = [0.0] * len(weights_input)
    st.warning("Please assign some weight to your assets.")


if st.button("Optimize Portfolio"):
    if not assets_input or not weights_input_normalized:
        st.error("Please provide asset names and weights.")
    else:
        st.subheader("1. LLM Generates Code (Simulated)")
        user_prompt = "Calculate the expected return and standard deviation for my portfolio."
        generated_code = simulate_llm_code_generation(user_prompt, assets_input, weights_input_normalized)
        st.code(generated_code, language="python")

        st.subheader("2. Executing Generated Code")
        with st.spinner("Executing financial calculations..."):
            execution_output, error = execute_generated_code(generated_code, assets_input, weights_input_normalized)

            if error:
                st.error(f"Error during code execution: {error}")
                st.code(execution_output)
            else:
                st.success("Code executed successfully!")
                st.code(execution_output)

        st.subheader("3. LLM Integrates Results & Provides Recommendations (Simulated)")
        # In a real PAL setup, the LLM would now take 'execution_output'
        # and formulate a natural language response.
        st.markdown(f"""
        Based on the precise calculations:
        - {execution_output.strip()}

        **LLM's Recommendation (Simulated):**
        "Given these metrics, your portfolio demonstrates an expected return and risk profile as calculated.
        For a more aggressive strategy, consider reallocating towards assets with higher expected returns,
        while carefully managing the covariance. For a conservative approach, prioritize assets with lower
        volatility and explore diversification benefits. Remember to periodically rebalance your portfolio
        to maintain your desired risk-return trade-off."
        """
        )
