
import streamlit as st
import numpy as np
import pandas as pd
import scipy.optimize as sco
import io
import contextlib
import sys

# --- LLM Simulation Functions ---

def simulate_llm_code_generation(assets_list):
    """
    Simulates an LLM generating Python code for mean-variance portfolio optimization.
    The generated code uses synthetic historical data for demonstration purposes.
    """
    assets_str = ', '.join([f'"{asset.strip()}"' for asset in assets_list])
    num_assets = len(assets_list)

    # Generate synthetic historical data for demonstration
    # In a real scenario, this data would come from a database or API
    code = f"""
import numpy as np
import pandas as pd
import scipy.optimize as sco

# User-defined assets
assets = [{assets_str}]
num_assets = len(assets)

# --- Synthetic Historical Data Generation ---
# For a real application, this would load actual historical price data.
# We're creating 252 daily returns (approx 1 year)
np.random.seed(42) # for reproducibility
returns = np.random.randn(252, num_assets)
returns = pd.DataFrame(returns, columns=assets)

# Assume daily returns, convert to annual
daily_mean_returns = returns.mean()
daily_cov_matrix = returns.cov()

# Annualized returns and covariance
annual_mean_returns = daily_mean_returns * 252
annual_cov_matrix = daily_cov_matrix * 252

# --- Portfolio Optimization Functions ---

def portfolio_return(weights, mean_returns):
    return np.sum(mean_returns * weights) * 252 # Annualized

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) # Annualized

def min_func_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_ret = portfolio_return(weights, mean_returns / 252) # Use daily mean for calculation then annualize
    p_vol = portfolio_volatility(weights, cov_matrix / 252) # Use daily cov for calculation then annualize
    if p_vol == 0: return 1e10 # Avoid division by zero
    return -(p_ret - risk_free_rate) / p_vol

# --- Optimization Execution ---
num_portfolios = 10000 # Number of random portfolios to generate for visualization (optional)

def optimize_portfolio():
    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for x in range(num_assets))

    # Initial guess for weights (equal distribution)
    initial_weights = num_assets * [1. / num_assets,]

    # Optimize for maximum Sharpe Ratio
    optimized_results = sco.minimize(
        min_func_sharpe,
        initial_weights,
        args=(daily_mean_returns, daily_cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimized_weights = optimized_results['x']
    opt_return = portfolio_return(optimized_weights, daily_mean_returns)
    opt_volatility = portfolio_volatility(optimized_weights, daily_cov_matrix)
    opt_sharpe = (opt_return - 0.01) / opt_volatility # Assuming 1% risk-free rate

    results = {
        'weights': optimized_weights.tolist(),
        'assets': assets,
        'return': opt_return,
        'volatility': opt_volatility,
        'sharpe_ratio': opt_sharpe
    }
    print("OPTIMIZATION_RESULTS:" + str(results))

optimize_portfolio()
"""
    return code

def simulate_llm_report_generation(optimization_results):
    """
    Simulates an LLM interpreting the numerical results and generating a human-readable report.
    """
    if not optimization_results:
        return "Unable to generate report due to missing optimization results."

    weights_str = []
    for asset, weight in zip(optimization_results['assets'], optimization_results['weights']):
        weights_str.append(f"  - {asset}: {weight:.2%}")

    report = f"""
### Portfolio Optimization Report

Based on your investment goals and the market data analysis, here is the optimal portfolio allocation:

**Optimal Asset Allocation:**
{'\n'.join(weights_str)}

**Key Performance Indicators:**
  - **Expected Annual Return:** {optimization_results['return']:.2%}
  - **Expected Annual Volatility (Risk):** {optimization_results['volatility']:.2%}
  - **Sharpe Ratio:** {optimization_results['sharpe_ratio']:.2f} (Assuming a 1% risk-free rate)

**Analysis:**
This allocation aims to maximize the Sharpe Ratio, indicating the best risk-adjusted return, given the synthetic historical data. The expected return provides an estimate of your annual gains, while the volatility indicates the potential fluctuation in your portfolio's value.

**Disclaimer:**
This analysis is based on historical data and simulated market conditions. Past performance is not indicative of future results. Investment involves risks. Please consult with a financial advisor before making any investment decisions.
"""
    return report

# --- Secure Code Execution Environment ---

@contextlib.contextmanager
def capture_stdout():
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    try:
        yield new_stdout
    finally:
        sys.stdout = old_stdout

def execute_code_in_sandbox(code_string, initial_globals=None):
    """
    Executes Python code in a restricted sandbox environment and captures its stdout.
    WARNING: This is a simplified sandbox for demonstration. A production system
    would require a more robust and secure sandboxing mechanism (e.g., dedicated
    containers, RestrictedPython, or isolated processes with strict resource limits).
    """
    if initial_globals is None:
        # Limit available built-ins and prevent imports or file system access
        safe_builtins = {
            'print': print,
            'len': len,
            'sum': sum,
            'min': min,
            'max': max,
            'round': round,
            'abs': abs,
            'str': str,
            'int': int,
            'float': float,
            'dict': dict,
            'list': list,
            'tuple': tuple,
            'set': set,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
        }
        globals_dict = {
            "__builtins__": safe_builtins,
            "np": np, # Allow numpy for numerical operations
            "pd": pd, # Allow pandas for data structures
            "sco": sco, # Allow scipy.optimize for optimization
        }
    else:
        globals_dict = initial_globals.copy()
        globals_dict['__builtins__'] = initial_globals.get('__builtins__', {})

    local_dict = {}
    output_capture = ""
    error_capture = ""

    with capture_stdout() as stdout_buffer:
        try:
            exec(code_string, globals_dict, local_dict)
            output_capture = stdout_buffer.getvalue()
        except Exception as e:
            error_capture = f"Execution Error: {e}"

    return output_capture, error_capture

# --- Streamlit Frontend ---

st.set_page_config(layout="wide")
st.title("📈 Financial Portfolio Optimization and Risk Assessment System")
st.markdown("--- Developed using the **Code-Assisted Reasoning (CAR) Pattern** --- ")

st.sidebar.header("Input Investment Parameters")

# User Inputs
assets_input = st.sidebar.text_area(
    "Enter Asset Ticker Symbols (one per line):",
    value="AAPL\nMSFT\nGOOGL\nAMZN",
    height=150
)

# target_return = st.sidebar.slider(
#     "Target Annual Return (")%):",
#     min_value=0.0, max_value=50.0, value=15.0, step=1.0
# )
# 
# risk_tolerance = st.sidebar.slider(
#     "Risk Tolerance (0 = Low, 10 = High):",
#     min_value=0, max_value=10, value=5, step=1
# )

if st.sidebar.button("Analyze Portfolio"):
    if not assets_input:
        st.error("Please enter at least one asset ticker symbol.")
    else:
        assets_list = [asset.strip().upper() for asset in assets_input.split('\n') if asset.strip()]
        if len(assets_list) < 2:
            st.error("Please enter at least two asset ticker symbols for portfolio optimization.")
        else:
            with st.spinner("Generating and executing financial code..."):
                # Step 1: LLM generates code
                generated_code = simulate_llm_code_generation(assets_list)
                st.subheader("Generated Python Code (by LLM)")
                st.code(generated_code, language="python")

                # Step 2: Execute generated code in sandbox
                st.subheader("Code Execution Output")
                execution_output, execution_error = execute_code_in_sandbox(generated_code)

                if execution_error:
                    st.error(f"Code Execution Error: {execution_error}")
                    st.text_area("Raw Execution Output (Error Details):", value=execution_output, height=150)
                else:
                    st.text_area("Raw Execution Output (from print statements):", value=execution_output, height=200)

                    # Step 3: Parse results from execution output (looking for 'OPTIMIZATION_RESULTS:')
                    optimization_results = {}
                    if "OPTIMIZATION_RESULTS:" in execution_output:
                        try:
                            results_str = execution_output.split("OPTIMIZATION_RESULTS:", 1)[1].strip()
                            # Need to safely evaluate the string representation of the dictionary
                            optimization_results = eval(results_str)
                        except Exception as e:
                            st.error(f"Failed to parse optimization results from executed code: {e}")

                    if optimization_results:
                        # Step 4: LLM interprets results and generates report
                        final_report = simulate_llm_report_generation(optimization_results)
                        st.subheader("Financial Portfolio Report (Generated by LLM)")
                        st.markdown(final_report)
                    else:
                        st.warning("No valid optimization results were found in the code execution output.")

st.sidebar.markdown("""
--- 
**About CAR Pattern:**
This system demonstrates how an LLM can leverage external programming environments for precise calculations, enhancing its reasoning capabilities.
""")



# To run this application:
# 1. Save the code as `main.py`
# 2. Open your terminal or command prompt.
# 3. Navigate to the directory where you saved `main.py`.
# 4. Run the command: `streamlit run main.py`
#    (Make sure you have `streamlit`, `numpy`, `pandas`, `scipy` installed: `pip install streamlit numpy pandas scipy`)
