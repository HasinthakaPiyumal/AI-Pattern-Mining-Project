import streamlit as st
import pandas as pd
import numpy as np
import io
import contextlib

# Mocking PyPortfolioOpt components since we can't install external libraries directly in this sandbox.
# In a real application, you would import them normally:
# from pypfopt import expected_returns, risk_models, efficient_frontier
# from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# --- Mock PyPortfolioOpt Classes/Functions ---
class MockExpectedReturns:
    @staticmethod
    def mean_historical_returns(prices):
        # Simulate calculation
        returns = prices.pct_change().dropna()
        return returns.mean() * 252 # Annualize

class MockRiskModels:
    @staticmethod
    def sample_cov(prices):
        # Simulate calculation
        returns = prices.pct_change().dropna()
        return returns.cov() * 252 # Annualize

class MockEfficientFrontier:
    def __init__(self, expected_returns, cov_matrix, prices=None):
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.prices = prices # Not fully used in mock, but for completeness
        self.weights = None

    def max_sharpe(self, risk_free_rate=0.02):
        # Simulate max Sharpe optimization
        # For demonstration, we'll just return some plausible weights
        num_assets = len(self.expected_returns)
        weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
        self.weights = pd.Series(weights, index=self.expected_returns.index)
        return self.weights

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):
        if self.weights is None:
            return 0, 0, 0 # Return zeros if not optimized
        
        # Simulate portfolio performance calculation
        mu = (self.weights * self.expected_returns).sum()
        sigma = np.sqrt(self.weights.T @ self.cov_matrix @ self.weights)
        sharpe = (mu - risk_free_rate) / sigma
        return mu, sigma, sharpe

# Assign mocks to variables that the generated code expects
# In a real setup, these would be the actual imported modules
definitely_not_pypfopt_expected_returns = MockExpectedReturns()
definitely_not_pypfopt_risk_models = MockRiskModels()
definitely_not_pypfopt_efficient_frontier = MockEfficientFrontier


# --- Code Execution Environment ---

def code_executor(code: str, provided_globals: dict) -> tuple[str, str]:
    """
    Safely executes generated Python code with a restricted environment.
    Captures stdout and stderr.
    """
    # Create a limited global dictionary. Only explicitly provided globals are accessible.
    # __builtins__ is set to an empty dict to prevent access to most built-in functions.
    globals_dict = {
        "__builtins__": {},
        "pd": pd, 
        "np": np,
        "expected_returns": definitely_not_pypfopt_expected_returns,
        "risk_models": definitely_not_pypfopt_risk_models,
        "EfficientFrontier": definitely_not_pypfopt_efficient_frontier,
        **provided_globals # Add any specific dataframes or variables needed by the code
    }
    
    locals_dict = {}
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_buffer):
            with contextlib.redirect_stderr(error_buffer):
                exec(code, globals_dict, locals_dict)
    except Exception as e:
        error_buffer.write(f"Execution Error: {e}")

    return output_buffer.getvalue(), error_buffer.getvalue()


# --- Simulate LLM Code Generation (In a real app, this would be an LLM call) ---

def simulate_llm_code_generation(user_goals: str, risk_tolerance: str, assets: list) -> str:
    """
    Simulates an LLM generating Python code for portfolio optimization.
    The generated code uses the provided 'prices' DataFrame.
    """
    asset_list_str = ", ".join([f'"{a}"' for a in assets])
    
    # This is an example of code the LLM might generate.
    # It assumes 'prices' dataframe is available in the execution environment.
    generated_code = f"""
import pandas as pd
import numpy as np
# Using mocked pypfopt components that are provided in the execution environment
# In a real scenario, you'd just use 'from pypfopt import ...'

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_returns(prices)
S = risk_models.sample_cov(prices)

# Initialize EfficientFrontier object
# Using the mocked EfficientFrontier class
ef = EfficientFrontier(mu, S)

# Optimize for maximum Sharpe ratio
print("Optimizing for max Sharpe ratio...")
weights = ef.max_sharpe()

# Get portfolio performance
ret, std, sharpe = ef.portfolio_performance(verbose=True)

print("\nOptimized Weights:")
for asset, weight in weights.items():
    print(f"  {{asset}}: {{weight:.2%}}")

print(f"\nExpected annual return: {{ret*100:.2f}}%")
print(f"Annual volatility: {{std*100:.2f}}%")
print(f"Sharpe Ratio: {{sharpe:.2f}}")
"""
    return generated_code


# --- Simulate LLM Recommendation Generation (In a real app, this would be an LLM call) ---

def simulate_llm_recommendation(execution_output: str, user_goals: str, risk_tolerance: str) -> str:
    """
    Simulates an LLM generating an investment recommendation based on code execution output.
    """
    if "Execution Error" in execution_output or not execution_output.strip():
        return (
            "I encountered an error during the portfolio optimization calculation or received no output.\n"
            "Please review your inputs or the system's ability to process the request."
        )

    recommendation = (
        f"Based on your investment goals ({user_goals}) and risk tolerance ({risk_tolerance}), "
        f"the optimized portfolio analysis yielded the following results:\n\n"
        f"```\n{execution_output}\n```\n"
        f"Considering these figures, I recommend allocating your assets as per the 'Optimized Weights' shown above "
        f"to achieve an expected annual return of approximately {float(execution_output.split('Expected annual return: ')[1].split('%')[0]):.2f}% "
        f"with an annual volatility of {float(execution_output.split('Annual volatility: ')[1].split('%')[0]):.2f}%. "
        f"The Sharpe Ratio of {float(execution_output.split('Sharpe Ratio: ')[1].split('\n')[0]):.2f} indicates a good risk-adjusted return."
    )
    return recommendation


# --- Streamlit Application ---

st.set_page_config(layout="wide")
st.title("🧠 Code-Assisted Financial Portfolio Optimizer 📈")
st.markdown("This tool demonstrates the Code-Assisted Reasoning (CAR) pattern by generating and executing Python code for portfolio optimization based on your inputs.")

# --- Sidebar for Inputs ---
st.sidebar.header("Your Investment Profile")
user_goals = st.sidebar.text_area(
    "Investment Goals (e.g., 'long-term growth', 'income generation', 'capital preservation')",
    "long-term growth with moderate risk"
)
risk_tolerance = st.sidebar.selectbox(
    "Risk Tolerance",
    ("Low", "Moderate", "High"),
    index=1 # Default to Moderate
)

default_assets = "AAPL, MSFT, GOOGL, AMZN, TSLA"
asset_input = st.sidebar.text_input(
    "Comma-separated stock tickers (e.g., AAPL, MSFT)",
    default_assets
)
assets = [a.strip().upper() for a in asset_input.split(",") if a.strip()]

# --- Synthetic Data Generation (for demonstration) ---

def get_synthetic_stock_data(tickers, num_days=252*3, start_price=100):
    """
    Generates synthetic historical stock price data for given tickers.
    """
    dates = pd.date_range(end=pd.Timestamp.today(), periods=num_days, freq='B')
    data = pd.DataFrame(index=dates)
    
    for ticker in tickers:
        # Simulate daily returns with some randomness
        daily_returns = np.random.normal(0.0005, 0.01, num_days) # mean 0.05%, std 1%
        price_series = start_price * (1 + daily_returns).cumprod()
        data[ticker] = price_series
    return data

prices_df = get_synthetic_stock_data(assets)


# --- Main Application Logic ---

st.header("1. Your Input")
st.write(f"**Goals:** {user_goals}")
st.write(f"**Risk Tolerance:** {risk_tolerance}")
st.write(f"**Selected Assets:** {', '.join(assets)}")
st.subheader("Synthetic Historical Prices (first 5 rows)")
st.dataframe(prices_df.head())


if st.sidebar.button("Optimize Portfolio"):
    st.session_state["analysis_triggered"] = True
    st.header("2. LLM Generated Code for Optimization")
    
    # Simulate LLM generating code
    generated_code = simulate_llm_code_generation(user_goals, risk_tolerance, assets)
    st.code(generated_code, language="python")
    
    st.header("3. Code Execution Output")
    with st.spinner("Executing generated code..."):
        # Provide the 'prices' dataframe to the code execution environment
        execution_output, execution_error = code_executor(generated_code, {"prices": prices_df})
        
        if execution_error:
            st.error(f"Error during code execution:\n{execution_error}")
        else:
            st.text(execution_output)
            st.session_state["execution_output"] = execution_output


if st.session_state.get("analysis_triggered") and "execution_output" in st.session_state:
    st.header("4. LLM Investment Recommendation")
    with st.spinner("Generating recommendations..."):
        recommendation = simulate_llm_recommendation(st.session_state["execution_output"], user_goals, risk_tolerance)
        st.write(recommendation)
else:
    st.info("Enter your investment details in the sidebar and click 'Optimize Portfolio' to begin.")

