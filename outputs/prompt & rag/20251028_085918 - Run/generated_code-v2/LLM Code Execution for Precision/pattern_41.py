import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import io
import contextlib

# --- Financial Computation & Optimization Layer Functions ---

def get_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data.dropna()
    except Exception as e:
        st.error(f"Error fetching data for {tickers}: {e}")
        return pd.DataFrame()

def calculate_returns_covariance(df):
    returns = df.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights) * 252  # Annualized
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) # Annualized
    return portfolio_return, portfolio_std_dev

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_return, p_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std_dev

def optimize_portfolio_weights(mean_returns, cov_matrix, num_assets, risk_free_rate, risk_tolerance):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]

    if risk_tolerance == "conservative":
        # Minimize volatility
        objective_func = lambda weights: portfolio_performance(weights, mean_returns, cov_matrix)[1]
    elif risk_tolerance == "moderate":
        # Maximize Sharpe Ratio
        objective_func = lambda weights: neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)
    elif risk_tolerance == "aggressive":
        # Maximize Return for a given (higher) risk level (simplified to higher target return within Sharpe optimization framework)
        # For aggressive, we still use Sharpe but perhaps implicitly allow higher risk by not explicitly bounding std dev low.
        # Or, we could directly try to maximize return subject to a max std dev, but Sharpe is a good proxy.
        objective_func = lambda weights: neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)
    else:
        objective_func = lambda weights: neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)

    optimal_results = minimize(objective_func, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimal_results.x if optimal_results.success else None

# --- LLM Orchestration Layer (Simulated) ---

def generate_optimization_code(user_params):
    # This function simulates an LLM generating code based on user parameters.
    # In a real PAL implementation, the LLM would dynamically write this string.
    # For demonstration, we'll construct a fixed template for the core optimization.
    
    tickers_str = str(user_params['tickers'])
    start_date_str = user_params['start_date'].strftime('%Y-%m-%d')
    end_date_str = user_params['end_date'].strftime('%Y-%m-%d')
    risk_tolerance_str = user_params['risk_tolerance']
    risk_free_rate_str = str(user_params.get('risk_free_rate', 0.01))

    code = f"""
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna()

def calculate_returns_covariance(df):
    returns = df.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return portfolio_return, portfolio_std_dev

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_return, p_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std_dev

def optimize_portfolio_weights_internal(mean_returns, cov_matrix, num_assets, risk_free_rate, risk_tolerance):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]

    if risk_tolerance == \"conservative\":
        objective_func = lambda weights: portfolio_performance(weights, mean_returns, cov_matrix)[1]
    elif risk_tolerance == \"moderate\":
        objective_func = lambda weights: neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)
    elif risk_tolerance == \"aggressive\":
        objective_func = lambda weights: neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)
    else:
        objective_func = lambda weights: neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)

    optimal_results = minimize(objective_func, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimal_results.x if optimal_results.success else None

tickers = {tickers_str}
start_date = '{start_date_str}'
end_date = '{end_date_str}'
risk_tolerance = '{risk_tolerance_str}'
risk_free_rate = {risk_free_rate_str}

stock_data = get_stock_data(tickers, start_date, end_date)

if not stock_data.empty and len(stock_data.columns) > 1:
    mean_returns, cov_matrix = calculate_returns_covariance(stock_data)
    num_assets = len(stock_data.columns)
    optimal_weights = optimize_portfolio_weights_internal(mean_returns, cov_matrix, num_assets, risk_free_rate, risk_tolerance)

    if optimal_weights is not None:
        # Ensure weights sum to 1 due to potential floating point inaccuracies
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        optimal_portfolio_return, optimal_portfolio_std_dev = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
        
        print("---OPTIMIZATION_RESULTS_START---")
        print(f"Optimal Weights: {list(zip(stock_data.columns.tolist(), [f'{w:.4f}' for w in optimal_weights]))}")
        print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
        print(f"Annual Volatility: {optimal_portfolio_std_dev:.4f}")
        if optimal_portfolio_std_dev > 0:
            sharpe_ratio = (optimal_portfolio_return - risk_free_rate) / optimal_portfolio_std_dev
            print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print("---OPTIMIZATION_RESULTS_END---")
    else:
        print("---OPTIMIZATION_RESULTS_START---")
        print("Optimization failed or no valid weights found.")
        print("---OPTIMIZATION_RESULTS_END---")
elif not stock_data.empty and len(stock_data.columns) == 1:
    print("---OPTIMIZATION_RESULTS_START---")
    print(f"Only one asset ({stock_data.columns[0]}) selected. No optimization performed.")
    print(f"Expected Annual Return: {stock_data.pct_change().dropna().mean().iloc[0] * 252:.4f}")
    print(f"Annual Volatility: {stock_data.pct_change().dropna().std().iloc[0] * np.sqrt(252):.4f}")
    print("---OPTIMIZATION_RESULTS_END---")
else:
    print("---OPTIMIZATION_RESULTS_START---")
    print("Could not retrieve sufficient stock data for optimization.")
    print("---OPTIMIZATION_RESULTS_END---")
"""
    return code

def execute_and_interpret_code(generated_code, user_params, investment_amount):
    # Execute the generated code in a safe context (simulated here with a captured stdout)
    # WARNING: Using exec() with untrusted input is a security risk. 
    # A real application should use a secure sandbox (e.g., separate process, container).
    
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        try:
            exec(generated_code, {'__builtins__': {}})
        except Exception as e:
            return f"Error during code execution: {e}"
    
    raw_output = output_buffer.getvalue()
    
    # Interpret the raw output from the executed code
    if "---OPTIMIZATION_RESULTS_START---" in raw_output and "---OPTIMIZATION_RESULTS_END---" in raw_output:
        start_index = raw_output.find("---OPTIMIZATION_RESULTS_START---") + len("---OPTIMIZATION_RESULTS_START---")
        end_index = raw_output.find("---OPTIMIZATION_RESULTS_END---")
        result_lines = raw_output[start_index:end_index].strip().split('\n')
        
        parsed_results = {}
        for line in result_lines:
            if ":" in line:
                key, value = line.split(':', 1)
                parsed_results[key.strip()] = value.strip()
        
        if "Optimal Weights" in parsed_results:
            weights_str = parsed_results["Optimal Weights"].strip('[]')
            asset_weights = []
            for item in weights_str.split('), '):
                item = item.strip('(').strip(')').strip("'")
                if "," in item:
                    ticker, weight = item.split("'")
                    ticker = ticker.strip(', ').strip("('")
                    weight = float(weight.strip(' ,'))
                    asset_weights.append((ticker, weight))
            
            recommendation = f"Based on your preferences ({user_params['risk_tolerance']} risk tolerance and target return of {user_params['desired_return'] * 100:.1f}%):\n\n"
            recommendation += "Here is your optimized portfolio allocation:\n"
            total_allocated_amount = 0
            for ticker, weight in asset_weights:
                amount = investment_amount * weight
                recommendation += f"- **{ticker}**: {weight*100:.2f}% (approx. ${amount:,.2f})\n"
                total_allocated_amount += amount
            
            recommendation += f"\nTotal allocated amount: ${total_allocated_amount:,.2f}\n"
            recommendation += f"Expected Annual Return: **{float(parsed_results.get('Expected Annual Return', 0)) * 100:.2f}%**\n"
            recommendation += f"Annual Volatility (Risk): **{float(parsed_results.get('Annual Volatility', 0)) * 100:.2f}%**\n"
            if 'Sharpe Ratio' in parsed_results:
                 recommendation += f"Sharpe Ratio: **{float(parsed_results.get('Sharpe Ratio', 0)):.2f}**\n"
            recommendation += "\nThis portfolio aims to provide a diversified approach while aligning with your risk profile. Remember that past performance is not indicative of future results, and market conditions can change.\n"
            return recommendation
        elif "Optimization failed" in raw_output or "no valid weights found" in raw_output:
            return "The optimization process could not find a suitable portfolio based on the provided criteria. This might happen with very restrictive constraints or insufficient data. Please try adjusting your inputs."
        elif "Only one asset" in raw_output:
            return f"You selected only one asset. No diversification optimization performed.\n{parsed_results.get('Optimal Weights', '')}\nExpected Annual Return: {float(parsed_results.get('Expected Annual Return', 0)) * 100:.2f}%\nAnnual Volatility: {float(parsed_results.get('Annual Volatility', 0)) * 100:.2f}%"
        elif "Could not retrieve sufficient stock data" in raw_output:
             return "We couldn't retrieve sufficient historical data for the selected assets. Please check the ticker symbols or try different ones."
    return "Could not parse optimization results. Raw output: " + raw_output

# --- Streamlit UI Layer ---

st.set_page_config(layout="wide", page_title="PAL Investment Optimizer")

st.title("📈 Personalized Investment Portfolio Optimizer (PAL Powered)")
st.markdown("This assistant uses **Program-Aided Language Models (PAL) prompting** to optimize your investment portfolio. It generates and executes Python code to perform complex financial calculations and then interprets the results to provide natural language recommendations.")

st.header("Your Investment Preferences")

col1, col2 = st.columns(2)

with col1:
    investment_amount = st.number_input("Total Investment Amount ($)", min_value=1000, value=50000, step=1000)
    desired_return = st.slider("Target Annual Return (%)", min_value=1.0, max_value=20.0, value=7.0, step=0.5) / 100.0
    
with col2:
    risk_tolerance = st.selectbox(
        "Your Risk Tolerance",
        ("conservative", "moderate", "aggressive"),
        index=1 # Moderate by default
    )
    st.markdown("*(Conservative: Focus on minimizing risk; Moderate: Balance risk and return (Maximize Sharpe); Aggressive: Focus on higher returns, accepting more risk)*")
    

st.subheader("Select Assets for Your Portfolio")
# Example common ETFs/stocks. In a real app, this would be dynamic or user-searchable.
default_tickers = ["SPY", "QQQ", "BND", "GLD", "VNQ", "MSFT", "AAPL"]
selected_tickers = st.multiselect(
    "Choose stocks/ETFs (e.g., SPY, QQQ, BND, AAPL)",
    options=default_tickers + ["GOOGL", "AMZN", "TSLA", "JPM", "VTI", "VXUS"],
    default=default_tickers
)

# Date range for historical data
current_date = pd.to_datetime('today')
default_start_date = current_date - pd.DateOffset(years=5)
start_date = st.date_input("Historical Data Start Date", value=default_start_date)
end_date = st.date_input("Historical Data End Date", value=current_date)

risk_free_rate = st.sidebar.number_input("Risk-Free Rate (e.g., T-Bill yield)", min_value=0.0, max_value=0.1, value=0.01, step=0.001)

if st.button("Optimize Portfolio"):
    if not selected_tickers:
        st.warning("Please select at least two assets for optimization.")
    elif start_date >= end_date:
        st.warning("Start date must be before end date.")
    else:
        st.info("Generating optimization code and executing...")

        user_params = {
            "investment_amount": investment_amount,
            "desired_return": desired_return,
            "risk_tolerance": risk_tolerance,
            "tickers": selected_tickers,
            "start_date": pd.to_datetime(start_date),
            "end_date": pd.to_datetime(end_date),
            "risk_free_rate": risk_free_rate
        }

        # Step 1: LLM (simulated) generates code
        generated_code = generate_optimization_code(user_params)
        
        st.subheader("Generated Optimization Code (Simulated LLM Output)")
        st.code(generated_code, language='python')

        # Step 2 & 3: External Interpreter (simulated exec()) executes code and LLM interprets results
        st.subheader("Optimization Results & Interpretation (Simulated LLM)")
        with st.spinner("Running portfolio optimization and interpreting results..."):            
            final_recommendation = execute_and_interpret_code(generated_code, user_params, investment_amount)
            st.write(final_recommendation)
        
        st.success("Optimization complete!")

st.sidebar.markdown("### About this App")
st.sidebar.markdown("This app demonstrates the **Program-Aided Language Models (PAL) Prompting** pattern. An AI (simulated here) generates Python code for complex financial calculations, executes it, and then interprets the numerical results into a natural language recommendation.")
