import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

class LLM_Simulator:
    def __init__(self):
        pass

    def generate_code_for_portfolio_optimization(self, query):
        tickers_str = ""
        # Simple keyword extraction for demonstration purposes
        if "Apple" in query: tickers_str += "'AAPL',"
        if "Google" in query or "Alphabet" in query: tickers_str += "'GOOG',"
        if "Microsoft" in query: tickers_str += "'MSFT',"
        if "Amazon" in query: tickers_str += "'AMZN',"
        if "Tesla" in query: tickers_str += "'TSLA',"
        if "Nvidia" in query: tickers_str += "'NVDA',"
        
        if not tickers_str: # Default if no specific stocks mentioned
            tickers_str = "'AAPL','GOOG','MSFT'" # Fallback example

        tickers_list = f"[{tickers_str.strip(',')}]"

        code = f"""
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

def get_data(tickers, start_date="2018-01-01", end_date=None):
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    return data

def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return portfolio_return, portfolio_std_dev, sharpe_ratio

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    return -calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def optimize_portfolio(mean_returns, cov_matrix, num_assets):
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    optimal_weights_sharpe = minimize(negative_sharpe_ratio, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimal_weights_sharpe.x

tickers = {tickers_list}
data = get_data(tickers)
log_returns = np.log(data / data.shift(1)).dropna()
mean_returns = log_returns.mean()
cov_matrix = log_returns.cov()

num_assets = len(tickers)
optimal_weights = optimize_portfolio(mean_returns, cov_matrix, num_assets)

optimal_return, optimal_std_dev, optimal_sharpe_ratio = calculate_portfolio_metrics(optimal_weights, mean_returns, cov_matrix)

output_data = {{
    "optimal_weights": {{t: w for t, w in zip(tickers, optimal_weights.round(4))}},
    "expected_annual_return": optimal_return.round(4),
    "annual_volatility": optimal_std_dev.round(4),
    "sharpe_ratio": optimal_sharpe_ratio.round(4)
}}

print(output_data)
"""
        return code

    def generate_explanation(self, user_query, code_output):
        weights = code_output.get("optimal_weights", {})
        expected_return = code_output.get("expected_annual_return", 0)
        volatility = code_output.get("annual_volatility", 0)
        sharpe = code_output.get("sharpe_ratio", 0)

        explanation = f"Based on your query: \"{user_query}\", and our financial analysis, here is a recommended portfolio optimization:\n\n"
        explanation += "### Optimal Asset Allocation:\n"
        if weights:
            for ticker, weight in weights.items():
                explanation += f"- **{ticker}**: {weight*100:.2f}%\n"
        else:
            explanation += "No specific asset allocation could be determined.\n"

        explanation += f"\n### Key Performance Metrics:\n"
        explanation += f"- **Expected Annual Return**: {expected_return*100:.2f}%\n"
        explanation += f"- **Annual Volatility (Risk)**: {volatility*100:.2f}%\n"
        explanation += f"- **Sharpe Ratio**: {sharpe:.2f} (Higher is better, indicating better risk-adjusted return)\n\n"
        explanation += "This portfolio aims to maximize your risk-adjusted returns given the historical performance of the selected assets. Please remember that past performance is not indicative of future results, and this is for informational purposes only. Consult with a financial advisor for personalized advice."

        return explanation

def execute_generated_code(code_string):
    output_capture = {}
    try:
        import io
        import sys
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        exec_globals = {'__builtins__': None, 'pd': pd, 'np': np, 'yf': yf, 'minimize': minimize}
        exec_locals = {}
        exec(code_string, exec_globals, exec_locals)
        
        sys.stdout = old_stdout
        captured_output = redirected_output.getvalue()

        if captured_output:
            try:
                output_capture = eval(captured_output)
            except Exception as e:
                st.error(f"Error parsing code output: {e}")
                output_capture = {"error": f"Failed to parse output: {e}"}

    except Exception as e:
        output_capture = {"error": f"Error during code execution: {e}"}
    return output_capture

st.set_page_config(layout="wide")
st.title("💰 AI-powered Financial Portfolio Optimizer")
st.markdown("""
This assistant helps retail investors optimize their investment portfolios by leveraging a simulated Program-Aided Language Model (PAL) approach. 
You provide your investment goals and current holdings in natural language, and the system generates and executes Python code to perform complex financial calculations, then provides a human-readable explanation.
""")

user_query = st.text_area(
    "Tell me about your investment goals, risk tolerance, and current holdings (e.g., 'I want to invest in Apple, Google, and Microsoft with a moderate risk tolerance for maximum growth.')",
    height=150,
    value="I want to invest $10,000 in Apple, Google, and Microsoft. My risk tolerance is moderate, and I aim for maximum returns over the next year."
)

if st.button("Analyze Portfolio"):
    if not user_query:
        st.warning("Please enter your investment query.")
    else:
        st.info("Analyzing your request...")
        llm = LLM_Simulator()

        st.subheader("1. LLM Generates Code")
        generated_code = llm.generate_code_for_portfolio_optimization(user_query)
        st.code(generated_code, language="python")

        st.subheader("2. Executing Generated Code")
        with st.spinner("Performing financial calculations..."):
            execution_results = execute_generated_code(generated_code)
        
        if "error" in execution_results:
            st.error(f"Error executing financial model: {execution_results['error']}")
        else:
            st.success("Financial calculations completed successfully!")
            st.json(execution_results)

            st.subheader("3. LLM Synthesizes Explanation")
            with st.spinner("Generating financial recommendations..."):
                final_explanation = llm.generate_explanation(user_query, execution_results)
            
            st.markdown(final_explanation)
