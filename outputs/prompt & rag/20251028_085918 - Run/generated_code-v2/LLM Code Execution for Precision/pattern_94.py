import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def simulate_llm_code_generation(user_query: str) -> str:
    if "optimize my portfolio" in user_query.lower() and "tech stocks" in user_query.lower():
        return """
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def get_historical_data(tickers, start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    data = {}
    np.random.seed(42)
    for ticker in tickers:
        prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        data[ticker] = prices
    df = pd.DataFrame(data, index=dates)
    return df

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
start_date = '2020-01-01'
end_date = '2023-01-01'

df = get_historical_data(tickers, start_date, end_date)

returns = df.pct_change().dropna()

cov_matrix_annual = returns.cov() * 252

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def portfolio_return(weights, avg_returns):
    return np.sum(avg_returns * weights) * 252

avg_returns = returns.mean()

def minimize_volatility(weights, cov_matrix, avg_returns, target_return=None):
    if target_return is None:
        return portfolio_variance(weights, cov_matrix)
    else:
        return portfolio_variance(weights, cov_matrix)

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))

initial_weights = np.array(len(tickers) * [1. / len(tickers)])

optimized_results = minimize(minimize_volatility, initial_weights, args=(cov_matrix_annual, avg_returns),
                             method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = optimized_results.x
min_volatility = np.sqrt(portfolio_variance(optimal_weights, cov_matrix_annual))
expected_return = portfolio_return(optimal_weights, avg_returns)

output = {
    "optimal_weights": dict(zip(tickers, optimal_weights.round(4))),
    "min_volatility_annual": round(min_volatility, 4),
    "expected_annual_return": round(expected_return, 4)
}

print(output)
"""
    return "I'm sorry, I can only optimize tech stock portfolios at the moment."

def execute_generated_code(code: str) -> str:
    import io
    import sys
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    exec_globals = {'pd': pd, 'np': np, 'minimize': minimize}
    try:
        exec(code, exec_globals)
        captured_output = redirected_output.getvalue()
        return captured_output
    except Exception as e:
        return f"Error during code execution: {e}"
    finally:
        sys.stdout = old_stdout

def generate_final_answer(user_query: str, code_output: str) -> str:
    if "Error during code execution" in code_output:
        return f"I encountered an issue while performing the calculation: {code_output}. Please check the query or try again."
    
    if "I'm sorry" in code_output:
        return code_output

    try:
        result_dict = eval(code_output.strip())
        
        optimal_weights = result_dict.get("optimal_weights", {})
        min_volatility = result_dict.get("min_volatility_annual")
        expected_return = result_dict.get("expected_annual_return")

        response = f"Based on your request to optimize your portfolio, I performed the following analysis:\n\n"
        response += f"**Optimized Portfolio Allocation:**\n"
        for ticker, weight in optimal_weights.items():
            response += f"- {ticker}: {weight*100:.2f}%\n"
        response += f"\n**Expected Annual Return:** {expected_return*100:.2f}%\n"
        response += f"**Minimum Annual Volatility (Risk):** {min_volatility*100:.2f}%\n\n"
        response += "This allocation aims to achieve the minimum volatility for your selected tech stocks, given historical data trends."
        return response
    except Exception as e:
        return f"I processed the code, but there was an error interpreting its output: {e}\nRaw output: {code_output}"

st.title("AI-powered Financial Analyst Assistant 📈")
st.write("Ask me to optimize your portfolio of tech stocks and get detailed insights!")

user_input = st.text_area("Enter your financial query:", "Optimize my portfolio of tech stocks and calculate the expected return and risk.")

if st.button("Analyze Portfolio"):
    if user_input:
        st.info("Generating Python code for your request...")
        generated_code = simulate_llm_code_generation(user_input)
        
        if "I'm sorry" in generated_code:
            st.warning(generated_code)
        else:
            st.subheader("Generated Python Code:")
            st.code(generated_code, language="python")

            st.info("Executing the generated code...")
            execution_output = execute_generated_code(generated_code)
            
            st.subheader("Code Execution Output:")
            st.text(execution_output)

            st.info("Generating final financial analysis...")
            final_response = generate_final_answer(user_input, execution_output)
            
            st.subheader("Financial Analysis Report:")
            st.markdown(final_response)
    else:
        st.warning("Please enter a query to analyze.")