import os
import subprocess
import re
import json
from datetime import datetime, timedelta

# --- financial_utils.py content ---
FINANCIAL_UTILS_CODE = """
import yfinance as yf
import pandas as pd
import numpy as np

def get_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def calculate_returns(data):
    returns = data["Adj Close"].pct_change().dropna()
    return returns

def calculate_volatility(returns):
    volatility = returns.std() * np.sqrt(252)
    return volatility

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    avg_return = returns.mean() * 252
    volatility = calculate_volatility(returns)
    sharpe_ratio = (avg_return - risk_free_rate) / volatility
    return sharpe_ratio
"""

# --- main.py content ---
def write_financial_utils_file(filename="financial_utils.py"):
    with open(filename, "w") as f:
        f.write(FINANCIAL_UTILS_CODE)

def extract_code(llm_response_content):
    code_match = re.search(r"```python\n(.*?)```", llm_response_content, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return None

def execute_code(code_to_execute):
    temp_script_name = "temp_script.py"
    try:
        with open(temp_script_name, "w") as f:
            f.write(code_to_execute)
        process = subprocess.run(
            ["python", temp_script_name],
            capture_output=True,
            text=True,
            check=True
        )
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error executing code: {e.stderr}"
    finally:
        if os.path.exists(temp_script_name):
            os.remove(temp_script_name)

def simulate_llm_response(user_query, financial_result=None):
    if financial_result is None:
        if "sharpe ratio for AAPL" in user_query.lower() or "apple stock performance" in user_query.lower():
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")
            code = f"""
import financial_utils

ticker = "AAPL"
start_date = "{start_date}"
end_date = "{end_date}"

data = financial_utils.get_historical_data(ticker, start_date, end_date)
returns = financial_utils.calculate_returns(data)
sharpe_ratio = financial_utils.calculate_sharpe_ratio(returns)
print(f"AAPL Sharpe Ratio: {{sharpe_ratio:.4f}}")
"""
            return {"type": "code", "content": code}
        elif "volatility for MSFT" in user_query.lower() or "microsoft stock risk" in user_query.lower():
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")
            code = f"""
import financial_utils

ticker = "MSFT"
start_date = "{start_date}"
end_date = "{end_date}"

data = financial_utils.get_historical_data(ticker, start_date, end_date)
returns = financial_utils.calculate_returns(data)
volatility = financial_utils.calculate_volatility(returns)
print(f"MSFT Annualized Volatility: {{volatility:.4f}}")
"""
            return {"type": "code", "content": code}
        else:
            return {"type": "text", "content": "I can help with investment analysis like calculating Sharpe ratio or volatility for a given stock. Please specify a ticker and what you'd like to analyze."}
    else:
        if "sharpe ratio" in user_query.lower():
            sharpe_value_match = re.search(r"Sharpe Ratio: ([\d.-]+)", financial_result)
            if sharpe_value_match:
                sharpe_value = sharpe_value_match.group(1)
                return {"type": "text", "content": f"Based on my analysis, the AAPL Sharpe Ratio is {sharpe_value}. This indicates the risk-adjusted return of the stock over the past year. A higher Sharpe ratio generally suggests a better return for the amount of risk taken."}
            else:
                return {"type": "text", "content": f"Here is the result of the calculation: {financial_result}. This metric helps you understand the stock's performance relative to its risk."}
        elif "volatility" in user_query.lower():
            volatility_value_match = re.search(r"Volatility: ([\d.-]+)", financial_result)
            if volatility_value_match:
                volatility_value = volatility_value_match.group(1)
                return {"type": "text", "content": f"Based on my analysis, the MSFT Annualized Volatility is {volatility_value}. This represents the degree of variation of a trading price series over time. Higher volatility means higher risk and potentially higher reward."}
            else:
                return {"type": "text", "content": f"Here is the result of the calculation: {financial_result}. This metric helps you understand the risk associated with the stock."}
        else:
            return {"type": "text", "content": f"Here are the financial insights you requested: {financial_result}"}

def main():
    write_financial_utils_file() # Ensure financial_utils.py exists

    print("Welcome to the Smart Financial Assistant!")
    print("I can help you analyze stock performance and risk.")

    while True:
        user_query = input("\nHow can I help you today? (e.g., \"sharpe ratio for AAPL\", \"volatility for MSFT\", or \"exit\"): ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        llm_initial_response = simulate_llm_response(user_query)

        if llm_initial_response["type"] == "code":
            print("Thinking... performing calculations...")
            # The simulated LLM's content is already the code, no need for extra ```python block for extraction
            generated_code = llm_initial_response["content"]
            
            if generated_code:
                calculation_result = execute_code(generated_code)
                
                final_llm_response = simulate_llm_response(user_query, financial_result=calculation_result)
                print(f"Assistant: {final_llm_response['content']}")
            else:
                print("Assistant: I encountered an issue generating executable code for your request.")
        else:
            print(f"Assistant: {llm_initial_response['content']}")

    if os.path.exists("financial_utils.py"):
        os.remove("financial_utils.py") # Clean up

if __name__ == "__main__":
    main()
