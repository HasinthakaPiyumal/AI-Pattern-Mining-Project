import streamlit as st
import io
import contextlib

def execute_python_code(code_string: str) -> str:
    """Executes a Python code string and captures its output."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code_string, {'__builtins__': {}})
        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        if errors:
            return f"Execution Error:\n{errors}"
        return output
    except Exception as e:
        return f"Runtime Error: {e}\n{stderr_capture.getvalue()}"


def mock_llm_generate_code(user_prompt: str) -> str:
    """Simulates an LLM generating Python code based on the user's prompt.
    In a real application, this would be an API call to an LLM.
    """
    # Simple keyword-based code generation for demonstration
    if "compound interest" in user_prompt.lower() or "future value" in user_prompt.lower():
        return """
def calculate_compound_interest(principal, rate, time):
    amount = principal * (1 + rate / 100)**time
    return round(amount - principal, 2)

def calculate_future_value(principal, rate, time):
    amount = principal * (1 + rate / 100)**time
    return round(amount, 2)

# Example usage (replace with actual extracted values from prompt in a real LLM scenario)
# For demonstration, we'll assume the LLM extracted these values from the prompt:
# e.g., if prompt was 'Calculate the compound interest for $1000 at 5% for 10 years'
principal_val = 1000
rate_val = 5
time_val = 10

interest = calculate_compound_interest(principal_val, rate_val, time_val)
future_value = calculate_future_value(principal_val, rate_val, time_val)
print(f"Principal: ${principal_val}")
print(f"Annual Rate: {rate_val}%")
print(f"Years: {time_val}")
print(f"Compound Interest Earned: ${interest}")
print(f"Future Value: ${future_value}")
"""
    elif "net present value" in user_prompt.lower() or "npv" in user_prompt.lower():
        return """
import numpy as np

def calculate_npv(rate, cash_flows):
    return np.npv(rate / 100, cash_flows)

# Example usage (replace with actual extracted values from prompt)
# e.g., if prompt was 'What is the Net Present Value of a project with initial cost of $1000 and cash flows of $300, $400, $500 with a discount rate of 10%?'
discount_rate = 10
cash_flows_list = [-1000, 300, 400, 500] # Initial investment as negative

npv = calculate_npv(discount_rate, cash_flows_list)
print(f"Discount Rate: {discount_rate}%")
print(f"Cash Flows: {cash_flows_list}")
print(f"Net Present Value (NPV): ${round(npv, 2)}")
"""
    elif "analyze stock performance" in user_prompt.lower() or "stock X" in user_prompt.lower():
        return """
import pandas as pd
# import yfinance as yf # Uncomment and install yfinance for real stock data

def get_mock_stock_data(symbol, start_date, end_date):
    # Simulate fetching stock data (replace with yfinance or other API in real app)
    dates = pd.date_range(start=start_date, end=end_date)
    prices = np.random.rand(len(dates)) * 100 + 50 # Random prices between 50 and 150
    df = pd.DataFrame({'Close': prices}, index=dates)
    return df

def analyze_stock_performance(symbol, start_date, end_date):
    # stock_data = yf.download(symbol, start=start_date, end=end_date) # Real data
    stock_data = get_mock_stock_data(symbol, start_date, end_date) # Mock data

    if stock_data.empty:
        return f"Could not retrieve data for {symbol}"

    initial_price = stock_data['Close'].iloc[0]
    final_price = stock_data['Close'].iloc[-1]
    price_change = final_price - initial_price
    percentage_change = (price_change / initial_price) * 100

    return f"Stock: {symbol}\n" \
           f"Period: {start_date} to {end_date}\n" \
           f"Initial Price: ${initial_price:.2f}\n" \
           f"Final Price: ${final_price:.2f}\n" \
           f"Price Change: ${price_change:.2f}\n" \
           f"Percentage Change: {percentage_change:.2f}%"

# Example usage (replace with actual extracted values from prompt)
stock_symbol = "AAPL" # Example stock
start = "2022-01-01"
end = "2022-12-31"

performance_report = analyze_stock_performance(stock_symbol, start, end)
print(performance_report)
"""
    else:
        return """
print("I'm sorry, I don't have a specific code function for that request yet. "
      "Please ask a financial question related to compound interest, net present value, or stock performance.")
"""

def mock_llm_formulate_answer(code_output: str, original_prompt: str) -> str:
    """Simulates an LLM formulating a natural language answer based on code output.
    In a real application, this would be another API call to an LLM.
    """
    if "Execution Error" in code_output or "Runtime Error" in code_output:
        return f"I encountered an error while processing your request.\nError Details:\n{code_output}\nPlease try rephrasing your question or check the input values."

    # Basic post-processing for demonstration
    if "Compound Interest Earned" in code_output:
        interest = code_output.split("Compound Interest Earned: ")[1].split("\n")[0]
        future_value = code_output.split("Future Value: ")[1].split("\n")[0]
        return f"Based on your request, I calculated the following:\n- The compound interest earned is {interest}.\n- The future value of your investment will be {future_value}."
    elif "Net Present Value (NPV)" in code_output:
        npv_value = code_output.split("Net Present Value (NPV): ")[1].split("\n")[0]
        return f"The Net Present Value (NPV) for the given cash flows and discount rate is {npv_value}."
    elif "Stock: " in code_output and "Performance Report" in original_prompt:
        lines = code_output.split("\n")
        report_summary = "Here is the stock performance summary:\n"
        for line in lines:
            report_summary += f"- {line}\n"
        return report_summary
    elif "I'm sorry, I don't have a specific code function" in code_output:
        return "I apologize, but I couldn't generate specific code for that financial request. Please try asking about compound interest, Net Present Value, or stock performance for a more detailed analysis."
    else:
        return f"Here is the result of my calculation:\n\n{code_output}\n\nIs there anything else I can help you with रिगarding your request?"


st.set_page_config(layout="wide", page_title="AI Financial Analyst Assistant")
st.title("📈 AI Financial Analyst Assistant (PAL Prompting Demo)")
st.markdown("This assistant uses **Program-Aided Language Models (PAL) Prompting** "
            "to perform precise financial calculations. It generates Python code, "
            "executes it, and then uses the results to formulate a natural language answer.")

user_question = st.text_area(
    "Ask a financial question (e.g., 'Calculate the compound interest for $1000 at 5% for 10 years', "
    "'What is the Net Present Value of a project with initial cost of $1000 and cash flows of $300, $400, $500 with a discount rate of 10%?', "
    "'Analyze the historical performance of stock AAPL from 2022-01-01 to 2022-12-31'):", 
    height=150
)

if st.button("Get Financial Analysis"): 
    if user_question:
        st.subheader("1. LLM Generates Code")
        generated_code = mock_llm_generate_code(user_question)
        st.code(generated_code, language="python")

        st.subheader("2. Executing Generated Code")
        code_output = execute_python_code(generated_code)
        st.text(code_output)

        st.subheader("3. LLM Formulates Answer")
        final_answer = mock_llm_formulate_answer(code_output, user_question)
        st.info(final_answer)
    else:
        st.warning("Please enter a financial question to get started!")

st.markdown("""
--- 
**How it works (Simplified):**
1.  You ask a financial question in natural language.
2.  A (simulated) Language Model generates Python code tailored to your request.
3.  The generated Python code is executed in a safe environment.
4.  The output from the code execution is fed back to the (simulated) Language Model.
5.  The Language Model uses the precise computational result to formulate a clear, natural language answer.

**Note:** For this demonstration, the LLM functions are mocked. A real application would integrate with actual LLM APIs (e.g., OpenAI, Gemini) for code generation and response formulation, and a secure sandboxed environment for code execution.
""")
