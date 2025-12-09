import streamlit as st
import pandas as pd
import yfinance as yf
import io
import sys
import os

# Placeholder for OpenAI API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- LLM Simulation Functions (Replace with actual LLM API calls) ---
def simulate_llm_code_generation(query: str) -> str:
    if "average closing price" in query.lower() and "aapl" in query.lower():
        return """import yfinance as yf
ticker = yf.Ticker("AAPL")
hist = ticker.history(period="5y")
if not hist.empty:
    average_price = hist["Close"].mean()
    print(f"The 5-year average closing price of AAPL is {average_price:.2f}")
else:
    print("Could not retrieve AAPL data for the last 5 years.")
"""
    elif "current price of" in query.lower():
        symbol = query.lower().split("current price of ")[-1].strip().upper()
        if symbol:
            return f"""import yfinance as yf
ticker = yf.Ticker("{symbol}")
try:
    current_price = ticker.history(period="1d")["Close"].iloc[-1]
    print(f"The current price of {symbol} is {current_price:.2f}")
except IndexError:
    print(f"Could not retrieve current price for {symbol}. Invalid symbol or no data.")
"""
    else:
        return "print(\"Sorry, I can only calculate average closing price for AAPL over 5 years or current price for a given stock symbol.\")"

def simulate_llm_nl_response(query: str, code_output: str) -> str:
    if "Sorry, I can only" in code_output:
        return code_output
    return f"Based on your query: \"{query}\", and the executed calculations, here is the result:\n\n{{code_output}}\n\nThis information was precisely calculated using live market data."

# --- Code Execution Sandbox ---
def execute_code_in_sandbox(code: str) -> str:
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    exec_globals = {
        "pd": pd,
        "yf": yf,
        "__builtins__": {},
    }

    try:
        exec(code, exec_globals)
        output = redirected_output.getvalue()
    except Exception as e:
        output = f"Error during code execution: {e}"
    finally:
        sys.stdout = old_stdout
    return output

# --- Streamlit Application ---
st.set_page_config(layout="wide")
st.title("AI-Powered Financial Analysis Assistant (PAL Prompting Demo)")

st.markdown(
    "This assistant uses Program-Aided Language Models (PAL) to perform precise financial calculations."
    " It generates and executes Python code based on your query, then provides a natural language summary of the results."
)

user_query = st.text_area("Enter your financial query:",
                           "What is the 5-year average closing price of AAPL?", height=100)

if st.button("Get Financial Analysis"):
    if user_query:
        st.subheader("Thinking Process (Simulated LLM Interaction)")
        st.write("1. LLM generates Python code based on your query...")

        generated_code = simulate_llm_code_generation(user_query)
        st.code(generated_code, language="python")

        st.write("2. Executing the generated code...")
        code_execution_output = execute_code_in_sandbox(generated_code)
        st.text_area("Code Execution Output:", code_execution_output, height=150)

        st.write("3. LLM synthesizes natural language response from code output...")
        final_response = simulate_llm_nl_response(user_query, code_execution_output)

        st.subheader("Financial Analysis Result")
        st.markdown(final_response)
    else:
        st.warning("Please enter a financial query.")
