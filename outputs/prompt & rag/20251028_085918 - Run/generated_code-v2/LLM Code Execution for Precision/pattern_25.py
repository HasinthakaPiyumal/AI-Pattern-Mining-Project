import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os

# --- FastAPI Backend --- #

app = FastAPI()

class PortfolioHolding(BaseModel):
    symbol: str
    quantity: float

class PortfolioAnalysisRequest(BaseModel):
    portfolio: list[PortfolioHolding]
    query: str

class PortfolioAnalysisResponse(BaseModel):
    explanation: str
    raw_results: dict = None

# Mock LLM for code generation and explanation (replace with actual OpenAI/LLM integration)
def mock_llm_generate_code_and_explanation(portfolio_data: list[dict], user_query: str):
    # This is a highly simplified mock. In a real scenario, the LLM would dynamically
    # generate Python code based on the query and existing financial functions.
    
    symbols = [h['symbol'] for h in portfolio_data]
    quantities = [h['quantity'] for h in portfolio_data]

    generated_code = ""
    explanation_prefix = ""
    results_dict = {}

    if "risk" in user_query.lower() or "volatility" in user_query.lower():
        generated_code = f"""
import yfinance as yf
import pandas as pd
import numpy as np

symbols = {symbols}
start_date = '2020-01-01'
end_date = '2023-12-31'

df = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
df_returns = df.pct_change().dropna()

portfolio_weights = np.array({quantities}) / np.sum({quantities})
portfolio_return = np.sum(df_returns.mean() * portfolio_weights) * 252
portfolio_std_dev = np.sqrt(np.dot(portfolio_weights.T, np.dot(df_returns.cov() * 252, portfolio_weights)))

results['portfolio_std_dev'] = portfolio_std_dev
results['portfolio_annual_return'] = portfolio_return
"""
        explanation_prefix = "Here's an analysis of your portfolio's risk and return:\n"

    elif "sharpe ratio" in user_query.lower():
        generated_code = f"""
import yfinance as yf
import pandas as pd
import numpy as np

symbols = {symbols}
quantities = np.array({quantities})
start_date = '2020-01-01'
end_date = '2023-12-31'
risk_free_rate = 0.02 # Example risk-free rate

df = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
df_returns = df.pct_change().dropna()

portfolio_weights = quantities / np.sum(quantities)
portfolio_return = np.sum(df_returns.mean() * portfolio_weights) * 252
portfolio_std_dev = np.sqrt(np.dot(portfolio_weights.T, np.dot(df_returns.cov() * 252, portfolio_weights)))

sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

results['sharpe_ratio'] = sharpe_ratio
results['portfolio_annual_return'] = portfolio_return
results['portfolio_std_dev'] = portfolio_std_dev
"""
        explanation_prefix = "Here's the Sharpe Ratio for your portfolio:\n"

    elif "return" in user_query.lower():
        generated_code = f"""
import yfinance as yf
import pandas as pd
import numpy as np

symbols = {symbols}
quantities = np.array({quantities})
start_date = '2020-01-01'
end_date = '2023-12-31'

df = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
df_returns = df.pct_change().dropna()

portfolio_weights = quantities / np.sum(quantities)
portfolio_return = np.sum(df_returns.mean() * portfolio_weights) * 252

results['portfolio_annual_return'] = portfolio_return
"""
        explanation_prefix = "Here's your portfolio's estimated annual return:\n"

    else:
        return "I can generate code for risk, return, or Sharpe ratio analysis. Please specify your request.", {}, "" # No code generated for unknown queries
    
    return generated_code, results_dict, explanation_prefix

@app.post("/analyze_portfolio", response_model=PortfolioAnalysisResponse)
async def analyze_portfolio_endpoint(request: PortfolioAnalysisRequest):
    try:
        portfolio_data_dicts = [h.model_dump() for h in request.portfolio]
        
        # 1. LLM generates code based on query and portfolio
        generated_code, results_container, explanation_prefix = mock_llm_generate_code_and_explanation(portfolio_data_dicts, request.query)
        
        if not generated_code:
            return PortfolioAnalysisResponse(explanation=explanation_prefix, raw_results={})

        # 2. Execute the generated code in a safe environment
        local_vars = {"results": results_container}
        # WARNING: Using exec() is inherently dangerous. For production, use a secure sandbox environment.
        exec(generated_code, {"pd": pd, "np": np, "yf": yf, "__builtins__": {}}, local_vars)
        
        # 3. LLM synthesizes explanation from numerical results
        final_results = local_vars["results"]
        
        explanation = explanation_prefix
        if "portfolio_std_dev" in final_results:
            explanation += f"\nAnnualized Volatility (Standard Deviation): {final_results['portfolio_std_dev']:.2%}"
        if "portfolio_annual_return" in final_results:
            explanation += f"\nAnnualized Return: {final_results['portfolio_annual_return']:.2%}"
        if "sharpe_ratio" in final_results:
            explanation += f"\nSharpe Ratio (assuming 2% risk-free rate): {final_results['sharpe_ratio']:.2f}"
        
        if not final_results:
            explanation += "\nNo specific financial metrics were calculated for your query. Please try asking for 'risk', 'return', or 'Sharpe Ratio'."

        return PortfolioAnalysisResponse(explanation=explanation, raw_results=final_results)
    except Exception as e:
        return PortfolioAnalysisResponse(explanation=f"An error occurred: {str(e)}", raw_results={})

# --- Streamlit Frontend --- #

def streamlit_app():
    st.set_page_config(page_title="AI Financial Analysis Assistant")
    st.title("📈 AI Financial Analysis Assistant")
    st.write("Get insights into your investment portfolio using AI-powered financial calculations.")

    st.subheader("Your Portfolio Holdings")
    portfolio_input = st.text_area(
        "Enter your portfolio (e.g., AAPL 10, MSFT 5, GOOG 2)",
        "AAPL 10\nMSFT 5\nGOOG 2",
        height=150
    )

    st.subheader("Your Financial Query")
    user_query = st.text_input(
        "What would you like to know about your portfolio? (e.g., 'What is the risk and return?', 'Calculate the Sharpe Ratio.')",
        "What is the risk and return of my portfolio?"
    )

    if st.button("Analyze Portfolio"): 
        if not portfolio_input:
            st.error("Please enter your portfolio holdings.")
            return
        if not user_query:
            st.error("Please enter your financial query.")
            return

        portfolio_holdings = []
        for line in portfolio_input.strip().split('\n'):
            parts = line.split()
            if len(parts) == 2:
                try:
                    symbol = parts[0].strip().upper()
                    quantity = float(parts[1].strip())
                    portfolio_holdings.append({"symbol": symbol, "quantity": quantity})
                except ValueError:
                    st.warning(f"Skipping invalid line: {line}. Please use format 'SYMBOL QUANTITY'.")
            else:
                st.warning(f"Skipping invalid line: {line}. Please use format 'SYMBOL QUANTITY'.")
        
        if not portfolio_holdings:
            st.error("No valid portfolio holdings found. Please check your input format.")
            return

        st.info("Analyzing your portfolio... This may take a moment.")

        try:
            # Assuming FastAPI backend is running on http://127.0.0.1:8000
            # In a deployed scenario, replace with your actual backend URL
            response = requests.post(
                "http://127.0.0.1:8000/analyze_portfolio",
                json={
                    "portfolio": portfolio_holdings,
                    "query": user_query
                }
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            analysis_result = response.json()

            st.subheader("Analysis Results")
            st.write(analysis_result["explanation"])
            
            if analysis_result["raw_results"]:
                st.json(analysis_result["raw_results"])

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Please ensure it is running at http://127.0.0.1:8000.")
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e}. Details: {response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


# --- How to Run This Application --- #

if __name__ == "__main__":
    st.write("To run the FastAPI backend: `uvicorn main:app --reload`")
    st.write("To run the Streamlit frontend: `streamlit run main.py`")
    st.write("\nThis file contains both the FastAPI backend and Streamlit frontend code.\n")
    st.write("1. Save this code as `main.py`.")
    st.write("2. Install required libraries: `pip install fastapi uvicorn 'python-multipart' streamlit requests yfinance pandas numpy pydantic`")
    st.write("3. In one terminal, navigate to the directory and run: `uvicorn main:app --reload` (This starts the backend API).")
    st.write("4. In a separate terminal, navigate to the same directory and run: `streamlit run main.py` (This starts the frontend UI).")

    # This part will only run if you execute `python main.py` directly, 
    # but for Streamlit it's `streamlit run main.py` and for FastAPI it's `uvicorn main:app`.
    # We include it here for clarity on how to start each component.

    # If you want to make Streamlit a sub-process of a single `python main.py` command, 
    # you'd need to use `subprocess` and manage processes, which is more complex for a single file example.
    
    # For demonstration, we'll just call the streamlit_app function if this file is run directly 
    # but with checks to avoid conflicts with uvicorn. In practice, they are separate processes.
    
    # A simple check to avoid running Streamlit app directly if uvicorn tries to load 'main:app'
    # (though uvicorn typically doesn't run __main__ directly like this for the 'app' object)
    if "streamlit" in os.environ.get("STREAMLUIT_SERVER_URL", ""): # Check if running within Streamlit context
        streamlit_app()
    elif "uvicorn" not in os.environ.get("UVICORN_APP", ""): # Basic check if uvicorn is not trying to load it as an app
        st.write("No application started. Please follow the instructions above to run FastAPI and Streamlit.")
        # You could also call streamlit_app() here directly, but it's better to guide the user to `streamlit run main.py`

