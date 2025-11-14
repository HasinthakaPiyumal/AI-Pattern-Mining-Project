
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import subprocess
import sys
import io
import logging
import os

# --- Configuration and Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Retrieval Module ---
class DataRetriever:
    def get_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetches historical stock data using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty:
                logging.warning(f"No data found for {ticker} for period {period}")
                return pd.DataFrame()
            logging.info(f"Successfully retrieved data for {ticker}")
            return hist
        except Exception as e:
            logging.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

# --- Secure Code Executor ---
class CodeExecutor:
    def execute_code(self, code: str, timeout: int = 10) -> tuple[str, str]:
        """
        Executes Python code in a subprocess, capturing stdout and stderr.
        Includes a basic timeout mechanism.
        """
        try:
            # Use a temporary file to execute the code to better isolate it
            temp_script_path = "temp_exec_script.py"
            with open(temp_script_path, "w") as f:
                f.write(code)

            process = subprocess.run(
                [sys.executable, temp_script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False # Do not raise exception for non-zero exit codes immediately
            )

            os.remove(temp_script_path)

            if process.returncode != 0:
                error_msg = f"Code execution failed with exit code {process.returncode}:\n{process.stderr}"
                logging.error(error_msg)
                return "", error_msg

            logging.info("Code executed successfully.")
            return process.stdout, process.stderr

        except subprocess.TimeoutExpired:
            error_msg = f"Code execution timed out after {timeout} seconds."
            logging.error(error_msg)
            return "", error_msg
        except Exception as e:
            error_msg = f"Unexpected error during code execution: {e}"
            logging.error(error_msg)
            return "", error_msg

# --- Simulated LLM Components (for demonstration) ---
# In a real application, these would be calls to an actual LLM API

class LLMSimulator:
    def __init__(self):
        self.data_retriever = DataRetriever()

    def request_interpreter(self, query: str) -> dict:
        """
        Simulates LLM interpreting a natural language query.
        Extracts key financial metrics, ticker, and timeframe.
        """
        interpreted = {
            "original_query": query,
            "ticker": None,
            "metrics": [],
            "period": "1y" # Default period
        }

        query_lower = query.lower()

        # Extract ticker (simple heuristic: 2-5 letter uppercase word)
        import re
        ticker_match = re.search(r"\b[A-Z]{2,5}\b", query)
        if ticker_match:
            interpreted["ticker"] = ticker_match.group(0)

        if "cagr" in query_lower or "compound annual growth rate" in query_lower:
            interpreted["metrics"].append("CAGR")
        if "debt-to-equity" in query_lower or "d/e ratio" in query_lower:
            interpreted["metrics"].append("DE_RATIO")
        if "volatility" in query_lower or "standard deviation" in query_lower:
            interpreted["metrics"].append("VOLATILITY")
        if "average price" in query_lower or "mean price" in query_lower:
            interpreted["metrics"].append("AVG_PRICE")

        # Extract period
        if "5-year" in query_lower or "5 year" in query_lower:
            interpreted["period"] = "5y"
        elif "3-year" in query_lower or "3 year" in query_lower:
            interpreted["period"] = "3y"
        elif "1-year" in query_lower or "1 year" in query_lower:
            interpreted["period"] = "1y"
        elif "6-month" in query_lower or "6 month" in query_lower:
            interpreted["period"] = "6mo"
        elif "3-month" in query_lower or "3 month" in query_lower:
            interpreted["period"] = "3mo"

        logging.info(f"Interpreted request: {interpreted}")
        return interpreted

    def code_generator(self, interpreted_request: dict) -> str:
        """
        Simulates LLM generating Python code based on the interpreted request.
        This function directly fetches data to be available for the generated code.
        """
        ticker = interpreted_request.get("ticker")
        metrics = interpreted_request.get("metrics", [])
        period = interpreted_request.get("period")

        if not ticker:
            return "print(\"Error: Stock ticker not found in the query.\")"

        # Fetch data and make it available in the execution environment
        df = self.data_retriever.get_stock_data(ticker, period)
        if df.empty:
            return f"print(\"Error: Could not retrieve data for {ticker} or no data for the specified period.\")"

        # Serialize DataFrame to a string that can be re-read by the generated script
        df_csv = df.to_csv(index=True)

        code_snippets = []
        code_snippets.append('''
import pandas as pd
import numpy as np
import io

df_csv = """''' + df_csv + '''"""
df = pd.read_csv(io.StringIO(df_csv), index_col=0, parse_dates=True)

results = {}
''')

        if "CAGR" in metrics:
            code_snippets.append(f"""
# Calculate CAGR
start_price = df["Close"].iloc[0]
end_price = df["Close"].iloc[-1]
num_years = (df.index[-1] - df.index[0]).days / 365.25
if num_years > 0:
    cagr = ((end_price / start_price) ** (1 / num_years)) - 1
    results["CAGR"] = f"{{cagr:.2%}}"
else:
    results["CAGR"] = "N/A (insufficient data for CAGR)"
""")

        if "DE_RATIO" in metrics:
            # Placeholder: In a real app, this would come from fundamental data API
            # For demonstration, we'll just simulate a value or require it to be passed.
            code_snippets.append(f"""
# Simulate Debt-to-Equity Ratio (requires fundamental data, simplified here)
# In a real scenario, this would involve fetching from a financial API with fundamental data
# For this simulation, we'll use a dummy value or a heuristic.
# Let's assume a hypothetical DE ratio for demonstration purposes.
# A real implementation would fetch this from a robust financial data source.
# For example, if we had a function `get_fundamental_data(ticker)`:
# fundamental_data = get_fundamental_data(\"{ticker}\")
# de_ratio = fundamental_data.get(\"debtToEquityRatio\", "N/A")

# Dummy value for demonstration:
results["Debt-to-Equity Ratio"] = "0.85 (Simulated - requires fundamental data)"
""")

        if "VOLATILITY" in metrics:
            code_snippets.append(f"""
# Calculate Volatility (Annualized Standard Deviation of Daily Returns)
returns = df["Close"].pct_change().dropna()
annualized_volatility = returns.std() * np.sqrt(252)
results["Annualized Volatility"] = f"{{annualized_volatility:.2%}}"
""")

        if "AVG_PRICE" in metrics:
            code_snippets.append(f"""
# Calculate Average Closing Price
avg_price = df["Close"].mean()
results["Average Closing Price"] = f"{{avg_price:.2f}}"
""")

        code_snippets.append('''
import json
print(json.dumps(results))
''')

        generated_code = "\n".join(code_snippets)
        logging.info(f"Generated code:\n{generated_code}")
        return generated_code

    def output_integrator(self, execution_result: str) -> dict:
        """
        Simulates LLM integrating the structured output from code execution.
        Parses JSON output from the executed code.
        """
        try:
            import json
            integrated_output = json.loads(execution_result)
            logging.info(f"Integrated output: {integrated_output}")
            return integrated_output
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from code execution result: {e}")
            return {"error": f"Failed to parse results: {e}", "raw_output": execution_result}

    def report_generator(self, integrated_output: dict, original_query: str) -> str:
        """
        Simulates LLM generating a human-readable report based on the integrated output.
        """
        report = f"### Financial Analysis Report\n\n"
        report += f"**Original Request:** {original_query}\n\n"

        if "error" in integrated_output:
            report += f"**Analysis Error:** {integrated_output['error']}\n"
            if "raw_output" in integrated_output:
                report += f"Raw output: ```\n{integrated_output['raw_output']}\n```\n"
            return report

        report += "**Key Metrics:**\n"
        for metric, value in integrated_output.items():
            report += f"- **{metric}:** {value}\n"

        report += "\n**Insights:**\n"
        report += "*(This section would typically be generated by a more sophisticated LLM based on the calculated metrics and general financial knowledge.)*\n"
        if "CAGR" in integrated_output:
            report += f"- The Compound Annual Growth Rate (CAGR) of {integrated_output.get('CAGR', 'N/A')} indicates the average annual growth over the period.\n"
        if "Debt-to-Equity Ratio" in integrated_output:
            report += f"- The Debt-to-Equity Ratio of {integrated_output.get('Debt-to-Equity Ratio', 'N/A')} provides insight into the company's financial leverage. A lower ratio typically indicates lower risk.\n"
        if "Annualized Volatility" in integrated_output:
            report += f"- The Annualized Volatility of {integrated_output.get('Annualized Volatility', 'N/A')} measures the degree of variation of a trading price series over time. Higher volatility means higher risk.\n"
        if "Average Closing Price" in integrated_output:
            report += f"- The Average Closing Price during the period was {integrated_output.get('Average Closing Price', 'N/A')}.\n"

        report += "\n\n---\n*Disclaimer: This report is generated by an AI tool and should not be considered financial advice. Always consult with a qualified financial professional.*"

        logging.info("Report generated successfully.")
        return report


# --- Streamlit Application ---
st.set_page_config(layout="wide", page_title="PAL Financial Analyzer")
st.title("📈 Program-Aided Language Models (PAL) Financial Analyzer")
st.markdown("This tool uses an AI to generate and execute code for accurate financial analysis.")

# Initialize components
data_retriever = DataRetriever()
code_executor = CodeExecutor()
llm_simulator = LLMSimulator()

user_query = st.text_area(
    "Enter your financial analysis request:",
    "Calculate the 5-year CAGR and Debt-to-Equity ratio for AAPL. Also, show its annualized volatility and average price."
)

if st.button("Analyze Financials"):
    if not user_query:
        st.warning("Please enter a financial analysis request.")
    else:
        with st.spinner("Interpreting request and performing analysis..."):
            try:
                # 1. Request Interpretation
                interpreted_request = llm_simulator.request_interpreter(user_query)

                # Check if ticker was found
                if not interpreted_request.get("ticker"):
                    st.error("Could not identify a stock ticker in your request. Please specify a valid ticker (e.g., AAPL, MSFT).")
                else:
                    # 2. Code Generation
                    generated_code = llm_simulator.code_generator(interpreted_request)

                    if "Error:" in generated_code:
                        st.error(generated_code.replace("print(\"", "").replace("\")", ""))
                    else:
                        # 3. Code Execution
                        stdout, stderr = code_executor.execute_code(generated_code)

                        if stderr:
                            st.error(f"Error during code execution:\n```\n{stderr}\n```")
                        elif not stdout:
                            st.error("Code execution returned no output.")
                        else:
                            # 4. Output Integration
                            integrated_output = llm_simulator.output_integrator(stdout)

                            # 5. Report Generation
                            final_report = llm_simulator.report_generator(integrated_output, user_query)

                            st.success("Analysis Complete!")
                            st.markdown(final_report)

                            st.subheader("Generated Code (for transparency):")
                            st.code(generated_code, language="python")

                            st.subheader("Raw Code Execution Output (for debugging):")
                            st.code(stdout, language="text")

            except Exception as e:
                logging.exception("An unexpected error occurred during analysis.")
                st.error(f"An unexpected error occurred: {e}. Please check the logs for details.")

