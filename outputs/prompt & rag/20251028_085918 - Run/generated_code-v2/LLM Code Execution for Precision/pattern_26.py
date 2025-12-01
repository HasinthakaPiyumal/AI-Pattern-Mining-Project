import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import subprocess
import json
import io
import base64
import numpy as np
import tempfile
import os


class FinancialDataConnector:
    def get_historical_data(self, ticker, start_date, end_date):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                return None
            return data
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return None


class BacktestingEngine:
    def run_strategy(self, data, strategy_code):
        # In a real scenario, this would execute the strategy_code dynamically.
        # For this demonstration, we'll simulate a simple SMA crossover.
        # The actual execution of LLM-generated code will happen via CodeExecutor.
        if data is None or data.empty:
            return None, None

        df = data.copy()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        df['Signal'] = 0
        df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] = 1  # Buy signal
        df.loc[df['SMA_20'] < df['SMA_50'], 'Signal'] = -1 # Sell signal

        # Generate trades based on signals
        df['Position'] = df['Signal'].diff()
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Returns'] * df['Position'].shift(1)

        # Calculate cumulative returns
        df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod() - 1
        return df['Cumulative_Strategy_Returns'].dropna(), df

    def calculate_metrics(self, cumulative_returns):
        if cumulative_returns is None or cumulative_returns.empty:
            return {"P&L": 0, "Sharpe Ratio": 0, "Max Drawdown": 0}

        total_return = cumulative_returns.iloc[-1] * 100
        daily_returns = cumulative_returns.diff().dropna()
        if daily_returns.std() == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

        # Max Drawdown
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min() * 100

        return {
            "P&L (%)": round(total_return, 2),
            "Sharpe Ratio": round(sharpe_ratio, 2),
            "Max Drawdown (%)": round(max_drawdown, 2)
        }

    def plot_equity_curve(self, cumulative_returns):
        if cumulative_returns is None or cumulative_returns.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        cumulative_returns.plot(ax=ax, title='Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns')
        ax.grid(True)
        
        # Convert plot to PNG image for display in Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64


class LLMOrchestrator:
    def generate_strategy_code(self, prompt):
        # SIMULATED LLM RESPONSE for generating Python code
        # In a real application, this would call an actual LLM API (e.g., OpenAI, Llama 2)
        # to generate a Python script based on the prompt.
        # The generated code should take 'data' (pandas DataFrame) as input
        # and return 'cumulative_strategy_returns' and the modified 'data' DataFrame.
        
        return """
import pandas as pd

def run_strategy_logic(data):
    df = data.copy()
    df['SMA_Short'] = df['Close'].rolling(window=20).mean()
    df['SMA_Long'] = df['Close'].rolling(window=50).mean()

    df['Signal'] = 0
    df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1  # Buy signal
    df.loc[df['SMA_Short'] < df['SMA_Long'], 'Signal'] = -1 # Sell signal

    df['Position'] = df['Signal'].diff().fillna(0)
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Position'].shift(1).fillna(0)

    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod() - 1
    return df['Cumulative_Strategy_Returns'].dropna(), df

# This part would be executed by the CodeExecutor, receiving 'data_input' as a JSON string
if __name__ == '__main__':
    import sys
    import json

    # Read input data (serialized DataFrame) from stdin
    input_json = sys.stdin.read()
    input_dict = json.loads(input_json)
    data_input = pd.read_json(input_dict['data'], orient='split')

    cumulative_returns, full_df = run_strategy_logic(data_input)

    # Prepare output: cumulative returns and any other relevant data
    output_data = {
        'cumulative_returns': cumulative_returns.to_json(orient='split'),
        # 'full_df': full_df.to_json(orient='split') # Can include full df if needed
    }
    print(json.dumps(output_data))
"""

    def analyze_results(self, backtest_metrics):
        # SIMULATED LLM RESPONSE for analyzing backtest results
        # In a real application, this would call an LLM API to interpret metrics
        # and provide human-readable recommendations/explanations.
        analysis = f"Based on the backtesting results: " \
                   f"P&L: {backtest_metrics['P&L (%)']}%, " \
                   f"Sharpe Ratio: {backtest_metrics['Sharpe Ratio']}, " \
                   f"Max Drawdown: {backtest_metrics['Max Drawdown (%)']}%.\n\n"
        
        if backtest_metrics['Sharpe Ratio'] > 0.5:
            analysis += "This strategy shows a reasonable risk-adjusted return and could be considered for further analysis. "
        elif backtest_metrics['Sharpe Ratio'] > 0:
            analysis += "The strategy has positive returns but the risk-adjusted performance (Sharpe Ratio) is modest. Consider optimizing parameters or adding more robust filters. "
        else:
            analysis += "The strategy has not performed well in backtesting. It is recommended to re-evaluate the strategy logic, market conditions, or parameters. "

        if backtest_metrics['Max Drawdown (%)'] < -15:
            analysis += "The maximum drawdown is significant, indicating high risk exposure. "

        analysis += "Further refinements could include dynamic position sizing, stop-loss/take-profit levels, or incorporating other technical indicators/fundamental analysis." 
        return analysis


def execute_python_code(code_string, input_data_df):
    # SECURITY WARNING: Executing arbitrary code from an LLM is a significant security risk.
    # For a production system, a highly secure, isolated sandboxing solution (e.g., Docker, dedicated VMs)
    # is essential. This implementation uses subprocess for demonstration purposes ONLY and is NOT secure.

    temp_file = None
    try:
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file_obj:
            temp_file_obj.write(code_string)
            temp_file = temp_file_obj.name

        # Prepare input data for the subprocess
        # Convert DataFrame to JSON string and then to a dictionary to pass via stdin
        input_dict = {'data': input_data_df.to_json(orient='split')}
        input_json_str = json.dumps(input_dict)
        
        # Execute the Python file using subprocess
        process = subprocess.run(
            ['python', temp_file],
            input=input_json_str,  # Pass input via stdin
            capture_output=True,
            text=True, # Decode stdout/stderr as text
            check=False # Don't raise an exception for non-zero exit codes
        )

        if process.returncode != 0:
            st.error(f"Code execution failed with error:\n{process.stderr}")
            return None

        try:
            output_data = json.loads(process.stdout)
            cumulative_returns_json = output_data.get('cumulative_returns')
            if cumulative_returns_json:
                cumulative_returns = pd.read_json(cumulative_returns_json, orient='split')
                return cumulative_returns
            else:
                st.warning("Code executed, but no cumulative_returns found in output.")
                return None
        except json.JSONDecodeError:
            st.error(f"Failed to parse JSON output from executed code. Raw output:\n{process.stdout}")
            return None

    except Exception as e:
        st.error(f"Error during code execution setup: {e}")
        return None
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


# Streamlit UI
def main():
    st.set_page_config(layout="wide", page_title="PAL Trading Strategy Generator")
    st.title("Algorithmic Trading Strategy Generator with PAL Prompting")

    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", "AAPL").upper()
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))
    
    user_prompt = st.text_area(
        "Describe your trading strategy (e.g., 'Generate a simple moving average crossover strategy for tech stocks with 20-day and 50-day periods.')",
        "Generate a simple moving average crossover strategy for the given stock with 20-day and 50-day periods."
    )

    if st.button("Generate and Backtest Strategy"):
        if not ticker:
            st.error("Please enter a stock ticker.")
            return
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            return

        st.subheader("1. Fetching Historical Data...")
        data_connector = FinancialDataConnector()
        data = data_connector.get_historical_data(ticker, start_date, end_date)

        if data is None or data.empty:
            st.warning("No data fetched or data is empty. Please check the ticker and date range.")
            return
        st.success("Historical data fetched successfully.")
        st.dataframe(data.tail())

        st.subheader("2. LLM Generating Strategy Code...")
        llm_orchestrator = LLMOrchestrator()
        generated_code = llm_orchestrator.generate_strategy_code(user_prompt)
        
        st.code(generated_code, language='python')

        st.subheader("3. Executing Generated Code (Backtesting)...")
        with st.spinner("Executing code and backtesting strategy..."):
            # Pass the fetched data to the executed script
            cumulative_returns = execute_python_code(generated_code, data)

            if cumulative_returns is not None and not cumulative_returns.empty:
                st.success("Code executed and backtesting completed.")

                st.subheader("4. Backtesting Results")
                backtesting_engine = BacktestingEngine()
                metrics = backtesting_engine.calculate_metrics(cumulative_returns)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Total P&L", value=f"{metrics['P&L (%)']:.2f} %")
                with col2:
                    st.metric(label="Sharpe Ratio", value=f"{metrics['Sharpe Ratio']:.2f}")
                with col3:
                    st.metric(label="Max Drawdown", value=f"{metrics['Max Drawdown (%)']:.2f} %")

                st.write("Equity Curve:")
                equity_curve_image = backtesting_engine.plot_equity_curve(cumulative_returns)
                if equity_curve_image:
                    st.image(f"data:image/png;base64,{equity_curve_image}", use_column_width=True)
                
                st.subheader("5. LLM Analysis and Recommendations")
                analysis = llm_orchestrator.analyze_results(metrics)
                st.write(analysis)
            else:
                st.error("Backtesting failed or returned no results.")

if __name__ == "__main__":
    main()
