import streamlit as st
import openai
import os
import subprocess
import json
import sqlite3
import pandas as pd
import numpy as np # Needed for potential direct use in interpreter or as context for LLM
from scipy.optimize import minimize # Needed for potential direct use in interpreter or as context for LLM
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI # Updated import for OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    st.error("OPENAI_API_KEY not found in environment variables. Please set it.")

# --- Database Setup ---
DB_NAME = "portfolio_optimizer.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS optimizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            capital REAL,
            risk_tolerance TEXT,
            horizon TEXT,
            generated_code TEXT,
            optimization_results TEXT,
            llm_explanation TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_optimization_results(capital, risk_tolerance, horizon, generated_code, optimization_results, llm_explanation):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO optimizations (capital, risk_tolerance, horizon, generated_code, optimization_results, llm_explanation)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (capital, risk_tolerance, horizon, generated_code, json.dumps(optimization_results), llm_explanation))
    conn.commit()
    conn.close()

# --- LLM Orchestrator and Code Generation ---
llm_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=OPENAI_API_KEY) # Using gpt-3.5-turbo as it's typically faster and cheaper for initial demos

CODE_GEN_TEMPLATE = """
You are an expert financial analyst and a Python programmer.
Your task is to generate Python code to perform portfolio optimization based on user inputs.
The user wants to optimize a portfolio given a set of assets, expected returns, and covariance matrix.
The optimization should aim to find asset weights that either maximize the Sharpe ratio or minimize volatility for a target return.
Assume the following data is available:
- `expected_returns`: A list or numpy array of expected returns for each asset.
- `cov_matrix`: A numpy array representing the covariance matrix of asset returns.
- `risk_free_rate`: A float representing the risk-free rate.

The generated code MUST:
1.  Define a function `optimize_portfolio(expected_returns, cov_matrix, risk_free_rate)` that returns a dictionary with the following keys:
    - `"optimal_weights"`: A list of optimal asset weights.
    - `"portfolio_return"`: The expected return of the optimized portfolio.
    - `"portfolio_volatility"`: The volatility (standard deviation) of the optimized portfolio.
    - `"sharpe_ratio"`: The Sharpe ratio of the optimized portfolio.
2.  Use `numpy` for array operations and `scipy.optimize.minimize` for optimization.
3.  Include constraints that weights sum to 1 and are non-negative.
4.  Print the results dictionary to standard output using `json.dumps()`.

Here is an example structure for the `optimize_portfolio` function using Markowitz optimization (feel free to adapt for maximizing Sharpe ratio):

```python
import numpy as np
from scipy.optimize import minimize
import json

def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate):
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix, risk_free_rate)

    # Function to calculate portfolio return
    def portfolio_return(weights, expected_returns):
        return np.sum(expected_returns * weights)

    # Function to calculate portfolio volatility
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Function to maximize Sharpe Ratio (negative for minimization)
    def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
        p_return = portfolio_return(weights, expected_returns)
        p_volatility = portfolio_volatility(weights, cov_matrix)
        return -(p_return - risk_free_rate) / p_volatility

    # Constraints
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]

    # Optimize for maximum Sharpe Ratio
    opt_results = minimize(neg_sharpe_ratio, initial_weights, args=args,
                           method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = opt_results.x
    p_return = portfolio_return(optimal_weights, expected_returns)
    p_volatility = portfolio_volatility(optimal_weights, cov_matrix)
    sharpe_r = (p_return - risk_free_rate) / p_volatility

    return {
        "optimal_weights": optimal_weights.tolist(),
        "portfolio_return": p_return,
        "portfolio_volatility": p_volatility,
        "sharpe_ratio": sharpe_r
    }

# Example usage (will be replaced by dynamic data in actual execution)
# expected_returns = np.array([0.10, 0.15, 0.20])
# cov_matrix = np.array([
#     [0.01, 0.005, 0.002],
#     [0.005, 0.02, 0.007],
#     [0.002, 0.007, 0.03]
# ])
# risk_free_rate = 0.03
# results = optimize_portfolio(expected_returns, cov_matrix, risk_free_rate)
# print(json.dumps(results))
```

Please generate the `optimize_portfolio` function correctly within the provided structure.
The specific assets, expected returns, and covariance matrix will be dynamically provided when executing the generated code.
"""

code_gen_prompt = PromptTemplate(template=CODE_GEN_TEMPLATE, input_variables=[])

def generate_optimization_code():
    # In a real application, you might provide context to the LLM based on user input or available data.
    # For this example, we're asking for a generic optimization function.
    # The actual data (expected_returns, cov_matrix, risk_free_rate) will be injected at execution.
    try:
        response = llm_model.invoke(code_gen_prompt.format())
        return response.content
    except Exception as e:
        st.error(f"Error generating code: {e}")
        return None

# --- Code Execution Environment ---
def execute_generated_code(code_string, expected_returns, cov_matrix, risk_free_rate):
    """
    Executes the generated Python code in a sandboxed environment and captures its output.
    Injects dynamic data (expected_returns, cov_matrix, risk_free_rate) into the script.
    """
    try:
        # Create a temporary Python file
        temp_script_path = "temp_optimization_script.py"
        with open(temp_script_path, "w") as f:
            f.write(code_string)
            f.write("\n\n")
            f.write(f"expected_returns = np.array({expected_returns.tolist()})\n")
            f.write(f"cov_matrix = np.array({cov_matrix.tolist()})\n")
            f.write(f"risk_free_rate = {risk_free_rate}\n")
            f.write("results = optimize_portfolio(expected_returns, cov_matrix, risk_free_rate)\n")
            f.write("print(json.dumps(results))\n")

        # Execute the script using subprocess
        # IMPORTANT: In a production environment, implement robust sandboxing
        # (e.g., using a container, restricted execution environment)
        # to prevent malicious code execution.
        process = subprocess.run(
            ["python", temp_script_path],
            capture_output=True,
            text=True,
            check=True, # Raise an exception for non-zero exit codes
            timeout=30 # Add a timeout
        )

        os.remove(temp_script_path) # Clean up temporary file

        output_json = process.stdout.strip()
        st.write(f"Raw code execution output: {output_json}") # Debugging
        return json.loads(output_json)

    except subprocess.CalledProcessError as e:
        st.error(f"Error during code execution: {e.stderr}")
        return {"error": e.stderr}
    except json.JSONDecodeError:
        st.error(f"Failed to parse JSON output from code execution. Output: {output_json}")
        return {"error": "JSON decoding failed", "raw_output": output_json}
    except FileNotFoundError:
        st.error("Python interpreter not found. Please ensure Python is installed and in your PATH.")
        return {"error": "Python interpreter not found"}
    except Exception as e:
        st.error(f"An unexpected error occurred during code execution: {e}")
        return {"error": str(e)}

# --- LLM Interpretation Module ---
INTERPRETATION_TEMPLATE = """
You are an AI financial advisor. Based on the following portfolio optimization results, provide a clear, concise, and professional explanation to the user.
Explain the optimal asset allocation, the reasoning behind it, the expected performance metrics, and any relevant market insights or risk considerations.
The user's investment capital is ${capital}, their risk tolerance is {risk_tolerance}, and their investment horizon is {investment_horizon}.

Optimization Results:
{optimization_results}

Please structure your explanation as follows:
1.  **Summary of Investment Strategy**: Briefly explain the overall approach based on their risk tolerance.
2.  **Optimal Portfolio Allocation**: Detail the percentage allocation for each asset.
3.  **Performance Metrics**: Explain the expected return, volatility (risk), and Sharpe Ratio.
4.  **Risk and Market Insights**: Discuss potential risks and how the portfolio addresses them, or broader market considerations.
5.  **Disclaimer**: Add a standard financial advice disclaimer.
"""

interpretation_prompt = PromptTemplate(
    template=INTERPRETATION_TEMPLATE,
    input_variables=["capital", "risk_tolerance", "investment_horizon", "optimization_results"]
)

def interpret_results(capital, risk_tolerance, investment_horizon, optimization_results):
    try:
        response = llm_model.invoke(
            interpretation_prompt.format(
                capital=capital,
                risk_tolerance=risk_tolerance,
                investment_horizon=investment_horizon,
                optimization_results=json.dumps(optimization_results, indent=2)
            )
        )
        return response.content
    except Exception as e:
        st.error(f"Error interpreting results: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="CAR Financial Advisor", layout="wide")

def main():
    init_db()
    st.title("💰 Code-Assisted Financial Portfolio Optimizer")
    st.write("Leveraging LLMs and code execution for precise financial recommendations.")

    with st.sidebar:
        st.header("Your Investment Profile")
        investment_capital = st.number_input(
            "Investment Capital ($)",
            min_value=1000,
            value=10000,
            step=1000
        )
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ["Low", "Medium", "High"]
        )
        investment_horizon = st.selectbox(
            "Investment Horizon",
            ["Short-term (1-3 years)", "Medium-term (3-7 years)", "Long-term (7+ years)"]
        )
        st.markdown("---")
        st.header("Asset Data (Example)")
        st.write("For this demo, we use a fixed set of example assets.")

        # Example asset data (can be made dynamic if needed)
        assets = ["Asset A", "Asset B", "Asset C", "Asset D"]
        expected_returns_data = np.array([0.08, 0.12, 0.15, 0.10])
        cov_matrix_data = np.array([
            [0.005, 0.001, 0.002, 0.001],
            [0.001, 0.010, 0.003, 0.002],
            [0.002, 0.003, 0.015, 0.004],
            [0.001, 0.002, 0.004, 0.008]
        ])
        risk_free_rate_data = 0.02

        st.subheader("Expected Returns")
        st.dataframe(pd.DataFrame({"Asset": assets, "Expected Return": expected_returns_data}))
        st.subheader("Covariance Matrix")
        st.dataframe(pd.DataFrame(cov_matrix_data, index=assets, columns=assets))
        st.write(f"Risk-Free Rate: {risk_free_rate_data * 100:.2f}%")


    st.header("Optimization Results & Recommendations")
    if st.button("Generate Portfolio Recommendation"):
        with st.spinner("Generating optimization code..."):
            generated_code = generate_optimization_code()

        if generated_code:
            st.subheader("Generated Python Code for Optimization")
            st.code(generated_code, language="python")

            with st.spinner("Executing optimization code..."):
                optimization_results = execute_generated_code(
                    generated_code,
                    expected_returns_data,
                    cov_matrix_data,
                    risk_free_rate_data
                )

            if optimization_results and "error" not in optimization_results:
                st.subheader("Portfolio Optimization Results")
                st.json(optimization_results)

                with st.spinner("Interpreting results and generating recommendations..."):
                    llm_explanation = interpret_results(
                        investment_capital,
                        risk_tolerance,
                        investment_horizon,
                        optimization_results
                    )

                if llm_explanation:
                    st.subheader("Financial Advisor's Recommendation")
                    st.markdown(llm_explanation)

                    # Save results to DB
                    save_optimization_results(
                        investment_capital,
                        risk_tolerance,
                        investment_horizon,
                        generated_code,
                        optimization_results,
                        llm_explanation
                    )
                    st.success("Recommendation saved successfully!")
                else:
                    st.error("Could not get LLM interpretation.")
            else:
                st.error("Could not execute optimization code or parse results.")
        else:
            st.error("Could not generate optimization code.")

    st.markdown("---")
    st.subheader("Previous Optimizations")
    conn = sqlite3.connect(DB_NAME)
    df_history = pd.read_sql_query("SELECT * FROM optimizations ORDER BY timestamp DESC LIMIT 10", conn)
    conn.close()

    if not df_history.empty:
        st.dataframe(df_history[['timestamp', 'capital', 'risk_tolerance', 'horizon', 'generated_code', 'optimization_results', 'llm_explanation']]) # Display summary
        if st.checkbox("Show full history details"):
            for index, row in df_history.iterrows():
                st.subheader(f"Optimization {row['id']} on {row['timestamp']}")
                st.write(f"**Capital:** ${row['capital']}, **Risk Tolerance:** {row['risk_tolerance']}, **Horizon:** {row['horizon']}")
                st.code(row['generated_code'], language="python")
                st.json(json.loads(row['optimization_results']))
                st.markdown(row['llm_explanation'])
                st.markdown("---")
    else:
        st.info("No previous optimizations found.")

if __name__ == "__main__":
    main()