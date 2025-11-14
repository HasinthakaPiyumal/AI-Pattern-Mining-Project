import streamlit as st
import openai
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize # Ensure this is imported for exec context
import io
import contextlib
import re
import os

# --- Configuration ---
# Set your OpenAI API key here or as an environment variable
# It's highly recommended to set it as an environment variable for security
# Example: os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key.")
    st.stop()

# --- LLM Interaction Functions ---

def call_llm(prompt, model="gpt-4", max_tokens=1500, temperature=0.7):
    """Makes an API call to the OpenAI LLM."""
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Respond concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return None

def generate_code_prompt(goal, risk, capital):
    """Generates a prompt for the LLM to create Python code for portfolio optimization."""
    prompt = f"""
    You are an expert financial analyst. A user wants to optimize their investment portfolio.
    Their investment goal is: {goal}
    Their risk tolerance is: {risk}
    Their initial capital is: ${capital:,.2f}

    Generate Python code to perform the following steps for Markowitz Mean-Variance Optimization:
    1.  **Define a list of at least 5 diverse common stock tickers** (e.g., AAPL, MSFT, GOOGL, AMZN, JPM, VOO, SPG).
    2.  **Fetch historical adjusted close prices** for these tickers for the last 5 years using `yfinance`.
        Handle potential data fetching errors (e.g., `yf.download` returning empty DataFrame).
    3.  **Calculate daily returns** for each stock.
    4.  **Calculate the annualized mean returns and annualized covariance matrix** of the stock returns. Assume 252 trading days per year.
    5.  **Define a function `portfolio_volatility(weights, cov_matrix)`** to calculate portfolio volatility.
    6.  **Define a function `portfolio_return(weights, mean_returns)`** to calculate portfolio returns.
    7.  **Define a function `sharpe_ratio(returns, volatility, risk_free_rate)`** to calculate the Sharpe Ratio (use a `risk_free_rate` of 0.01).
    8.  **Perform Markowitz Mean-Variance Optimization** to find optimal portfolio weights that **minimize volatility**.
        *   The sum of weights must be 1 (constraint `{{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}}`).
        *   Each weight must be between 0 and 1 (bounds `(0, 1)` for each weight).
        *   Use `scipy.optimize.minimize`.
    9.  **Calculate the optimized portfolio's expected annual return, annual volatility, and Sharpe ratio.**
    10. **Print the results clearly.**
        Include:
        *   A dictionary of optimized portfolio weights (ticker to percentage, e.g., 'AAPL': 25.5%).
        *   Expected Annual Return (%).
        *   Annual Volatility (%).
        *   Sharpe Ratio.
        Example output format:
        "Optimized Weights: {{'AAPL': 0.3, 'MSFT': 0.2, ...}}\nExpected Annual Return: 12.34%\nAnnual Volatility: 15.67%\nSharpe Ratio: 0.78"
    Ensure the code is self-contained and runnable. Use `pandas`, `numpy`, `yfinance`, and `scipy.optimize.minimize`.
    The output should only contain the python code, no extra text or markdown comments outside the code block.
    """
    return prompt

def generate_recommendation_prompt(user_inputs, execution_results):
    """Generates a prompt for the LLM to provide investment recommendations."""
    prompt = f"""
    Based on the following user investment profile and the precise numerical results from a portfolio optimization script,
    provide personalized, actionable investment recommendations and comprehensive explanations.

    **User Investment Profile:**
    - Investment Goal: {user_inputs['goal']}
    - Risk Tolerance: {user_inputs['risk']}
    - Initial Capital: ${user_inputs['capital']:,.2f}

    **Portfolio Optimization Results (JSON-like format for easy parsing, but treat as text if not perfect JSON):**
    ```
    {execution_results}
    ```

    **Your Task:**
    1.  **Summarize the Key Findings:** Briefly explain what the optimization results indicate (e.g., optimal asset allocation, expected return, risk level, Sharpe ratio).
    2.  **Provide Actionable Investment Recommendations:** Based on the user's goal, risk tolerance, and the optimized portfolio.
        *   Suggest how the user's initial capital could be distributed according to the optimal weights.
        *   Discuss the implications of the expected return and volatility in relation to their goal/risk.
    3.  **Explain the Rationale:** Clearly articulate why these recommendations are suitable, linking them to the optimization outcomes and sound financial principles.
    4.  **Include Important Considerations & Warnings:**
        *   Emphasize that this is not personal financial advice.
        *   Mention market volatility, diversification benefits, and the importance of a long-term perspective.
        *   Advise seeking a professional financial advisor.
    5.  **Format:** The response should be in clear, concise natural language, professional in tone, and well-structured with headings or bullet points where appropriate.
    """
    return prompt

# --- Code Execution Function ---

def execute_generated_code(code_string):
    """
    Executes the given Python code string in a limited scope and captures its output.
    WARNING: Using exec() with untrusted code is a significant security risk.
    For a production system, a more robust and isolated sandbox environment (e.g., a separate container,
    a dedicated execution service, or a secure sandboxing library) is absolutely crucial.
    This implementation is for demonstration purposes and provides only basic isolation.
    """
    # Create a dictionary for the execution scope
    # Only allow safe built-ins and specific necessary modules
    safe_globals = {
        "__builtins__": {
            "print": print,
            "str": str, "int": int, "float": float, "bool": bool,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "len": len, "min": min, "max": max, "sum": sum,
            "range": range, "abs": abs, "round": round,
            "Exception": Exception # Allow exceptions for error handling within generated code
        },
        "pd": pd,
        "np": np,
        "yf": yf,
        "minimize": minimize, # from scipy.optimize
        "io": io, # for capturing output
        "contextlib": contextlib, # for capturing output
    }
    local_vars = {} # Empty locals for this demonstration

    # Capture stdout
    redirected_stdout = io.StringIO()

    try:
        with contextlib.redirect_stdout(redirected_stdout):
            exec(code_string, safe_globals, local_vars)
        output = redirected_stdout.getvalue()
        return True, output
    except Exception as e:
        error_output = redirected_stdout.getvalue() # Capture any output before the error
        return False, f"Code execution failed: {e}\nTraceback:\n{error_output}"
    finally:
        pass # Nothing to restore, stdout is already managed by contextlib

# --- Streamlit Application ---

st.set_page_config(layout="wide")
st.title("💰 Program-Aided Financial Advisor")
st.markdown("This tool leverages AI to generate and execute financial analysis code for personalized investment insights.")
st.markdown("---")

st.sidebar.header("Your Investment Profile")
capital = st.sidebar.number_input("Initial Capital ($)", min_value=100.0, value=10000.0, step=100.0)
goal = st.sidebar.selectbox("Investment Goal", ["Growth", "Income", "Balanced", "Capital Preservation"])
risk = st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"])

user_inputs = {
    "capital": capital,
    "goal": goal,
    "risk": risk
}

if st.sidebar.button("Get Investment Recommendation"):
    if not openai.api_key:
        st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    else:
        st.session_state["generated_code"] = None
        st.session_state["execution_output"] = None
        st.session_state["final_recommendations"] = None

        with st.spinner("Step 1: Generating financial analysis code..."):
            code_prompt = generate_code_prompt(goal, risk, capital)
            generated_code_raw = call_llm(code_prompt, model="gpt-4", temperature=0.1) # Lower temp for code generation

            if generated_code_raw:
                # Extract code block if LLM wraps it in markdown
                match = re.search(r"```python\n(.*?)```", generated_code_raw, re.DOTALL)
                if match:
                    generated_code = match.group(1).strip()
                else:
                    generated_code = generated_code_raw.strip() # Assume it's just code if no markdown

                st.session_state["generated_code"] = generated_code
                st.subheader("🤖 Generated Python Code for Portfolio Optimization")
                st.code(generated_code, language="python")

                with st.spinner("Step 2: Executing generated code... (This can take a moment due to yfinance data fetching)"):
                    success, execution_output = execute_generated_code(generated_code)
                    st.session_state["execution_output"] = execution_output

                    if success:
                        st.subheader("✅ Code Execution Results")
                        st.text(execution_output)

                        with st.spinner("Step 3: Generating final investment recommendations..."):
                            recommendation_prompt = generate_recommendation_prompt(user_inputs, execution_output)
                            final_recommendations = call_llm(recommendation_prompt, model="gpt-4", temperature=0.7) # Higher temp for natural language
                            st.session_state["final_recommendations"] = final_recommendations

                            if final_recommendations:
                                st.subheader("✨ Your Personalized Investment Recommendations")
                                st.write(final_recommendations)
                            else:
                                st.error("Failed to generate final recommendations.")
                    else:
                        st.error("❌ Error during code execution:")
                        st.error(execution_output)
            else:
                st.error("❌ Failed to generate Python code.")

st.markdown("---")
st.info("""
**⚠️ Disclaimer:** This tool is for informational and educational purposes only and does not constitute financial advice.
Investment involves risks, including possible loss of principal. Always consult with a qualified financial professional before making any investment decisions.
The accuracy of recommendations depends on the underlying data, algorithms, and the capabilities of the Large Language Model.
The code execution environment used here is for demonstration and provides basic isolation. In a production setting, a much more robust and secure sandbox would be essential.
""")