import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import re

class MockOpenAI:
    def chat(self):
        return self

    def completions(self):
        return self

    def create(self, model, messages=None, prompt=None, max_tokens=None, stop=None, temperature=None):
        user_message = ""
        if messages:
            user_message = messages[-1]["content"]
        elif prompt:
            user_message = prompt

        if "generate python code" in user_message.lower():
            code = """
import numpy as np
import pandas as pd
import json

def calculate_portfolio_metrics(weights, returns, cov_matrix, risk_free_rate=0.01):
    portfolio_return = np.sum(returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return portfolio_return, portfolio_std_dev, sharpe_ratio

def perform_monte_carlo_simulation(num_portfolios, asset_returns, cov_matrix):
    num_assets = len(asset_returns)
    all_weights = np.zeros((num_portfolios, num_assets))
    ret_arr = np.zeros(num_portfolios)
    std_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        all_weights[i,:] = weights

        p_return, p_std_dev, p_sharpe = calculate_portfolio_metrics(weights, asset_returns, cov_matrix)
        ret_arr[i] = p_return
        std_arr[i] = p_std_dev
        sharpe_arr[i] = p_sharpe

    return pd.DataFrame({'Return': ret_arr, 'Volatility': std_arr, 'Sharpe Ratio': sharpe_arr, 'Weights': list(all_weights)})

# --- START OF SCRIPT EXECUTION ---
# In a real scenario, asset_returns and cov_matrix would be calculated from historical data
# or derived from more detailed user holdings information.
# For this demo, we use example values.
asset_returns = np.array([0.0005, 0.0007, 0.0004]) # Daily returns for 3 example assets
cov_matrix = np.array([
    [0.0001, 0.00005, 0.00003],
    [0.00005, 0.00015, 0.00004],
    [0.00003, 0.00004, 0.00008]
])
num_portfolios_to_simulate = 2000 # Increased for better simulation

simulation_results_df = perform_monte_carlo_simulation(num_portfolios_to_simulate, asset_returns, cov_matrix)

# Find the optimal portfolio (max Sharpe Ratio)
optimal_portfolio = simulation_results_df.loc[simulation_results_df['Sharpe Ratio'].idxmax()]

output = {
    "optimal_portfolio_return": float(optimal_portfolio['Return']),
    "optimal_portfolio_volatility": float(optimal_portfolio['Volatility']),
    "optimal_portfolio_sharpe_ratio": float(optimal_portfolio['Sharpe Ratio']),
    "optimal_portfolio_weights": optimal_portfolio['Weights'].tolist(),
    "all_simulation_results_json": simulation_results_df.to_json(orient="split") # Store as JSON string
}
            """
            return {"choices": [{"message": {"content": code}}]}
        elif "explain portfolio optimization results" in user_message.lower():
            try:
                risk_tolerance_match = re.search(r"with a '(\w+)' risk tolerance", user_message)
                financial_goal_match = re.search(r"and a goal to '([^']+)'", user_message)

                risk_tolerance_str = risk_tolerance_match.group(1) if risk_tolerance_match else "moderate"
                financial_goal_str = financial_goal_match.group(1) if financial_goal_match else "optimize your investments"

                results_json_str_match = re.search(r"Results: ({.*})", user_message, re.DOTALL)
                results = {}
                if results_json_str_match:
                    results_json_str = results_json_str_match.group(1)
                    results = json.loads(results_json_str)

                optimal_return = results.get("optimal_portfolio_return", 0.0)
                optimal_volatility = results.get("optimal_portfolio_volatility", 0.0)
                optimal_sharpe_ratio = results.get("optimal_portfolio_sharpe_ratio", 0.0)
                optimal_weights = results.get("optimal_portfolio_weights", [])

                explanation = f"""
Based on the sophisticated analysis and considering your goal to '{financial_goal_str}' with a '{risk_tolerance_str}' risk profile, your optimal investment strategy is designed to achieve the following:

Key Metrics for Your Optimized Portfolio:
- **Expected Annual Return:** {optimal_return:.2%} – This is the anticipated annual growth of your investment, reflecting the potential gains.
- **Annual Volatility (Risk):** {optimal_volatility:.2%} – This indicates the potential fluctuation of your portfolio's value over a year. With a '{risk_tolerance_str.lower()}' risk profile, this level of volatility is considered manageable for your investment horizon.
- **Sharpe Ratio:** {optimal_sharpe_ratio:.2f} – This critical metric measures your risk-adjusted return. A higher Sharpe ratio suggests you are getting more return for each unit of risk taken, making this an efficient portfolio.

The recommended asset allocation for achieving these targets is approximately:
{', '.join([f'Asset {i+1}: {w:.2%}' for i, w in enumerate(optimal_weights)])}.
(Note: Asset 1, 2, 3 represent generic assets in this simulation; in a real application, these would be specific stocks/ETFs based on your holdings input.)

This allocation aims to provide a robust balance, leveraging market opportunities while managing potential downturns, aligning with your objective to '{financial_goal_str}'. Please remember that past performance is not indicative of future results, and market conditions can change. It is always wise to periodically review and adjust your portfolio as your goals and market conditions evolve.
"""
            except Exception as e:
                explanation = f"I encountered an issue while processing the results for explanation: {e}. Please ensure the results are in a valid format for explanation. Original message: {user_message}"

            return {"choices": [{"message": {"content": explanation}}]}
        return {"choices": [{"message": {"content": "I am an AI assistant and can help with financial portfolio optimization. Please provide your financial goals, risk tolerance, and current holdings."}}]}

openai_client = MockOpenAI()

st.set_page_config(layout="wide")
st.title("Financial Portfolio Optimizer (PAL Prompting Demo)")
st.markdown("""
This application demonstrates the **Program-Aided Language Models (PAL) Prompting** pattern.
The workflow is as follows:
1.  **User Input:** You provide your financial goals, risk tolerance, and current holdings.
2.  **LLM Generates Code:** A Language Model (simulated here) receives your input and generates executable Python code tailored for financial calculations (e.g., Monte Carlo simulations for portfolio optimization).
3.  **Execute Generated Code:** An external Python interpreter executes this code.
4.  **Numerical Output:** The execution yields precise numerical results (e.g., optimal portfolio weights, returns, volatility, Sharpe ratio).
5.  **LLM Formulates Final Answer:** These numerical results are then fed back to the Language Model, which uses them to formulate a natural language explanation and personalized recommendations.

This approach leverages the LLM's natural language understanding and generation capabilities with the precision and computational power of programming languages.
""")

st.header("1. User Input")
with st.form("portfolio_form"):
    financial_goal = st.text_area("What are your primary financial goals?", "Grow wealth over 10 years for retirement, aiming for a consistent annual return of 7-10%.")
    risk_tolerance = st.selectbox("What is your risk tolerance?", ["Low", "Medium", "High"], index=1)
    current_holdings = st.text_area("Describe your current holdings or preferred asset classes (e.g., '60% stocks (tech, S&P 500), 30% bonds (government), 10% real estate ETF').", "Currently holding 60% in diverse equity ETFs and 40% in short-term government bonds.")
    
    submitted = st.form_submit_button("Optimize Portfolio")

if submitted:
    st.markdown("---")
    st.header("2. LLM Generates Code (PAL Prompting in action)")
    st.info("Simulating LLM generating Python code for portfolio optimization based on your inputs. This code will perform a Monte Carlo simulation.")

    llm_code_generation_prompt = f"""
    You are an expert financial quantitative analyst. Based on the user's financial goals, risk tolerance, and current holdings, generate Python code to perform a Monte Carlo simulation for portfolio optimization.
    The code should calculate portfolio return, volatility, and Sharpe ratio for various asset allocations.
    User's financial goal: {financial_goal}
    User's risk tolerance: {risk_tolerance}
    User's current holdings: {current_holdings}
    Ensure the output of the script is a dictionary named 'output' containing the optimal portfolio metrics (return, volatility, sharpe ratio, weights) and optionally all simulation results as JSON.
    Generate python code only, no additional text, comments or docstrings outside the functions. The script should be self-contained and ready for execution.
    """
    
    try:
        response_code = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": llm_code_generation_prompt}],
            temperature=0.0
        )
        generated_code = response_code["choices"][0]["message"]["content"]
        st.subheader("Generated Python Code:")
        st.code(generated_code, language="python")

        st.markdown("---")
        st.header("3. Execute Generated Code")
        st.info("Executing the generated Python code in an external interpreter to obtain precise numerical results...")

        script_globals = {}
        script_locals = {}

        old_stdout = io.StringIO()
        sys.stdout = old_stdout

        try:
            exec(generated_code, script_globals, script_locals)
            st.success("Python code executed successfully!")
            
            execution_results = script_locals.get("output", {})
            
            sys.stdout = sys.__stdout__
            if old_stdout.getvalue():
                st.subheader("Script Console Output (if any):")
                st.text(old_stdout.getvalue())

            if execution_results:
                st.subheader("Numerical Results from Code Execution:")
                st.json(execution_results)

                st.markdown("---")
                st.header("4. LLM Formulates Final Answer (Explanation & Recommendations)")
                st.info("Feeding numerical results back to the LLM for natural language explanation and personalized recommendations...")

                llm_explanation_prompt = f"""
                Explain the following portfolio optimization results in natural language, suitable for an investor with a '{risk_tolerance}' risk tolerance and a goal to '{financial_goal}'.
                Provide a concise summary and personalized recommendations based on these results.
                Results: {json.dumps(execution_results)}
                """
                response_explanation = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": llm_explanation_prompt}],
                    temperature=0.7
                )
                explanation = response_explanation["choices"][0]["message"]["content"]
                st.subheader("Personalized Portfolio Explanation and Recommendations:")
                st.write(explanation)
            else:
                st.error("No valid 'output' dictionary found from code execution to generate an explanation.")

        except Exception as e:
            sys.stdout = sys.__stdout__
            st.error(f"Error executing generated Python code: {e}")
            st.exception(e)
            if old_stdout.getvalue():
                st.subheader("Script Console Output (before error):")
                st.text(old_stdout.getvalue())

    except Exception as e:
        st.error(f"Error during LLM code generation simulation: {e}")
        st.exception(e)
