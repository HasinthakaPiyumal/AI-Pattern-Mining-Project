import streamlit as st
import pandas as pd
import numpy as np
import io
import contextlib
import re

# --- Simulated LLM Core Functions (Conceptual) ---

def llm_generate_code(query: str) -> str:
    """
    Simulates an LLM generating Python code based on a financial query.
    In a real application, this would involve a sophisticated LLM call.
    """
    query_lower = query.lower()

    if "discounted cash flow" in query_lower or "dcf" in query_lower:
        # Extract cash flows, discount rate, and optionally growth rate
        cash_flows_match = re.search(r"cash flows\s*\[([\d,\s.]+)\]", query_lower)
        discount_rate_match = re.search(r"discount rate of (\d*\.?\d+)", query_lower)
        
        cash_flows_str = cash_flows_match.group(1) if cash_flows_match else "100, 110, 120"
        cash_flows = [float(x.strip()) for x in cash_flows_str.split(',')]
        discount_rate = float(discount_rate_match.group(1)) if discount_rate_match else 0.08
        
        code = f"""
import numpy as np

def calculate_dcf(cash_flows, discount_rate):
    npv = 0
    for i, cf in enumerate(cash_flows):
        npv += cf / (1 + discount_rate)**(i + 1)
    return npv

cash_flows = {cash_flows}
discount_rate = {discount_rate}
result = calculate_dcf(cash_flows, discount_rate)
print(f"Discounted Cash Flow (NPV): {{result:.2f}}")
"""
        return code

    elif "monte carlo" in query_lower or "stock simulation" in query_lower:
        # Extract initial price, daily return, volatility, days, simulations
        initial_price_match = re.search(r"initial price (\d*\.?\d+)", query_lower)
        daily_return_match = re.search(r"daily return (\d*\.?\d+)", query_lower)
        volatility_match = re.search(r"volatility (\d*\.?\d+)", query_lower)
        days_match = re.search(r"for (\d+) days", query_lower)
        simulations_match = re.search(r"(\d+) simulations", query_lower)

        initial_price = float(initial_price_match.group(1)) if initial_price_match else 100.0
        daily_return = float(daily_return_match.group(1)) if daily_return_match else 0.0005
        volatility = float(volatility_match.group(1)) if volatility_match else 0.02
        num_days = int(days_match.group(1)) if days_match else 252
        num_simulations = int(simulations_match.group(1)) if simulations_match else 1000

        code = f"""
import numpy as np

def monte_carlo_simulation(initial_price, daily_return, volatility, num_days, num_simulations):
    dt = 1 # daily step
    price_paths = np.zeros((num_days + 1, num_simulations))
    price_paths[0] = initial_price

    for t in range(1, num_days + 1):
        rand = np.random.standard_normal(num_simulations)
        price_paths[t] = price_paths[t-1] * np.exp((daily_return - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * rand)
    
    final_prices = price_paths[-1, :]
    return final_prices

initial_price = {initial_price}
daily_return = {daily_return}
volatility = {volatility}
num_days = {num_days}
num_simulations = {num_simulations}

final_prices = monte_carlo_simulation(initial_price, daily_return, volatility, num_days, num_simulations)
print(f"Average final price: {{np.mean(final_prices):.2f}}")
print(f"Median final price: {{np.median(final_prices):.2f}}")
print(f"Min final price: {{np.min(final_prices):.2f}}")
print(f"Max final price: {{np.max(final_prices):.2f}}")
print(f"Standard deviation of final prices: {{np.std(final_prices):.2f}}")
"""
        return code

    else:
        return "print(\"I cannot generate code for this specific financial query yet. Please try 'Discounted Cash Flow' or 'Monte Carlo Simulation'.\")"

def execute_code(code_string: str) -> tuple[str, str]:
    """
    Executes the given Python code string in a sandboxed environment
    and captures its output and any errors.
    """
    old_stdout = io.StringIO()
    old_stderr = io.StringIO()
    result_output = ""
    error_output = ""

    try:
        with contextlib.redirect_stdout(old_stdout):
            with contextlib.redirect_stderr(old_stderr):
                exec(code_string, {'np': np, 'pd': pd})
        result_output = old_stdout.getvalue()
        error_output = old_stderr.getvalue()
    except Exception as e:
        error_output = f"Execution Error: {e}\n{old_stderr.getvalue()}"

    return result_output.strip(), error_output.strip()

def llm_interpret_results(numerical_output: str, query: str) -> str:
    """
    Simulates an LLM interpreting numerical results and providing recommendations.
    """
    query_lower = query.lower()

    if "discounted cash flow" in query_lower or "dcf" in query_lower:
        if "Discounted Cash Flow (NPV):" in numerical_output:
            npv_str = numerical_output.split("Discounted Cash Flow (NPV): ")[1].strip()
            try:
                npv = float(npv_str)
                if npv > 0:
                    return f"Based on the Discounted Cash Flow (DCF) analysis, the Net Present Value (NPV) is {npv:.2f}. This positive NPV suggests that the project or investment is expected to generate more value than its cost, making it potentially attractive. Further due diligence is recommended.\n\n**Recommendation:** Consider this investment favorably, but cross-reference with other valuation methods and market conditions."
                elif npv < 0:
                    return f"Based on the Discounted Cash Flow (DCF) analysis, the Net Present Value (NPV) is {npv:.2f}. This negative NPV indicates that the project or investment might not generate sufficient returns to cover its costs. \n\n**Recommendation:** Exercise caution or reconsider this investment, as it may destroy value."
                else:
                    return f"Based on the Discounted Cash Flow (DCF) analysis, the Net Present Value (NPV) is {npv:.2f}. This neutral NPV suggests the project might break even. \n\n**Recommendation:** Further analysis is needed to determine marginal benefits and risks."
            except ValueError:
                return f"Could not parse DCF result for interpretation. Raw output: {numerical_output}"
        else:
            return f"Could not find DCF result in output for interpretation. Raw output: {numerical_output}"

    elif "monte carlo" in query_lower or "stock simulation" in query_lower:
        if "Average final price:" in numerical_output:
            try:
                avg_price = float(re.search(r"Average final price: ([\d.]+)", numerical_output).group(1))
                median_price = float(re.search(r"Median final price: ([\d.]+)", numerical_output).group(1))
                min_price = float(re.search(r"Min final price: ([\d.]+)", numerical_output).group(1))
                max_price = float(re.search(r"Max final price: ([\d.]+)", numerical_output).group(1))
                std_dev = float(re.search(r"Standard deviation of final prices: ([\d.]+)", numerical_output).group(1))

                interpretation = f"**Monte Carlo Simulation Results:**\n"
                interpretation += f"- Average Simulated Final Price: ${avg_price:.2f}\n"
                interpretation += f"- Median Simulated Final Price: ${median_price:.2f}\n"
                interpretation += f"- Minimum Simulated Final Price: ${min_price:.2f}\n"
                interpretation += f"- Maximum Simulated Final Price: ${max_price:.2f}\n"
                interpretation += f"- Standard Deviation of Final Prices: ${std_dev:.2f}\n\n"
                
                risk_assessment = ""
                if std_dev > (0.05 * avg_price):
                    risk_assessment = "This simulation shows a relatively high standard deviation, indicating significant price volatility and risk. "
                else:
                    risk_assessment = "The simulation suggests relatively lower volatility, indicating moderate risk. "

                recommendation = f"**Interpretation and Recommendation:** {risk_assessment}The average final price provides a central tendency, while the range (min to max) and standard deviation indicate the potential spread and risk. Investors should consider their risk tolerance against these potential outcomes. \n\n**Recommendation:** Use these simulation results as one input among many (e.g., fundamental analysis, market sentiment) for making investment decisions. Consider setting stop-loss orders or taking profits at certain thresholds based on your risk profile."
                return interpretation + recommendation
            except AttributeError: # For re.search().group(1) if not found
                 return f"Could not parse Monte Carlo simulation results for interpretation. Raw output: {numerical_output}"
            except ValueError:
                return f"Could not parse Monte Carlo simulation results for interpretation. Raw output: {numerical_output}"
        else:
            return f"Could not find Monte Carlo simulation results in output for interpretation. Raw output: {numerical_output}"

    else:
        return f"No specific interpretation model for this query. Raw computation result: {numerical_output}"


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🧠 Code-Assisted Financial Analysis Tool 📊")
st.markdown(
    """ 
    This tool uses a simulated AI to generate and execute Python code for financial analysis, 
    then interprets the results to provide insights and recommendations. 
    
    **Disclaimer:** This is a demonstration for the Code-Assisted Reasoning (CAR) pattern. 
    The LLM functions are simulated, and the code execution environment is basic. 
    Do not use this for actual financial decisions without professional advice.
    """
)

st.header("Your Financial Query")
user_query = st.text_area(
    "Enter your financial analysis request (e.g., 'Calculate DCF for cash flows [100, 110, 120] with a discount rate of 0.08' or 'Run Monte Carlo simulation for a stock with initial price 100, daily return 0.001, volatility 0.02, for 252 days, 1000 simulations'):", 
    height=150
)

if st.button("Analyze"):
    if user_query:
        st.subheader("1. AI-Generated Code")
        generated_code = llm_generate_code(user_query)
        st.code(generated_code, language="python")

        st.subheader("2. Code Execution Output")
        execution_output, error_message = execute_code(generated_code)
        
        if execution_output:
            st.text("Output:\n" + execution_output)
        if error_message:
            st.error("Error during execution:\n" + error_message)
        
        st.subheader("3. AI Interpretation & Recommendation")
        if not error_message: # Only interpret if no execution errors
            interpretation = llm_interpret_results(execution_output, user_query)
            st.markdown(interpretation)
        else:
            st.warning("Cannot interpret results due to execution errors.")

    else:
        st.warning("Please enter a financial query to analyze.")
