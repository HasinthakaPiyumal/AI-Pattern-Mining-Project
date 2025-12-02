import streamlit as st
import pandas as pd
import numpy as np
import io
import contextlib
import re

def mock_financial_data_api(company_ticker: str, metric: str, years: int = 5):
    if company_ticker.upper() == "AAPL":
        if metric == "revenue":
            base_revenue = 200 * 10**9
            revenue_data = {
                2023: base_revenue * 1.07,
                2022: base_revenue * 1.05,
                2021: base_revenue * 1.10,
                2020: base_revenue * 1.02,
                2019: base_revenue * 1.00,
                2018: base_revenue * 0.98,
            }
            sorted_years = sorted(revenue_data.keys(), reverse=True)
            relevant_years = sorted_years[:years]
            return pd.Series([revenue_data[year] for year in relevant_years], index=relevant_years[::-1])
        elif metric == "cash_flows":
            return pd.Series([90e9, 95e9, 100e9, 105e9, 110e9], index=[2023, 2024, 2025, 2026, 2027])
    elif company_ticker.upper() == "GOOG":
        if metric == "revenue":
            base_revenue = 150 * 10**9
            revenue_data = {
                2023: base_revenue * 1.08,
                2022: base_revenue * 1.06,
                2021: base_revenue * 1.12,
                2020: base_revenue * 1.03,
                2019: base_revenue * 1.01,
                2018: base_revenue * 0.99,
            }
            sorted_years = sorted(revenue_data.keys(), reverse=True)
            relevant_years = sorted_years[:years]
            return pd.Series([revenue_data[year] for year in relevant_years], index=relevant_years[::-1])
        elif metric == "cash_flows":
            return pd.Series([70e9, 75e9, 80e9, 85e9, 90e9], index=[2023, 2024, 2025, 2026, 2027])
    return pd.Series([], dtype='float64')

def mock_llm_generate_code_and_explanation(query: str):
    generated_code = ""
    explanation_template = ""
    result_variable_name = "result_output"

    query_lower = query.lower()

    if "cagr" in query_lower and ("revenue" in query_lower or "growth rate" in query_lower):
        match = re.search(r"company (\w+)'s revenue", query_lower)
        company = match.group(1).upper() if match else "AAPL"
        match_years = re.search(r"(\d+)-year", query_lower)
        years = int(match_years.group(1)) if match_years else 5

        generated_code = f"""
initial_revenue = float(fetch_data("{company}", "revenue", years={years}).iloc[0])
final_revenue = float(fetch_data("{company}", "revenue", years={years}).iloc[-1])
num_periods = {years - 1}
cagr = ((final_revenue / initial_revenue)**(1/num_periods)) - 1
print(f"{{'CAGR': cagr}}")
"""
        explanation_template = f"The {years}-year Compound Annual Growth Rate (CAGR) for {company}'s revenue is {{CAGR_value:.2%}}. This indicates the average annual growth rate over the specified period, assuming the profits were reinvested."
    elif "discounted cash flow" in query_lower or "dcf" in query_lower:
        match = re.search(r"company (\w+)", query_lower)
        company = match.group(1).upper() if match else "AAPL"
        
        generated_code = f"""
cash_flows = fetch_data("{company}", "cash_flows")
discount_rate = 0.10
terminal_growth_rate = 0.03

pv_cash_flows = 0
for i, cf in enumerate(cash_flows):
    pv_cash_flows += cf / ((1 + discount_rate)**(i + 1))

last_cf = cash_flows.iloc[-1]
terminal_value = (last_cf * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
pv_terminal_value = terminal_value / ((1 + discount_rate)**len(cash_flows))

dcf_value = pv_cash_flows + pv_terminal_value
print(f"{{'DCF_Value': dcf_value}}")
"""
        explanation_template = f"Based on a discounted cash flow (DCF) analysis for {company}, with assumed discount rate of 10% and a terminal growth rate of 3%, the estimated intrinsic value is approximately ${{DCF_Value:,.2f}}. This valuation method discounts future cash flows back to their present value to determine an investment's worth."
    else:
        generated_code = "print({'message': 'No specific financial calculation identified. Please refine your query.'})"
        explanation_template = "I couldn't identify a specific financial calculation for your query. Please ask for a CAGR, DCF, or other specific financial metric. {{message}}"

    return generated_code, explanation_template, result_variable_name

def execute_code_safely(code: str, custom_globals: dict):
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    captured_result = {}

    with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
        try:
            local_scope = {"__builtins__": None}
            exec(code, custom_globals, local_scope)

            raw_output = output_buffer.getvalue().strip()
            if raw_output:
                try:
                    captured_result = eval(raw_output)
                except Exception:
                    captured_result = {"raw_output": raw_output}

        except Exception as e:
            error_message = error_buffer.getvalue() + str(e)
            captured_result = {"error": error_message}
    
    return captured_result, error_buffer.getvalue()

st.set_page_config(layout="wide", page_title="Financial Analysis AI")

st.title("Automated Financial Analysis and Reporting System")
st.markdown("Ask me to calculate financial metrics like CAGR or perform a DCF analysis.")

user_query = st.text_area("Enter your financial query:", "Calculate the 5-year CAGR for Apple's revenue.")

if st.button("Analyze"):
    if user_query:
        st.subheader("Processing Request...")
        
        generated_code, explanation_template, result_var_name = mock_llm_generate_code_and_explanation(user_query)

        st.markdown("**Generated Python Code:**")
        st.code(generated_code, language="python")

        execution_globals = {
            "fetch_data": mock_financial_data_api,
            "pd": pd,
            "np": np,
            "__builtins__": {
                "print": print,
                "float": float,
                "len": len,
                "range": range,
                "sum": sum,
                "dict": dict,
                "str": str,
            }
        }
        
        execution_results, exec_errors = execute_code_safely(generated_code, execution_globals)

        if "error" in execution_results:
            st.error(f"**Code Execution Error:**\n```\n{execution_results['error']}\n```")
        else:
            st.subheader("Code Execution Output:")
            st.json(execution_results)

            final_report = explanation_template
            
            if 'CAGR' in execution_results:
                final_report = final_report.format(CAGR_value=execution_results['CAGR'])
            elif 'DCF_Value' in execution_results:
                final_report = final_report.format(DCF_Value=execution_results['DCF_Value'])
            elif 'message' in execution_results:
                final_report = final_report.format(message=execution_results['message'])
            
            st.subheader("Financial Analysis Report:")
            st.write(final_report)
    else:
        st.warning("Please enter a query to analyze.")

st.markdown("""
---
**Important Security Warning:**
This demonstration uses Python's `exec()` function for code execution. In a real-world application,
executing arbitrary code from an LLM in a direct and un-sandboxed manner is a severe security risk.
A robust, secure sandboxing solution (e.g., a dedicated microservice, Docker container, or
RestrictedPython environment) is **critical** for production systems.
""")