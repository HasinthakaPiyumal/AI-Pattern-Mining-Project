
import streamlit as st
import pandas as pd
import io
import contextlib

# --- 3. Financial Data API Wrapper (Simulated) ---
class FinancialDataAPI:
    def get_company_financials(self, company_name):
        # Simulated data for demonstration
        if company_name.lower() == "company x":
            return {
                "Q3_Revenue": 150000000,
                "Q3_NetIncome": 25000000,
                "Q2_Revenue": 140000000,
                "Q2_NetIncome": 23000000,
                "Q3_EPS": 1.25,
                "SharesOutstanding": 20000000,
                "LastYear_Revenue": 120000000,
                "TwoYearsAgo_Revenue": 100000000,
                "LastYear_NetIncome": 20000000,
            }
        elif company_name.lower() == "company y":
            return {
                "Q3_Revenue": 100000000,
                "Q3_NetIncome": 15000000,
                "Q3_EPS": 0.75,
                "SharesOutstanding": 20000000,
                "LastYear_Revenue": 90000000,
                "LastYear_NetIncome": 12000000,
            }
        return {}

    def get_industry_average(self, metric):
        # Simulated industry averages
        if metric.lower() == "revenue growth":
            return 0.10 # 10%
        elif metric.lower() == "profit margin":
            return 0.15 # 15%
        elif metric.lower() == "eps":
            return 1.00
        return None

    def calculate_cagr(self, start_value, end_value, years):
        if start_value == 0 or years == 0:
            return 0
        return ((end_value / start_value) ** (1 / years)) - 1

    def calculate_profit_margin(self, revenue, net_income):
        if revenue == 0:
            return 0
        return net_income / revenue
    
    def calculate_eps(self, net_income, shares_outstanding):
        if shares_outstanding == 0:
            return 0
        return net_income / shares_outstanding

# --- 4. Code Execution Environment ---
class CodeExecutor:
    def safe_execute_code(self, code_string, globals_dict, locals_dict):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with contextlib.redirect_stderr(buffer):
                try:
                    exec(code_string, globals_dict, locals_dict)
                    return "SUCCESS", buffer.getvalue()
                except Exception as e:
                    return "ERROR", buffer.getvalue() + f"\nError during execution: {e}"

# --- 2 & 5. Language Model (LLM) Integration & Report Generation (Simulated) ---
class LLMIntegration:
    def __init__(self):
        self.financial_api = FinancialDataAPI()

    def generate_code_from_query(self, query):
        query_lower = query.lower()
        if "q3 performance of company x and compare it to industry average" in query_lower:
            return """
financial_api = FinancialDataAPI()
company_x_data = financial_api.get_company_financials("Company X")

# Calculate Q3 revenue growth
q3_revenue_growth_company_x = (company_x_data['Q3_Revenue'] - company_x_data['Q2_Revenue']) / company_x_data['Q2_Revenue'] if company_x_data['Q2_Revenue'] else 0
industry_revenue_growth = financial_api.get_industry_average("revenue growth")

# Calculate Q3 profit margin
q3_profit_margin_company_x = financial_api.calculate_profit_margin(company_x_data['Q3_Revenue'], company_x_data['Q3_NetIncome'])
industry_profit_margin = financial_api.get_industry_average("profit margin")

# Get Q3 EPS
q3_eps_company_x = company_x_data['Q3_EPS']
industry_eps = financial_api.get_industry_average("eps")

print(f"Company X Q3 Revenue Growth: {q3_revenue_growth_company_x:.2%}")
print(f"Industry Average Revenue Growth: {industry_revenue_growth:.2%}")
print(f"Company X Q3 Profit Margin: {q3_profit_margin_company_x:.2%}")
print(f"Industry Average Profit Margin: {industry_profit_margin:.2%}")
print(f"Company X Q3 EPS: {q3_eps_company_x:.2f}")
print(f"Industry Average EPS: {industry_eps:.2f}")

# Store results in a dictionary for interpretation
results = {
    "q3_revenue_growth_company_x": q3_revenue_growth_company_x,
    "industry_revenue_growth": industry_revenue_growth,
    "q3_profit_margin_company_x": q3_profit_margin_company_x,
    "industry_profit_margin": industry_profit_margin,
    "q3_eps_company_x": q3_eps_company_x,
    "industry_eps": industry_eps,
}
"""
        elif "calculate compound annual growth rate for company x revenue" in query_lower:
            return """
financial_api = FinancialDataAPI()
company_x_data = financial_api.get_company_financials("Company X")

start_revenue = company_x_data.get('TwoYearsAgo_Revenue', 0)
end_revenue = company_x_data.get('Q3_Revenue', 0)
years = 2 # Assuming Q3 is current, LastYear, TwoYearsAgo represents 2 years
cagr = financial_api.calculate_cagr(start_revenue, end_revenue, years)

print(f"Company X Revenue (2 years ago): {start_revenue:,.2f}")
print(f"Company X Current Revenue: {end_revenue:,.2f}")
print(f"Company X 2-Year Revenue CAGR: {cagr:.2%}")

results = {"cagr": cagr}
"""
        else:
            return "print(\"No specific code generated for this query. Try 'Analyze the Q3 performance of Company X and compare it to industry average for revenue growth, profit margin, and EPS'\")\nresults = {}"

    def interpret_results(self, query, execution_output, execution_locals):
        interpretation = f"### Analysis for: '{query}'\n\n"
        
        if "ERROR" in execution_output:
            interpretation += "An error occurred during code execution. Please check the query or the generated code.\n"
            interpretation += f"\nExecution Log:\n```\n{execution_output}\n```"
            return interpretation

        interpretation += "The following calculations were performed:\n\n"
        interpretation += f"```\n{execution_output}\n```\n\n"

        results = execution_locals.get('results', {})

        if "q3_revenue_growth_company_x" in results:
            company_x_rev_growth = results['q3_revenue_growth_company_x']
            industry_rev_growth = results['industry_revenue_growth']
            interpretation += f"**Revenue Growth:** Company X's Q3 revenue growth was {company_x_rev_growth:.2%}. "
            if company_x_rev_growth > industry_rev_growth:
                interpretation += f"This is higher than the industry average of {industry_rev_growth:.2%}, indicating strong top-line performance. "
            else:
                interpretation += f"This is lower than the industry average of {industry_rev_growth:.2%}, suggesting potential challenges in revenue generation. "

            company_x_profit_margin = results['q3_profit_margin_company_x']
            industry_profit_margin = results['industry_profit_margin']
            interpretation += f"**Profit Margin:** Company X's Q3 profit margin was {company_x_profit_margin:.2%}. "
            if company_x_profit_margin > industry_profit_margin:
                interpretation += f"This is above the industry average of {industry_profit_margin:.2%}, reflecting efficient cost management. "
            else:
                interpretation += f"This is below the industry average of {industry_profit_margin:.2%}, suggesting room for improvement in profitability. "

            company_x_eps = results['q3_eps_company_x']
            industry_eps = results['industry_eps']
            interpretation += f"**Earnings Per Share (EPS):** Company X's Q3 EPS was {company_x_eps:.2f}. "
            if company_x_eps > industry_eps:
                interpretation += f"This is higher than the industry average of {industry_eps:.2f}, indicating strong earnings for shareholders. "
            else:
                interpretation += f"This is lower than the industry average of {industry_eps:.2f}, which might concern investors. "

            interpretation += "\nOverall, Company X shows a mixed performance with strong revenue growth but some challenges in profitability compared to the industry. Further investigation into operational costs and market conditions would be beneficial."

        elif "cagr" in results:
            cagr_value = results['cagr']
            interpretation += f"The Compound Annual Growth Rate (CAGR) for Company X's revenue over the last 2 years is **{cagr_value:.2%}**. This indicates the average annual growth rate over the specified period."

        else:
            interpretation += "No specific interpretation generated for this output. The query might be too general or the results were not recognized."
        
        return interpretation

# --- 1. User Interface (Streamlit) ---
st.set_page_config(layout="wide")
st.title("📈 AI-Powered Financial Analysis & Reporting")
st.markdown("Enter your financial analysis query below and let the AI generate a report.")

user_query = st.text_area(
    "Your Financial Query:",
    "Analyze the Q3 performance of Company X and compare it to industry average for revenue growth, profit margin, and EPS"
)

llm_integrator = LLMIntegration()
code_executor = CodeExecutor()

if st.button("Generate Financial Report"):
    if user_query:
        st.subheader("Generating Report...")
        
        # 1. LLM generates code
        generated_code = llm_integrator.generate_code_from_query(user_query)
        st.write("### Generated Code (for review):")
        st.code(generated_code, language="python")

        # 2. Execute the generated code
        # Prepare a safe global environment for execution
        exec_globals = {
            "FinancialDataAPI": FinancialDataAPI, 
            "financial_api": FinancialDataAPI(), # Instantiate for direct use in generated code
            "pd": pd, 
            "np": None # numpy is not explicitly used in current simulated code but good to have a placeholder
        }
        exec_locals = {}
        
        execution_status, execution_output = code_executor.safe_execute_code(
            generated_code, 
            exec_globals, 
            exec_locals
        )

        st.write("### Code Execution Output:")
        if execution_status == "SUCCESS":
            st.success("Code executed successfully!")
            st.code(execution_output, language="text")
        else:
            st.error("Code execution failed!")
            st.code(execution_output, language="text")

        # 3. LLM interprets results and generates report
        interpretation = llm_integrator.interpret_results(user_query, execution_output, exec_locals)
        st.write("### Financial Report:")
        st.markdown(interpretation)
    else:
        st.warning("Please enter a financial query.")
