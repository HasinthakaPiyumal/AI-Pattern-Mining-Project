import io
import contextlib

def simulate_llm_generate_code(user_query: str, financial_data_input: dict) -> str:
    """
    Simulates an LLM generating Python code based on a user query and financial data.
    In a real application, this would be an actual LLM call.
    """
    code_snippets = {
        "current ratio": f"""
current_assets = {financial_data_input.get("current_assets", 0)}
current_liabilities = {financial_data_input.get("current_liabilities", 0)}
if current_liabilities != 0:
    current_ratio = current_assets / current_liabilities
    print(f"Current Ratio: {{current_ratio:.2f}}")
else:
    print("Cannot calculate Current Ratio: Current Liabilities are zero or not provided.")
""",
        "debt to equity ratio": f"""
total_debt = {financial_data_input.get("total_debt", 0)}
shareholders_equity = {financial_data_input.get("shareholders_equity", 0)}
if shareholders_equity != 0:
    debt_to_equity_ratio = total_debt / shareholders_equity
    print(f"Debt-to-Equity Ratio: {{debt_to_equity_ratio:.2f}}")
else:
    print("Cannot calculate Debt-to-Equity Ratio: Shareholders' Equity is zero or not provided.")
""",
        "revenue growth": f"""
current_year_revenue = {financial_data_input.get("current_year_revenue", 0)}
previous_year_revenue = {financial_data_input.get("previous_year_revenue", 0)}
if previous_year_revenue != 0:
    revenue_growth = ((current_year_revenue - previous_year_revenue) / previous_year_revenue) * 100
    print(f"Revenue Growth: {{revenue_growth:.2f}}%")
else:
    print("Cannot calculate Revenue Growth: Previous Year Revenue is zero or not provided.")
"""
    }

    user_query_lower = user_query.lower()
    for keyword, code in code_snippets.items():
        if keyword in user_query_lower:
            return code
    
    return f"""
print("No specific calculation found for your query: '{user_query}'")
print("Available calculations: Current Ratio, Debt to Equity Ratio, Revenue Growth.")
"""

def execute_python_code(code_string: str) -> str:
    """
    Executes the given Python code string and captures its standard output.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            # We use a limited global scope to prevent arbitrary system access
            # For a real system, more robust sandboxing would be necessary.
            exec(code_string, {'__builtins__': {}})
        except Exception as e:
            print(f"Error during code execution: {e}")
    return buffer.getvalue().strip()

def simulate_llm_generate_report(analysis_results: str, initial_query: str) -> str:
    """
    Simulates an LLM generating a natural language report based on analysis results.
    In a real application, this would be an actual LLM call.
    """
    if "Error" in analysis_results or "Cannot calculate" in analysis_results:
        return f"Could not perform analysis for '{initial_query}' due to: {analysis_results}"

    report_intro = f"Financial Analysis Report for query: '{initial_query}'\n\n"
    report_body = f"Calculations performed and results:\n{analysis_results}\n\n"
    report_conclusion = "These results provide key insights into the financial health and performance based on the specific metrics requested. Further analysis would involve comparing these figures to industry benchmarks and historical trends."

    if "Current Ratio" in analysis_results:
        try:
            ratio_value = float(analysis_results.split("Current Ratio: ")[1].split(" ")[0])
            if ratio_value >= 2.0:
                report_conclusion = "A Current Ratio of {ratio_value:.2f} generally indicates good short-term liquidity, suggesting the company can comfortably cover its short-term obligations.".format(ratio_value=ratio_value)
            elif ratio_value >= 1.0:
                report_conclusion = "A Current Ratio of {ratio_value:.2f} indicates adequate short-term liquidity, but it's important to compare to industry averages.".format(ratio_value=ratio_value)
            else:
                report_conclusion = "A Current Ratio of {ratio_value:.2f} suggests potential short-term liquidity issues, as current liabilities might exceed current assets.".format(ratio_value=ratio_value)
        except (ValueError, IndexError):
            pass # Fallback to generic conclusion

    return report_intro + report_body + report_conclusion

def main():
    print("Welcome to the Financial Statement Analysis Tool (PAL Prompting Demo)")
    print("--------------------------------------------------------------------")

    # --- Step 1: Simulate User Input of Financial Data ---
    # In a real app, this would come from a database, file upload, or user form.
    # For this demo, we'll use a hardcoded dictionary.
    financial_data = {
        "current_assets": 150000,
        "current_liabilities": 75000,
        "total_debt": 200000,
        "shareholders_equity": 300000,
        "current_year_revenue": 1000000,
        "previous_year_revenue": 800000
    }
    print("\n--- Simulated Financial Data ---")
    for key, value in financial_data.items():
        print(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
    print("--------------------------------")

    user_query = input("\nEnter your financial analysis query (e.g., 'calculate current ratio', 'debt to equity ratio', 'revenue growth'): ")

    # --- Step 2: LLM Generates Code (Simulated) ---
    print(f"\n--- LLM (Simulated) Generating Code for query: '{user_query}' ---")
    generated_code = simulate_llm_generate_code(user_query, financial_data)
    print("\nGenerated Python Code:")
    print("```python")
    print(generated_code)
    print("```")

    # --- Step 3: Execute Generated Code ---
    print("\n--- Executing Generated Code ---")
    analysis_results = execute_python_code(generated_code)
    print(f"\nExecution Output:\n{analysis_results}")

    # --- Step 4: LLM Generates Report based on results (Simulated) ---
    print("\n--- LLM (Simulated) Generating Financial Analysis Report ---")
    final_report = simulate_llm_generate_report(analysis_results, user_query)
    print("\nFinancial Analysis Report:")
    print(final_report)
    print("\n--------------------------------------------------------------------")
    print("Demo End.")

if __name__ == "__main__":
    main()