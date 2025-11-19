class FinancialAdvisorLLMWrapper:
    def generate_code(self, query):
        # Simulate parsing the query for income and deductions
        # A more robust solution would use NLP techniques
        income_str = "0"
        deductions_str = "0"

        parts = query.lower().split()
        for i, part in enumerate(parts):
            if part == "income" and i + 1 < len(parts):
                try:
                    income_str = parts[i+1].replace("$", "").replace(",", "")
                except ValueError:
                    pass
            elif part == "deductions" and i + 1 < len(parts):
                try:
                    deductions_str = parts[i+1].replace("$", "").replace(",", "")
                except ValueError:
                    pass
        
        try:
            income = float(income_str)
        except ValueError:
            income = 0.0
            
        try:
            deductions = float(deductions_str)
        except ValueError:
            deductions = 0.0

        generated_code = f"""
def calculate_tax(income, deductions):
    taxable_income = max(0, income - deductions)
    tax = 0
    
    # Simplified progressive tax brackets for demonstration
    if taxable_income <= 10000:
        tax = taxable_income * 0.10
    elif taxable_income <= 40000:
        tax = 10000 * 0.10 + (taxable_income - 10000) * 0.15
    else:
        tax = 10000 * 0.10 + 30000 * 0.15 + (taxable_income - 40000) * 0.25
    return tax

_income = {income}
_deductions = {deductions}
_calculated_tax = calculate_tax(_income, _deductions)
print(f"TAX_RESULT:{{_calculated_tax}}")
"""
        return generated_code

    def formulate_advice(self, query, calculated_result):
        advice = f"Based on your query: \"{query}\", the calculated tax is ${calculated_result:,.2f}.\n"
        
        if calculated_result > 0:
            advice += "Consider exploring additional deductions or tax-advantaged investments to potentially reduce your taxable income and future tax liability. Consulting a professional tax advisor for personalized strategies is recommended."
        else:
            advice += "It appears your current deductions cover your income, resulting in no estimated tax. Always verify with a tax professional."
        return advice


class CodeInterpreter:
    def execute_code(self, code_string):
        local_scope = {}
        try:
            # Redirect stdout to capture print statements
            import io
            import sys
            old_stdout = sys.stdout
            redirected_output = io.StringIO()
            sys.stdout = redirected_output
            
            exec(code_string, {}, local_scope)
            
            sys.stdout = old_stdout # Restore stdout
            output = redirected_output.getvalue()
            
            # Extract the tax result from the captured output
            if "TAX_RESULT:" in output:
                result_line = [line for line in output.split('\n') if "TAX_RESULT:" in line][0]
                tax_value_str = result_line.split(":")[1].strip()
                return float(tax_value_str)
            return None
        except Exception as e:
            return f"Error executing code: {e}"


# Main application flow (simulated user interaction)
def run_financial_advisor():
    llm_wrapper = FinancialAdvisorLLMWrapper()
    interpreter = CodeInterpreter()

    # Simulated User Query 1
    user_query_1 = "Calculate my estimated tax for an income of $70,000 with $5,000 in deductions."
    print(f"\nUser Query: {user_query_1}")
    generated_code_1 = llm_wrapper.generate_code(user_query_1)
    # print(f"\nGenerated Code:\n{generated_code_1}") # For debugging
    
    execution_result_1 = interpreter.execute_code(generated_code_1)
    
    if isinstance(execution_result_1, (float, int)):
        advice_1 = llm_wrapper.formulate_advice(user_query_1, execution_result_1)
        print(f"Financial Advisor: {advice_1}")
    else:
        print(f"Financial Advisor Error: {execution_result_1}")

    # Simulated User Query 2
    user_query_2 = "What's the tax on an income of $25,000 with $10,000 in deductions?"
    print(f"\nUser Query: {user_query_2}")
    generated_code_2 = llm_wrapper.generate_code(user_query_2)
    # print(f"\nGenerated Code:\n{generated_code_2}") # For debugging
    
    execution_result_2 = interpreter.execute_code(generated_code_2)
    
    if isinstance(execution_result_2, (float, int)):
        advice_2 = llm_wrapper.formulate_advice(user_query_2, execution_result_2)
        print(f"Financial Advisor: {advice_2}")
    else:
        print(f"Financial Advisor Error: {execution_result_2}")

    # Simulated User Query 3 - More complex scenario or error case
    user_query_3 = "Estimate tax for 150000 income and 20000 deductions."
    print(f"\nUser Query: {user_query_3}")
    generated_code_3 = llm_wrapper.generate_code(user_query_3)
    # print(f"\nGenerated Code:\n{generated_code_3}") # For debugging
    
    execution_result_3 = interpreter.execute_code(generated_code_3)
    
    if isinstance(execution_result_3, (float, int)):
        advice_3 = llm_wrapper.formulate_advice(user_query_3, execution_result_3)
        print(f"Financial Advisor: {advice_3}")
    else:
        print(f"Financial Advisor Error: {execution_result_3}")

if __name__ == "__main__":
    run_financial_advisor()