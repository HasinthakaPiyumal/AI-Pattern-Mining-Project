import re

def generate_financial_code(user_query):
    query_lower = user_query.lower()
    if "future value" in query_lower:
        match = re.search(r"investment of \$(\d+,?\d*\.?\d*).*at (\d+\.?\d*)% annual interest over (\d+) years", query_lower)
        if match:
            principal = float(match.group(1).replace(',', ''))
            rate = float(match.group(2)) / 100
            years = int(match.group(3))
            code = f"principal = {principal}\nrate = {rate}\nyears = {years}\nfuture_value = principal * (1 + rate)**years\nprint(future_value)"
            return code
    elif "simple interest" in query_lower:
        match = re.search(r"principal of \$(\d+,?\d*\.?\d*).*rate of (\d+\.?\d*)% over (\d+) years", query_lower)
        if match:
            principal = float(match.group(1).replace(',', ''))
            rate = float(match.group(2)) / 100
            years = int(match.group(3))
            code = f"principal = {principal}\nrate = {rate}\nyears = {years}\nsimple_interest = principal * rate * years\nprint(simple_interest)"
            return code
    return "print('Unable to generate code for this query.')"

def execute_python_code(code_string):
    local_vars = {}
    try:
        exec(code_string, {}, local_vars)
        if 'future_value' in local_vars: return local_vars['future_value']
        if 'simple_interest' in local_vars: return local_vars['simple_interest']
        return None
    except Exception as e:
        return f"Error during execution: {e}"

def generate_investment_recommendation(user_query, calculation_result):
    if calculation_result is None or isinstance(calculation_result, str):
        return f"I couldn't perform the exact calculation for your query: '{user_query}'. {calculation_result if isinstance(calculation_result, str) else ''} Please rephrase or try a different query."

    if "future value" in user_query.lower():
        return f"Based on your query: '{user_query}', the calculated future value is ${calculation_result:,.2f}. This indicates the potential growth of your investment over the specified period. Consider this value when planning your financial goals."
    elif "simple interest" in user_query.lower():
        return f"Based on your query: '{user_query}', the calculated simple interest is ${calculation_result:,.2f}. This represents the interest earned without compounding over the specified period. Always review your investment's interest structure."
    else:
        return f"For your query: '{user_query}', the calculation resulted in {calculation_result:,.2f}. Further analysis might be needed for a comprehensive recommendation."


def main():
    print("Welcome to the AI Financial Advisor! (Type 'exit' to quit)")
    while True:
        user_query = input("\nHow can I help with your investment today? ")
        if user_query.lower() == 'exit':
            break

        generated_code = generate_financial_code(user_query)
        print(f"\n[DEBUG] Generated Code:\n{generated_code}")

        if generated_code.startswith("print('Unable to generate code"):
            print(generate_investment_recommendation(user_query, None))
            continue

        # Capture print output from exec. This is a simple approach.
        # In a real system, you'd use a more robust sandboxed execution environment
        import io
        import sys
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        calculation_result = None
        try:
            exec(generated_code)
            exec_output = redirected_output.getvalue()
            if exec_output.strip():
                calculation_result = float(exec_output.strip())
        except Exception as e:
            calculation_result = f"Execution Error: {e}"
        finally:
            sys.stdout = old_stdout

        print(f"[DEBUG] Calculation Result: {calculation_result}")
        recommendation = generate_investment_recommendation(user_query, calculation_result)
        print(f"\nAI Advisor: {recommendation}")

if __name__ == "__main__":
    main()