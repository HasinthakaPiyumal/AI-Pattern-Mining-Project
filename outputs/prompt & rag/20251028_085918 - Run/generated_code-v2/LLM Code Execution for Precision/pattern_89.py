import io
import sys
import re

def execute_python_code(code: str) -> str:
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        exec(code, {'print': print})
        output = redirected_output.getvalue()
    except Exception as e:
        output = f"Error executing code: {e}"
    finally:
        sys.stdout = old_stdout
    return output.strip()

def generate_code_from_query(query: str) -> str:
    query_lower = query.lower()
    if "future value" in query_lower or "compound interest" in query_lower:
        principal_match = re.search(r"principal of (\d+(\.\d+)?)", query_lower)
        rate_match = re.search(r"rate of (\d+(\.\d+)?)%", query_lower)
        years_match = re.search(r"over (\d+) years", query_lower)

        principal = float(principal_match.group(1)) if principal_match else 1000.0
        rate = float(rate_match.group(1)) / 100 if rate_match else 0.05
        years = int(years_match.group(1)) if years_match else 10

        return f"principal = {principal}\nrate = {rate}\nyears = {years}\nfuture_value = principal * (1 + rate)**years\nprint(f'Future Value: {{future_value:.2f}}')"

    elif "simple interest" in query_lower:
        principal_match = re.search(r"principal of (\d+(\.\d+)?)", query_lower)
        rate_match = re.search(r"rate of (\d+(\.\d+)?)%", query_lower)
        years_match = re.search(r"for (\d+) years", query_lower)

        principal = float(principal_match.group(1)) if principal_match else 1000.0
        rate = float(rate_match.group(1)) / 100 if rate_match else 0.05
        years = int(years_match.group(1)) if years_match else 5

        return f"principal = {principal}\nrate = {rate}\nyears = {years}\nsimple_interest = principal * rate * years\nprint(f'Simple Interest: {{simple_interest:.2f}}')"

    elif "calculate" in query_lower or "what is" in query_lower:
        expression_match = re.search(r"calculate (.+)|what is (.+)", query_lower)
        if expression_match:
            expression = expression_match.group(1) or expression_match.group(2)
            # Basic sanitization for simple arithmetic
            if all(c.isdigit() or c in ['+', '-', '*', '/', '(', ')', '.', ' '] for c in expression):
                return f"result = {expression}\nprint(f'Result: {{result}}')"

    return "print('I'm sorry, I can only perform basic financial calculations and arithmetic at the moment.')"

def formulate_advice(query: str, calculation_result: str) -> str:
    if "Error executing code" in calculation_result:
        return f"I encountered an error while processing your request: {calculation_result}. Please try rephrasing or check the input values."
    elif "I'm sorry" in calculation_result:
        return calculation_result
    else:
        return f"Based on your query: \"{query}\", the calculation result is: {calculation_result}. Please note this is a simplified simulation."

def main():
    print("Welcome to the AI Financial Advisory Bot (PAL Prompting Demo)!")
    print("I can help with basic financial calculations like future value, simple interest, and simple arithmetic.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nEnter your financial query: ")
        if user_query.lower() == 'exit':
            break

        generated_code = generate_code_from_query(user_query)
        print(f"\n[Simulated LLM Generated Code]:\n{generated_code}")

        calculation_output = execute_python_code(generated_code)
        print(f"\n[Python Execution Output]:\n{calculation_output}")

        advice = formulate_advice(user_query, calculation_output)
        print(f"\n[Financial Advisory Bot]:\n{advice}")

if __name__ == "__main__":
    main()