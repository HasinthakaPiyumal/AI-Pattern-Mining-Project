import io
import contextlib

def simulate_llm_code_generation(user_query: str) -> str:
    if "future value" in user_query.lower() and "investment" in user_query.lower():
        # Extract numbers from query for demonstration, in a real scenario LLM would do this
        try:
            parts = user_query.lower().split()
            principal_index = parts.index("of") + 1
            principal = float(parts[principal_index].replace('$', ''))
            rate_index = parts.index("at") + 1
            rate_str = parts[rate_index].replace('%', '')
            rate = float(rate_str) / 100
            time_index = parts.index("for") + 1
            time = int(parts[time_index])
            
            return (
                f"principal = {principal}\n"
                f"rate = {rate}\n"
                f"time = {time}\n"
                "future_value = principal * (1 + rate)**time\n"
                "print(f\"Future Value: {future_value:.2f}\")"
            )
        except (ValueError, IndexError):
            return "print(\"Error: Could not parse input for future value calculation. Please provide principal, rate, and time.\")"
    elif "optimal asset allocation" in user_query.lower():
        return (
            "# Placeholder for complex asset allocation logic\n"
            "# In a real scenario, this would involve risk tolerance, desired returns, etc.\n"
            "import numpy as np\n"
            "portfolio_value = 100000  # Example value\n"
            "asset_weights = {'stocks': 0.6, 'bonds': 0.3, 'cash': 0.1}\n"
            "print(f\"Optimal Asset Allocation (example):\n\")"
            "for asset, weight in asset_weights.items():\n"
            "    print(f\"- {asset.capitalize()}: {weight * 100:.0f}% (${portfolio_value * weight:.2f})\")"
        )
    elif "compare loan options" in user_query.lower():
        return (
            "# Placeholder for loan comparison logic\n"
            "# This would involve calculating monthly payments, total interest, etc.\n"
            "def calculate_monthly_payment(principal, annual_rate, years):\n"
            "    monthly_rate = annual_rate / 12\n"
            "    num_payments = years * 12\n"
            "    if monthly_rate == 0:\n"
            "        return principal / num_payments\n"
            "    payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)\n"
            "    return payment\n\n"
            "loan1_principal = 100000\n"
            "loan1_annual_rate = 0.05\n"
            "loan1_years = 30\n"
            "loan1_payment = calculate_monthly_payment(loan1_principal, loan1_annual_rate, loan1_years)\n"
            "loan1_total_interest = (loan1_payment * loan1_years * 12) - loan1_principal\n\n"
            "loan2_principal = 100000\n"
            "loan2_annual_rate = 0.04\n"
            "loan2_years = 15\n"
            "loan2_payment = calculate_monthly_payment(loan2_principal, loan2_annual_rate, loan2_years)\n"
            "loan2_total_interest = (loan2_payment * loan2_years * 12) - loan2_principal\n\n"
            "print(f\"Loan Option 1 (30 years, 5%):\n\")"
            "print(f\"  Monthly Payment: ${loan1_payment:.2f}\")"
            "print(f\"  Total Interest: ${loan1_total_interest:.2f}\")\n"
            "print(f\"Loan Option 2 (15 years, 4%):\n\")"
            "print(f\"  Monthly Payment: ${loan2_payment:.2f}\")"
            "print(f\"  Total Interest: ${loan2_total_interest:.2f}\")"
        )
    else:
        return "print(\"No specific financial calculation identified. Please ask a question related to future value, asset allocation, or loan comparison.\")"

def execute_python_code_safely(code_string: str) -> str:
    old_stdout = io.StringIO()
    redirect = contextlib.redirect_stdout(old_stdout)
    
    # Basic sandboxing: restrict available builtins and global/local variables
    # In a real-world scenario, a much more robust sandboxing mechanism is needed
    restricted_globals = {
        "__builtins__": {
            "print": print,
            "float": float,
            "int": int,
            "str": str,
            "abs": abs,
            "min": min,
            "max": max,
            "pow": pow,
            "round": round
        }
    }
    restricted_locals = {}
    
    try:
        with redirect:
            exec(code_string, restricted_globals, restricted_locals)
        return old_stdout.getvalue().strip()
    except Exception as e:
        return f"Error during code execution: {e}"

def simulate_llm_result_interpretation(original_query: str, code_output: str) -> str:
    if "future value" in original_query.lower() and "investment" in original_query.lower():
        if "Future Value:" in code_output:
            future_value = code_output.split("Future Value: ")[1]
            return f"Based on your investment query, the calculated future value is {future_value}."
        else:
            return f"Could not calculate future value: {code_output}"
    elif "optimal asset allocation" in original_query.lower():
        if "Optimal Asset Allocation" in code_output:
            return f"Here is an example optimal asset allocation based on our calculations:\n{code_output}"
        else:
            return f"Could not determine optimal asset allocation: {code_output}"
    elif "compare loan options" in original_query.lower():
        if "Loan Option 1" in code_output and "Loan Option 2" in code_output:
            return f"Here is a comparison of the loan options:\n{code_output}"
        else:
            return f"Could not compare loan options: {code_output}"
    else:
        return f"I processed your request, and the computational result was:\n{code_output}\nI can help further with financial planning based on this."

def main():
    print("Welcome to the Smart Financial Advisor!\n")
    
    queries = [
        "Calculate the future value of an investment of $10000 at 7% for 5 years compounded annually.",
        "What is the optimal asset allocation for a balanced portfolio?",
        "Can you compare two loan options for $100,000, one at 5% for 30 years and another at 4% for 15 years?",
        "Tell me a joke."
    ]

    for i, query in enumerate(queries):
        print(f"--- User Query {i+1} ---")
        print(f"User: {query}")

        # Step 1: LLM generates code
        generated_code = simulate_llm_code_generation(query)
        print(f"\nGenerated Code:\n{generated_code}")

        # Step 2: Execute code safely
        code_execution_output = execute_python_code_safely(generated_code)
        print(f"\nCode Execution Output:\n{code_execution_output}")

        # Step 3: LLM interprets results and provides a natural language response
        final_response = simulate_llm_result_interpretation(query, code_execution_output)
        print(f"\nFinancial Advisor: {final_response}\n")
        print("="*50)

if __name__ == "__main__":
    main()