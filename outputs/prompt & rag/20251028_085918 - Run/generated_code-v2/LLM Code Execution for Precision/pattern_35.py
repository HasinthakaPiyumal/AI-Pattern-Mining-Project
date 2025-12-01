import io
import contextlib
import math

def simulate_llm_code_generation(query):
    query_lower = query.lower()

    if "investment return" in query_lower and "monthly contributions" in query_lower:
        try:
            parts = query_lower.split(" ")
            years_idx = parts.index("years") - 1
            years = int(parts[years_idx])

            contributions_idx = parts.index("contributions") + 2
            monthly_contribution = float(parts[contributions_idx].replace("$", ""))

            rate_idx = parts.index("rate") + 2
            annual_interest_rate_str = parts[rate_idx].replace("%", "")
            annual_interest_rate = float(annual_interest_rate_str) / 100

            code = f"""
import math
monthly_contribution = {monthly_contribution}
annual_interest_rate = {annual_interest_rate}
years = {years}

monthly_interest_rate = annual_interest_rate / 12
num_payments = years * 12

future_value_contributions = monthly_contribution * (((1 + monthly_interest_rate)**num_payments - 1) / monthly_interest_rate)

print(f"{{future_value_contributions:.2f}}")
"""
            response_template = f"Based on your inputs, your investment with monthly contributions of ${monthly_contribution:.2f} and an annual interest rate of {annual_interest_rate:.1%} for {years} years is projected to be approximately ${{result}}. This calculation assumes consistent contributions and interest accrual."
            return code, response_template
        except (ValueError, IndexError):
            return None, "I couldn't parse the details for the investment return calculation. Please ensure the format is clear (e.g., 'investment return after 10 years with monthly contributions of $500 and an annual interest rate of 7%')."

    elif "monthly payment" in query_lower and "loan" in query_lower:
        try:
            parts = query_lower.split(" ")
            principal_idx = parts.index("a") + 2
            principal = float(parts[principal_idx].replace("$", "").replace(",", ""))

            years_idx = parts.index("years") - 1
            years = int(parts[years_idx])

            rate_idx = parts.index("at") + 1
            annual_interest_rate_str = parts[rate_idx].replace("%", "")
            annual_interest_rate = float(annual_interest_rate_str) / 100

            code = f"""
import math
principal = {principal}
annual_interest_rate = {annual_interest_rate}
years = {years}

monthly_interest_rate = annual_interest_rate / 12
num_payments = years * 12

if monthly_interest_rate > 0:
    monthly_payment = principal * (monthly_interest_rate * (1 + monthly_interest_rate)**num_payments) / ((1 + monthly_interest_rate)**num_payments - 1)
else:
    monthly_payment = principal / num_payments

print(f"{{monthly_payment:.2f}}")
"""
            response_template = f"For a loan of ${principal:,.2f} over {years} years at an annual interest rate of {annual_interest_rate:.1%}, your estimated monthly payment would be ${{result}}. This does not include taxes or insurance."
            return code, response_template
        except (ValueError, IndexError):
            return None, "I couldn't parse the details for the loan payment calculation. Please ensure the format is clear (e.g., 'monthly payment for a $300,000 loan over 30 years at 4.5% interest')."

    return None, "I'm sorry, I can only assist with investment return and loan payment calculations at the moment. Please ask a specific financial question."

def execute_code(code):
    old_stdout = io.StringIO()
    with contextlib.redirect_stdout(old_stdout):
        try:
            exec(code, {'math': math})
            result = old_stdout.getvalue().strip()
            return result
        except Exception as e:
            return f"Error executing code: {e}"

def main():
    print("Welcome to the Smart Financial Advisor!")
    print("I can help with investment return and loan payment calculations.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nHow can I help you with your finances? ")
        if user_query.lower() == 'exit':
            break

        generated_code, response_template = simulate_llm_code_generation(user_query)

        if generated_code:
            print("\n(Simulated LLM generated code for execution):")
            print("```python")
            print(generated_code.strip())
            print("```")
            
            execution_result = execute_code(generated_code)
            
            if "Error executing code" in execution_result:
                print(f"\nError: {execution_result}")
            else:
                final_response = response_template.format(result=execution_result)
                print(f"\nSmart Financial Advisor: {final_response}")
        else:
            print(f"\nSmart Financial Advisor: {response_template}")

if __name__ == "__main__":
    main()