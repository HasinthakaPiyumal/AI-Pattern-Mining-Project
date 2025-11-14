
import io
import contextlib
import re

class PersonalFinanceAdvisor:
    def __init__(self):
        pass

    def _simulate_llm_code_generation(self, user_query: str) -> str:
        """
        Simulates an LLM generating Python code based on a financial query.
        In a real application, this would involve a sophisticated LLM call.
        """
        if "compound interest" in user_query.lower():
            match = re.search(r"principal of (\d+), annual rate of (\d+\.?\d*)%, over (\d+) years", user_query.lower())
            if match:
                principal = float(match.group(1))
                rate = float(match.group(2)) / 100
                years = int(match.group(3))
                # For simplicity, assume compounding annually
                return f"""
principal = {principal}
rate = {rate}
years = {years}
future_value = principal * (1 + rate)**years
print(f"The future value with compound interest is: {{future_value:.2f}}")
"""
            else:
                return "print('Please provide principal, annual rate, and years for compound interest calculation.')"
        elif "loan amortization" in user_query.lower() or "monthly payment" in user_query.lower():
            match = re.search(r"loan amount of (\d+), annual interest rate of (\d+\.?\d*)%, over (\d+) months", user_query.lower())
            if match:
                loan_amount = float(match.group(1))
                annual_rate = float(match.group(2)) / 100
                months = int(match.group(3))
                monthly_rate = annual_rate / 12
                if monthly_rate == 0:
                    monthly_payment = loan_amount / months
                else:
                    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
                return f"""
loan_amount = {loan_amount}
annual_rate = {annual_rate}
months = {months}
monthly_rate = annual_rate / 12
monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1) if monthly_rate != 0 else loan_amount / months
print(f"Your estimated monthly loan payment is: {{monthly_payment:.2f}}")
"""
            else:
                return "print('Please provide loan amount, annual interest rate, and number of months for loan amortization.')"
        elif "retirement savings" in user_query.lower():
             match = re.search(r"monthly savings of (\d+), annual return of (\d+\.?\d*)%, over (\d+) years", user_query.lower())
             if match:
                 monthly_savings = float(match.group(1))
                 annual_return = float(match.group(2)) / 100
                 years = int(match.group(3))
                 months = years * 12
                 monthly_return_rate = annual_return / 12
                 if monthly_return_rate == 0:
                     future_value_annuity = monthly_savings * months
                 else:
                     future_value_annuity = monthly_savings * (((1 + monthly_return_rate)**months - 1) / monthly_return_rate)

                 return f"""
monthly_savings = {monthly_savings}
annual_return = {annual_return}
years = {years}
months = years * 12
monthly_return_rate = annual_return / 12
future_value_annuity = monthly_savings * (((1 + monthly_return_rate)**months - 1) / monthly_return_rate) if monthly_return_rate != 0 else monthly_savings * months
print(f"Your estimated retirement savings will be: {{future_value_annuity:.2f}}")
"""
             else:
                 return "print('Please provide monthly savings, annual return rate, and years for retirement planning.')"
        else:
            return "print('I can help with compound interest, loan amortization, or retirement savings. Please specify your query.')"

    def _execute_python_code(self, code: str) -> str:
        """
        Executes the given Python code in an isolated environment and captures its output.
        """
        old_stdout = io.StringIO()
        with contextlib.redirect_stdout(old_stdout):
            try:
                exec(code, {'__builtins__': {}})
            except Exception as e:
                return f"Error during code execution: {e}"
        return old_stdout.getvalue().strip()

    def _simulate_llm_explanation_generation(self, original_query: str, calculation_result: str) -> str:
        """
        Simulates an LLM generating a natural language explanation from the results.
        In a real application, this would involve a sophisticated LLM call.
        """
        if "Error during code execution" in calculation_result:
            return f"I encountered an issue processing your request: {calculation_result}. Please try again with a valid query."
        elif calculation_result:
            return f"Based on your query: '{original_query}', I performed the calculation. Here is the result: {calculation_result}"
        else:
            return "I could not generate a specific financial explanation for your query. Please ensure your input is clear."

    def advise(self, user_query: str) -> str:
        """
        Orchestrates the PAL prompting pattern for financial advice.
        """
        print(f"\nUser Query: {user_query}")

        # 1. Simulate LLM Code Generation
        generated_code = self._simulate_llm_code_generation(user_query)
        print(f"\n--- Simulated LLM Generated Code ---\n{generated_code}\n------------------------------------")

        # 2. Execute the Generated Code
        execution_output = self._execute_python_code(generated_code)
        print(f"\n--- Code Execution Result ---\n{execution_output}\n-----------------------------")

        # 3. Simulate LLM Explanation Generation
        final_explanation = self._simulate_llm_explanation_generation(user_query, execution_output)
        print(f"\n--- Final Financial Advice ---\n{final_explanation}\n------------------------------")

        return final_explanation

if __name__ == "__main__":
    advisor = PersonalFinanceAdvisor()

    print("Welcome to the PAL-powered Personal Finance Advisor!\n")
    print("I can help with: ")
    print("- Compound interest (e.g., 'Calculate compound interest for a principal of 10000, annual rate of 5%, over 10 years.')")
    print("- Loan amortization (e.g., 'What is the monthly payment for a loan amount of 100000, annual interest rate of 4.5%, over 360 months?')")
    print("- Retirement savings (e.g., 'Estimate retirement savings with monthly savings of 500, annual return of 7%, over 30 years.')")
    print("- Type 'exit' to quit.")

    while True:
        user_input = input("\nYour financial query: ")
        if user_input.lower() == 'exit':
            print("Exiting advisor. Goodbye!")
            break
        if not user_input.strip():
            print("Please enter a query.")
            continue

        advisor.advise(user_input)

