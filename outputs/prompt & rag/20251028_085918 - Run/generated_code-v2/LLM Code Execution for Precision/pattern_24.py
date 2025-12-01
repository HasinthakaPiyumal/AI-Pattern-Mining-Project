import sys
import io

class FinancialAdvisor:
    def __init__(self):
        # In a real application, this would be an actual LLM API call
        pass

    def _simulate_llm_code_generation(self, user_query: str) -> str:
        """
        Simulates an LLM generating Python code based on a financial query.
        For demonstration, it generates code for a specific retirement investment scenario.
        """
        if "retirement" in user_query.lower() and "investment" in user_query.lower():
            # Example: "What's the optimal investment strategy for my retirement given my current savings of $50000, expected income growth of 3%, and a risk tolerance for a 7% annual return over 20 years?"
            # Extract numbers from the query (simplified parsing for demo)
            # A real LLM would extract these more robustly
            initial_investment = 50000
            annual_return_rate = 0.07
            years = 20
            monthly_contribution = 500 # Assume a monthly contribution for a more complex scenario

            code = f"""
def calculate_future_value(principal, annual_rate, years, monthly_contribution=0):
    # Simplified calculation for demonstration (assuming annual compounding for simplicity first, then monthly for contributions)
    # Future value of a lump sum
    fv_lump_sum = principal * (1 + annual_rate)**years

    # Future value of a series of monthly payments (annuity)
    # For monthly contributions, convert annual rate to monthly rate and years to months
    monthly_rate = annual_rate / 12
    num_months = years * 12
    fv_contributions = 0
    if monthly_contribution > 0 and monthly_rate > 0:
        # Using future value of an ordinary annuity formula
        # FV = P * [((1 + r)^n - 1) / r]
        # Where r is the monthly rate, n is the number of months, P is monthly contribution
        fv_contributions = monthly_contribution * (((1 + monthly_rate)**num_months - 1) / monthly_rate)
    elif monthly_contribution > 0 and monthly_rate == 0:
         fv_contributions = monthly_contribution * num_months # Simple sum if rate is 0

    return fv_lump_sum + fv_contributions

principal_amount = {initial_investment}
annual_interest_rate = {annual_return_rate}
investment_years = {years}
monthly_contribution_amount = {monthly_contribution}

result = calculate_future_value(
    principal_amount,
    annual_interest_rate,
    investment_years,
    monthly_contribution_amount
)
print(f"The calculated future value is: {{result:.2f}}")
"""
            return code
        return "print('Could not generate specific financial calculation code for this query.')"


    def _execute_generated_code(self, code: str) -> str:
        """
        Executes the given Python code in a sandboxed environment and captures its output.
        WARNING: Using exec() with untrusted input is a security risk.
                 In a real application, a secure sandboxing solution is essential.
        """
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        execution_globals = {}
        execution_locals = {}
        try:
            exec(code, execution_globals, execution_locals)
        except Exception as e:
            return f"Error during code execution: {e}"
        finally:
            sys.stdout = old_stdout # Restore stdout
        return redirected_output.getvalue().strip()

    def _simulate_llm_advice_generation(self, calculation_result: str, original_query: str) -> str:
        """
        Simulates an LLM generating natural language financial advice
        based on the calculation result and original query.
        """
        if "The calculated future value is:" in calculation_result:
            try:
                value_str = calculation_result.split(":")[1].strip().replace(",", "")
                future_value = float(value_str)
                advice = f"Based on your query regarding retirement investment and the calculation, your estimated future investment value is approximately ${future_value:,.2f} after considering your initial savings, monthly contributions, and desired annual return.\n\n"
                advice += "This calculation provides a projection, but actual returns may vary based on market conditions, inflation, and tax changes. It's advisable to regularly review your investment strategy and consider consulting a human financial expert for personalized planning."
                return advice
            except ValueError:
                return f"Could not parse calculation result to generate detailed advice. Raw result: {calculation_result}"
        return f"I have processed your request. Here's the raw calculation output:\n{calculation_result}\n\n" \
               "Please note that I'm an AI and this advice is for informational purposes only. Consult a financial professional for tailored advice."

    def provide_financial_advice(self, user_query: str) -> str:
        print(f"User Query: {user_query}\n")

        # Step 1 & 2: LLM generates code
        print("Simulating LLM generating Python code...")
        generated_code = self._simulate_llm_code_generation(user_query)
        print("--- Generated Code ---")
        print(generated_code)
        print("----------------------\n")

        # Step 3: Execute the code
        print("Executing generated code...")
        calculation_output = self._execute_generated_code(generated_code)
        print("--- Code Execution Output ---")
        print(calculation_output)
        print("---------------------------\n")

        # Step 4 & 5: LLM incorporates results and formulates advice
        print("Simulating LLM formulating final advice...")
        final_advice = self._simulate_llm_advice_generation(calculation_output, user_query)
        print("--- Final Financial Advice ---")
        print(final_advice)
        print("----------------------------\n")

        return final_advice

# Example Usage:
if __name__ == "__main__":
    advisor = FinancialAdvisor()
    query = "What's the optimal investment strategy for my retirement given my current savings of $50000, expected income growth of 3%, and a risk tolerance for a 7% annual return over 20 years, with monthly contributions of $500?"
    advisor.provide_financial_advice(query)
