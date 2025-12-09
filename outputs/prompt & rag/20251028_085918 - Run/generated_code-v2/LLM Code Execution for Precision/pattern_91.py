import io
import re

# 1. Financial Calculation Modules (Helper Functions)
def future_value(principal, annual_rate, periods):
    return principal * (1 + annual_rate)**periods

def present_value(future_value, annual_rate, periods):
    return future_value / (1 + annual_rate)**periods

def annuity_future_value(payment, annual_rate, periods):
    # This is for calculating the future value of a series of payments
    # Not directly used by the LLM in this current demo but kept for completeness
    if annual_rate == 0:
        return payment * periods
    return payment * (((1 + annual_rate)**periods - 1) / annual_rate)

def calculate_monthly_savings_for_retirement(goal_amount, years, annual_rate):
    if years <= 0:
        return 0
    monthly_rate = annual_rate / 12
    months = years * 12

    if monthly_rate == 0:
        return goal_amount / months
    
    # Formula for present value of an annuity due (if payment is at start of month) or ordinary annuity (end of month)
    # Most common interpretation for 'saving monthly' is at the end of the month (ordinary annuity)
    # We want to find P (monthly payment) such that the future value of these payments equals goal_amount
    # FV = P * [((1 + r)^n - 1) / r]
    # P = FV * [r / ((1 + r)^n - 1)]
    try:
        payment = goal_amount * (monthly_rate / ((1 + monthly_rate)**months - 1))
        return payment
    except ZeroDivisionError:
        return float('inf') # Or handle as an error if annual_rate is too low for target

# 2. Code Execution Environment
class CodeExecutionEnvironment:
    def execute_code(self, code_string, financial_functions):
        old_stdout = io.StringIO()
        try:
            import sys
            sys.stdout = old_stdout
            
            # Create a limited scope for execution
            exec_globals = {"__builtins__": None} # No builtins by default
            exec_locals = {}
            # Allow specific financial functions to be called
            exec_locals.update(financial_functions)

            exec(code_string, exec_globals, exec_locals)
            output = old_stdout.getvalue()
            
            # The LLM expects the result in a variable named 'result'
            if 'result' in exec_locals:
                return str(exec_locals['result'])
            return output.strip() # Fallback for any print statements
        except Exception as e:
            return f"Execution Error: {type(e).__name__}: {e}"
        finally:
            import sys
            sys.stdout = sys.__stdout__ # Restore stdout

# 3. Simulated LLM
class SimulatedLLM:
    def __init__(self, code_executor):
        self.code_executor = code_executor
        self.financial_functions = {
            "future_value": future_value,
            "present_value": present_value,
            "annuity_future_value": annuity_future_value,
            "calculate_monthly_savings_for_retirement": calculate_monthly_savings_for_retirement
        }

    def _parse_query_and_generate_code(self, query):
        query_lower = query.lower()
        code_to_execute = ""
        response_template = ""
        parsed_params = {}

        if "retire" in query_lower and "monthly" in query_lower and "save" in query_lower and "years" in query_lower and "return" in query_lower:
            goal_match = re.search(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", query)
            years_match = re.search(r"(\d+)\s+years", query_lower)
            return_match = re.search(r"(\d+(?:\.\d+)?)\s*(\%|percent)", query_lower) # Added \%|percent for flexibility

            if goal_match:
                parsed_params["goal_amount"] = float(goal_match.group(1).replace(",", ""))
            if years_match:
                parsed_params["years"] = int(years_match.group(1))
            if return_match:
                parsed_params["annual_rate"] = float(return_match.group(1)) / 100

            if all(k in parsed_params for k in ["goal_amount", "years", "annual_rate"]):
                code_to_execute = f"result = calculate_monthly_savings_for_retirement({parsed_params['goal_amount']}, {parsed_params['years']}, {parsed_params['annual_rate']})"
                response_template = "Based on your goals, to retire with ${goal_amount:,.2f} in {years} years with an annual return of {annual_rate:.2%}, you would need to save approximately ${result:,.2f} per month."
            else:
                return "", "I couldn't extract all the necessary parameters for retirement savings calculation. Please provide goal amount, years, and annual return.", {}
        elif "future value" in query_lower or "investment grow to" in query_lower:
            principal_match = re.search(r"invest\s+\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", query_lower)
            rate_match = re.search(r"(\d+(?:\.\d+)?)\s*(\%|percent)\s+annual", query_lower) # Added \%|percent
            periods_match = re.search(r"(\d+)\s+year", query_lower)

            if principal_match:
                parsed_params["principal"] = float(principal_match.group(1).replace(",", ""))
            if rate_match:
                parsed_params["annual_rate"] = float(rate_match.group(1)) / 100
            if periods_match:
                parsed_params["periods"] = int(periods_match.group(1))

            if all(k in parsed_params for k in ["principal", "annual_rate", "periods"]):
                code_to_execute = f"result = future_value({parsed_params['principal']}, {parsed_params['annual_rate']}, {parsed_params['periods']})"
                response_template = "If you invest ${principal:,.2f} at an annual rate of {annual_rate:.2%} for {periods} years, your investment will grow to approximately ${result:,.2f}."
            else:
                return "", "I couldn't extract all the necessary parameters for future value calculation. Please provide principal, annual rate, and periods.", {}
        else:
            return "", "I can't generate code for that financial query yet. Please ask about retirement savings or future value.", {}

        return code_to_execute, response_template, parsed_params

    def process_query(self, user_query):
        # print(f"LLM received query: '{user_query}'") # Commented out for cleaner output in final code
        code_to_execute, response_template, parsed_params = self._parse_query_and_generate_code(user_query)

        if not code_to_execute:
            return response_template # This will contain error message if generation failed

        # print(f"LLM generated code:\n{code_to_execute}") # Commented out for cleaner output in final code

        execution_output = self.code_executor.execute_code(code_to_execute, self.financial_functions)
        # print(f"Code execution output: {execution_output}") # Commented out for cleaner output in final code

        if "Execution Error" in execution_output:
            return f"An error occurred during calculation: {execution_output}. Please try again or rephrase your query."
        
        try:
            calculated_result = float(execution_output)
            parsed_params["result"] = calculated_result
            return response_template.format(**parsed_params)
        except ValueError:
            return f"I received an unexpected output from the calculator: '{execution_output}'. I cannot formulate a precise answer."
        except KeyError as e:
            return f"An internal error occurred while formatting the response: Missing parameter {e}. Please report this."


# Main simulation
def main():
    print("Welcome to FinAdvisor AI: Smart Investment & Retirement Planner!")
    print("I can help you with retirement savings and future value calculations.")
    print("Example queries:")
    print("  - How much do I need to save monthly to retire in 20 years with $1,000,000, assuming 7% annual return?")
    print("  - If I invest $5000 at 5% annual return for 10 years, what will be its future value?")
    print("Type 'exit' to quit.")

    code_executor = CodeExecutionEnvironment()
    finadvisor_llm = SimulatedLLM(code_executor)

    while True:
        user_input = input("\nYour financial query: ")
        if user_input.lower() == 'exit':
            break

        response = finadvisor_llm.process_query(user_input)
        print(f"\nFinAdvisor AI: {response}")

if __name__ == "__main__":
    main()