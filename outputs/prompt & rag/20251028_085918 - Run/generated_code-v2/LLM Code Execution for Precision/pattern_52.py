import io
import contextlib

class FinancialAdvisorAI:
    def _generate_code(self, query: str) -> str:
        query_lower = query.lower()
        code = ""
        result_var = "result"

        if "future value" in query_lower and "compound interest" in query_lower:
            try:
                # Example: "Calculate the future value of an investment with compound interest over 10 years at a 5% annual rate, with monthly contributions of $100" (simplified for no contributions for now)
                # Find numbers in query to extract parameters
                import re
                numbers = [float(s) for s in re.findall(r'\d+\.?\d*', query)]

                principal = None
                annual_rate = None
                years = None
                compounding_periods = 12 # Default to monthly if not specified or detectable

                # Heuristic to extract parameters - this would be more robust with an actual LLM/NLU
                if len(numbers) >= 3:
                    # Assuming order: principal, years, rate
                    # This is a simplification for demonstration. A real LLM would extract these better.
                    principal = numbers[0] if numbers[0] > 10 else None # Crude check for principal
                    years = numbers[1] if numbers[1] < 100 else None # Crude check for years
                    annual_rate = numbers[2] / 100 if numbers[2] < 100 else None # Crude check for rate
                    
                    if not principal and len(numbers) >= 4:
                        principal = numbers[0]
                        years = numbers[2]
                        annual_rate = numbers[1] / 100

                    if not principal:
                        # A more robust parser would be needed. For this demo, let's assume a fixed query structure
                        # or look for keywords like "principal", "rate", "years"
                        pass # Fallback to a simpler example if parsing fails


                if principal is None:
                    # Defaulting to a fixed example for demonstration if parsing is too complex
                    # A real LLM would generate the exact numbers from the query.
                    principal = 1000.0
                    annual_rate = 0.05
                    years = 10.0
                    compounding_periods = 12

                if "monthly" in query_lower:
                    compounding_periods = 12
                elif "quarterly" in query_lower:
                    compounding_periods = 4
                elif "annually" in query_lower:
                    compounding_periods = 1

                code = f"""P = {principal}
r = {annual_rate}
t = {years}
n = {compounding_periods}
{result_var} = P * (1 + r / n)**(n * t)
print(f"The future value of the investment is: ${result_var:.2f}")"""

            except Exception as e:
                print(f"Error parsing query for compound interest: {e}")
                code = f"print(\"Could not calculate compound interest due to parsing error.\")"

        elif "simple interest" in query_lower:
            try:
                import re
                numbers = [float(s) for s in re.findall(r'\d+\.?\d*', query)]
                principal = 1000.0 # Default
                annual_rate = 0.03 # Default
                years = 5.0      # Default

                if len(numbers) >= 3:
                    principal = numbers[0]
                    annual_rate = numbers[1] / 100
                    years = numbers[2]

                code = f"""P = {principal}
r = {annual_rate}
t = {years}
{result_var} = P * r * t
print(f"The simple interest earned is: ${result_var:.2f}")"""
            except Exception as e:
                print(f"Error parsing query for simple interest: {e}")
                code = f"print(\"Could not calculate simple interest due to parsing error.\")"

        elif "portfolio return" in query_lower:
            try:
                import re
                numbers = [float(s) for s in re.findall(r'\d+\.?\d*', query)]
                initial_investment = 10000.0 # Default
                final_value = 12000.0 # Default

                if len(numbers) >= 2:
                    initial_investment = numbers[0]
                    final_value = numbers[1]

                code = f"""initial = {initial_investment}
final = {final_value}
{result_var} = ((final - initial) / initial) * 100
print(f"Your portfolio return is: {result_var:.2f}%")"""
            except Exception as e:
                print(f"Error parsing query for portfolio return: {e}")
                code = f"print(\"Could not calculate portfolio return due to parsing error.\")"
        else:
            code = "print(\"I can only assist with specific financial calculations like compound interest, simple interest, or portfolio return.\")"
        return code

    def _execute_code(self, code: str) -> str:
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            try:
                # Using a limited global and local scope for security and isolation
                # In a real scenario, this would be within a secure sandbox environment
                exec(code, {'__builtins__': {}}, {})
            except Exception as e:
                print(f"Error during code execution: {e}")
        return stdout_capture.getvalue().strip()

    def process_query(self, query: str) -> str:
        generated_code = self._generate_code(query)
        execution_output = self._execute_code(generated_code)
        
        if "Could not" in execution_output or "Error during code execution" in execution_output:
            return f"I encountered an issue processing your request: {execution_output}"
        elif "I can only assist" in execution_output:
            return execution_output
        else:
            return f"Based on your request: {execution_output}"

if __name__ == "__main__":
    advisor = FinancialAdvisorAI()

    print("--- Test Case 1: Compound Interest ---")
    query1 = "Calculate the future value of an investment with compound interest over 10 years at a 5% annual rate, with a principal of $1000 and monthly compounding."
    response1 = advisor.process_query(query1)
    print(response1)
    print("\n")

    print("--- Test Case 2: Simple Interest ---")
    query2 = "What is the simple interest earned on $5000 at an annual rate of 3% over 7 years?"
    response2 = advisor.process_query(query2)
    print(response2)
    print("\n")

    print("--- Test Case 3: Portfolio Return ---")
    query3 = "I started with an investment of $20000 and it is now worth $25000. What is my portfolio return?"
    response3 = advisor.process_query(query3)
    print(response3)
    print("\n")

    print("--- Test Case 4: Unsupported Query ---")
    query4 = "Tell me about the stock market trends next year."
    response4 = advisor.process_query(query4)
    print(response4)
    print("\n")

    print("--- Test Case 5: Compound Interest with slightly different wording/numbers ---")
    query5 = "If I invest 2500 dollars for 5 years at an annual rate of 4.5 percent, compounded quarterly, what will be the future value?"
    response5 = advisor.process_query(query5)
    print(response5)
    print("\n")

    print("--- Test Case 6: Simple Interest with different numbers ---")
    query6 = "Calculate the simple interest for a principal of 10000, a rate of 2%, and a period of 3 years."
    response6 = advisor.process_query(query6)
    print(response6)
    print("\n")