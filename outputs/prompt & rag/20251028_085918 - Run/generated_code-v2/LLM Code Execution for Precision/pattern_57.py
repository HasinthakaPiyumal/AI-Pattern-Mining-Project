import re

class FinancialAdvisor:
    def __init__(self):
        pass

    def _generate_code(self, query: str) -> str:
        query_lower = query.lower()
        code = ""
        results_dict_template = "results = {}"

        if "retirement savings" in query_lower and "monthly contribute" in query_lower:
            # Extract numbers: monthly contribution, years, annual return, inflation
            monthly_contribution_match = re.search(r"\$(\d+)|(\d+)\s*monthly contribute", query_lower)
            years_match = re.search(r"(\d+)\s*years", query_lower)
            annual_return_match = re.search(r"(\d+\.?\d*)\%?\s*annual return", query_lower)
            inflation_match = re.search(r"(\d+\.?\d*)\%?\s*inflation", query_lower)

            try:
                monthly_contribution = float(monthly_contribution_match.group(1) or monthly_contribution_match.group(2)) if monthly_contribution_match else 0
                years = int(years_match.group(1)) if years_match else 0
                annual_return_rate = float(annual_return_match.group(1)) / 100 if annual_return_match else 0
                inflation_rate = float(inflation_match.group(1)) / 100 if inflation_match else 0
            except (AttributeError, ValueError):
                return "print('Error: Could not parse all required numbers for retirement savings.')\nresults = {'error': 'parsing_failed'}"

            code = f"""
monthly_contribution = {monthly_contribution}
years = {years}
annual_return_rate = {annual_return_rate}
inflation_rate = {inflation_rate}

if years > 0 and monthly_contribution > 0:
    num_months = years * 12
    monthly_return_rate = annual_return_rate / 12
    monthly_inflation_rate = inflation_rate / 12

    if monthly_inflation_rate >= 0 and monthly_return_rate >= monthly_inflation_rate:
        # Adjusted rate for real returns
        adjusted_monthly_rate = (1 + monthly_return_rate) / (1 + monthly_inflation_rate) - 1
    else:
        adjusted_monthly_rate = monthly_return_rate # No inflation adjustment or inflation > return

    future_value = 0
    if adjusted_monthly_rate != 0:
        future_value = monthly_contribution * (((1 + adjusted_monthly_rate)**num_months - 1) / adjusted_monthly_rate)
    else:
        future_value = monthly_contribution * num_months

    results = {{'type': 'retirement_savings', 'future_value': future_value, 'monthly_contribution': monthly_contribution, 'years': years, 'annual_return_rate': annual_return_rate, 'inflation_rate': inflation_rate}}
else:
    results = {{'type': 'retirement_savings', 'error': 'Invalid input for retirement calculation'}}
"""
        elif "loan payment" in query_lower:
            principal_match = re.search(r"\$(\d+\.?\d*)|(\d+\.?\d*)\s*loan principal", query_lower)
            interest_rate_match = re.search(r"(\d+\.?\d*)\%?\s*interest rate", query_lower)
            term_years_match = re.search(r"(\d+)\s*year[s]? term", query_lower)

            try:
                principal = float(principal_match.group(1) or principal_match.group(2)) if principal_match else 0
                annual_interest_rate = float(interest_rate_match.group(1)) / 100 if interest_rate_match else 0
                term_years = int(term_years_match.group(1)) if term_years_match else 0
            except (AttributeError, ValueError):
                return "print('Error: Could not parse all required numbers for loan payment.')\nresults = {'error': 'parsing_failed'}"

            code = f"""
principal = {principal}
annual_interest_rate = {annual_interest_rate}
term_years = {term_years}

if principal > 0 and annual_interest_rate >= 0 and term_years > 0:
    monthly_interest_rate = annual_interest_rate / 12
    num_payments = term_years * 12

    if monthly_interest_rate > 0:
        monthly_payment = principal * (monthly_interest_rate * (1 + monthly_interest_rate)**num_payments) / (((1 + monthly_interest_rate)**num_payments) - 1)
    else:
        monthly_payment = principal / num_payments # Simple division if no interest

    results = {{'type': 'loan_payment', 'monthly_payment': monthly_payment, 'principal': principal, 'annual_interest_rate': annual_interest_rate, 'term_years': term_years}}
else:
    results = {{'type': 'loan_payment', 'error': 'Invalid input for loan calculation'}}
"""
        else:
            code = "results = {'type': 'unsupported', 'message': 'I can only calculate retirement savings or loan payments currently.'}"

        return code

    def _execute_code(self, code: str) -> dict:
        local_vars = {}
        try:
            exec(code, {"__builtins__": {}}, local_vars)
            return local_vars.get("results", {"error": "No results dictionary found in executed code."})
        except Exception as e:
            return {"error": f"Code execution failed: {e}"}

    def _formulate_response(self, original_query: str, calculation_results: dict) -> str:
        if "error" in calculation_results:
            if calculation_results.get('error') == 'parsing_failed':
                return f"I apologize, but I couldn't understand all the numbers needed from your query: \"{original_query}\". Please ensure you provide all the necessary details like amounts, rates, and years clearly."
            return f"I encountered an error while processing your request: {calculation_results['error']}. Please try again."

        calculation_type = calculation_results.get("type")

        if calculation_type == "retirement_savings":
            future_value = calculation_results.get("future_value", 0)
            monthly_contribution = calculation_results.get("monthly_contribution", 0)
            years = calculation_results.get("years", 0)
            annual_return_rate = calculation_results.get("annual_return_rate", 0)
            inflation_rate = calculation_results.get("inflation_rate", 0)

            if "error" in calculation_results:
                return f"I couldn't perform the retirement savings calculation due to invalid input. Please check your monthly contribution, years, and rates."

            return (f"Based on your input, if you contribute ${monthly_contribution:,.2f} monthly for {years} years, "
                    f"with an estimated annual return of {annual_return_rate:.2%} and {inflation_rate:.2%} inflation, "
                    f"your retirement savings could be approximately ${future_value:,.2f}.")

        elif calculation_type == "loan_payment":
            monthly_payment = calculation_results.get("monthly_payment", 0)
            principal = calculation_results.get("principal", 0)
            annual_interest_rate = calculation_results.get("annual_interest_rate", 0)
            term_years = calculation_results.get("term_years", 0)

            if "error" in calculation_results:
                return f"I couldn't perform the loan payment calculation due to invalid input. Please check your principal, interest rate, and loan term."

            return (f"For a loan principal of ${principal:,.2f} with an annual interest rate of {annual_interest_rate:.2%} "
                    f"over a {term_years}-year term, your estimated monthly payment would be ${monthly_payment:,.2f}.")

        else:
            return calculation_results.get("message", "I could not process your request. Please ask about retirement savings or loan payments.")

    def process_query(self, query: str) -> str:
        # Step 1: LLM (mock) generates code
        generated_code = self._generate_code(query)

        # Step 2: Execute the code
        calculation_results = self._execute_code(generated_code)

        # Step 3: LLM (mock) formulates the response
        final_response = self._formulate_response(query, calculation_results)
        return final_response

if __name__ == "__main__":
    advisor = FinancialAdvisor()

    print("\n--- Retirement Savings Scenario 1 ---")
    query1 = "Calculate my retirement savings if I monthly contribute $500 for 20 years with an average 7% annual return, adjusting for 2% inflation."
    response1 = advisor.process_query(query1)
    print(f"Query: {query1}")
    print(f"Advisor: {response1}")

    print("\n--- Retirement Savings Scenario 2 (Edge Case: No Inflation) ---")
    query2 = "What will my retirement savings be if I put in 1000 monthly for 30 years at 5% annual return?"
    response2 = advisor.process_query(query2)
    print(f"Query: {query2}")
    print(f"Advisor: {response2}")

    print("\n--- Loan Payment Scenario 1 ---")
    query3 = "What is the monthly loan payment for a loan principal of $200000 with a 4.5% interest rate over a 30-year term?"
    response3 = advisor.process_query(query3)
    print(f"Query: {query3}")
    print(f"Advisor: {response3}")

    print("\n--- Loan Payment Scenario 2 (Missing Info) ---")
    query4 = "Calculate my loan payment for a principal of 50000."
    response4 = advisor.process_query(query4)
    print(f"Query: {query4}")
    print(f"Advisor: {response4}")

    print("\n--- Unsupported Query ---")
    query5 = "Tell me a joke."
    response5 = advisor.process_query(query5)
    print(f"Query: {query5}")
    print(f"Advisor: {response5}")

    print("\n--- Retirement Savings Scenario 3 (Parsing Error) ---")
    query6 = "Calculate my retirement savings if I monthly contribute for 20 years with an average 7% annual return, adjusting for 2% inflation."
    response6 = advisor.process_query(query6)
    print(f"Query: {query6}")
    print(f"Advisor: {response6}")

    print("\n--- Retirement Savings Scenario 4 (Inflation > Return) ---")
    query7 = "Calculate my retirement savings if I monthly contribute $500 for 20 years with an average 2% annual return, adjusting for 5% inflation."
    response7 = advisor.process_query(query7)
    print(f"Query: {query7}")
    print(f"Advisor: {response7}")

