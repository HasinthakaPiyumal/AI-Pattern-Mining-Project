import re
import math
import sys
import io

class FinCalcAI:
    def simulate_llm_code_generation(self, user_query: str) -> str:
        initial_savings = 0.0
        monthly_investment = 0.0
        annual_return_rate = 0.0
        years = 0
        inflation_rate = 0.0

        initial_savings_match = re.search(r"(?:current savings of|invest)\s?\$?([\d,.]+)", user_query, re.IGNORECASE)
        monthly_investment_match = re.search(r"monthly investment of \$?([\d,.]+)", user_query, re.IGNORECASE)
        annual_return_match = re.search(r"annual return of ([\d.]+)%", user_query, re.IGNORECASE)
        years_match = re.search(r"in ([\d.]+)\s?years", user_query, re.IGNORECASE)
        inflation_match = re.search(r"([\d.]+)%\s?annual inflation rate", user_query, re.IGNORECASE)

        if initial_savings_match:
            initial_savings = float(initial_savings_match.group(1).replace(",", ""))
        if monthly_investment_match:
            monthly_investment = float(monthly_investment_match.group(1).replace(",", ""))
        if annual_return_match:
            annual_return_rate = float(annual_return_match.group(1)) / 100
        if years_match:
            years = int(years_match.group(1))
        if inflation_match:
            inflation_rate = float(inflation_match.group(1)) / 100

        if (initial_savings == 0 and monthly_investment == 0) or years == 0 or annual_return_rate == 0:
            return "print('Error: Could not parse sufficient financial parameters from the query.')\nresult_value = None"

        generated_code = f"""
initial_savings = {initial_savings}
monthly_investment = {monthly_investment}
annual_return_rate = {annual_return_rate}
inflation_rate = {inflation_rate}
years = {years}

if years < 1:
    years = 1

monthly_return_rate = annual_return_rate / 12
number_of_months = years * 12

fv_initial_savings = 0.0
if initial_savings > 0:
    fv_initial_savings = initial_savings * (1 + annual_return_rate)**years

fv_monthly_investments = 0.0
if monthly_investment > 0 and monthly_return_rate > 0:
    fv_monthly_investments = monthly_investment * (( (1 + monthly_return_rate)**number_of_months - 1) / monthly_return_rate)
elif monthly_investment > 0 and monthly_return_rate == 0:
    fv_monthly_investments = monthly_investment * number_of_months

total_nominal_fv = fv_initial_savings + fv_monthly_investments

if inflation_rate > 0:
    result_value = total_nominal_fv / ((1 + inflation_rate)**years)
else:
    result_value = total_nominal_fv
"""
        return generated_code.strip()

    def execute_code_sandbox(self, code_to_execute: str) -> dict:
        local_scope = {'math': math}
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        try:
            exec(code_to_execute, {'__builtins__': {'abs', 'min', 'max', 'round', 'sum', 'len', 'pow', 'float', 'int', 'str', 'bool', 'range', 'print'}}, local_scope)
            output = redirected_output.getvalue()
            result = local_scope.get('result_value')
            return {"success": True, "result": result, "output": output}
        except Exception as e:
            return {"success": False, "error": str(e), "output": redirected_output.getvalue()}
        finally:
            sys.stdout = old_stdout

    def simulate_llm_explanation(self, original_query: str, calculated_result: float) -> str:
        if calculated_result is None:
            return "I was unable to perform the calculation based on your query. Please check the input and try again."

        formatted_result = f"${calculated_result:,.2f}"

        explanation = (
            f"Based on your query: '{original_query}', your projected financial value is approximately {formatted_result}.\n"
            "This calculation involved precise algorithmic execution to determine the future value, taking into account "
            "initial investments, recurring contributions, and growth rates. If an inflation rate was provided, the result "
            "has been adjusted to reflect real purchasing power.\n\n"
            "For detailed financial planning, it's always recommended to consult with a human financial advisor."
        )
        return explanation

    def process_financial_query(self, user_query: str) -> dict:
        print(f"User Query: {user_query}")

        generated_code = self.simulate_llm_code_generation(user_query)
        print("\n--- Generated Python Code ---")
        print(generated_code)

        execution_result = self.execute_code_sandbox(generated_code)

        if execution_result["success"]:
            calculated_value = execution_result["result"]
            print("\n--- Code Execution Result ---")
            print(f"Calculated Value: {calculated_value:,.2f}" if calculated_value is not None else "No numerical result obtained.")
            if execution_result["output"]:
                print(f"Console Output from Code:\n{execution_result['output']}")

            final_explanation = self.simulate_llm_explanation(user_query, calculated_value)
            print("\n--- FinCalcAI Response ---")
            print(final_explanation)
            return {"status": "success", "explanation": final_explanation, "calculated_value": calculated_value}
        else:
            error_message = f"Error during code execution: {execution_result['error']}"
            print("\n--- Code Execution Error ---")
            print(error_message)
            if execution_result["output"]:
                print(f"Console Output from Code:\n{execution_result['output']}")
            return {"status": "error", "message": error_message}

if __name__ == "__main__":
    fin_calc_ai = FinCalcAI()

    print("Complex Query Example:")
    query_complex = "Given my current savings of $50,000, monthly investment of $500, and an expected annual return of 7%, how much will I have in 20 years, assuming a 3% annual inflation rate?"
    fin_calc_ai.process_financial_query(query_complex)

    print("\n" + "="*80 + "\n")

    print("Simple Lump Sum Query Example:")
    query_simple_lump = "If I invest $1000 with an annual return of 5% for 10 years, how much will I have?"
    fin_calc_ai.process_financial_query(query_simple_lump)

    print("\n" + "="*80 + "\n")

    print("Monthly Investments Query Example:")
    query_monthly_only = "If I save $200 every month with an annual return of 6% for 15 years, what will be my total?"
    fin_calc_ai.process_financial_query(query_monthly_only)

    print("\n" + "="*80 + "\n")

    print("Lump Sum with Inflation Query Example:")
    query_lump_inflation = "What is the real value of $100,000 after 5 years if inflation is 2% annually and my investment has 0% return?"
    fin_calc_ai.process_financial_query(query_lump_inflation)

    print("\n" + "="*80 + "\n")

    print("Monthly Investments with Zero Return Query Example:")
    query_monthly_zero_return = "If I save $100 every month with an annual return of 0% for 5 years, what will be my total?"
    fin_calc_ai.process_financial_query(query_monthly_zero_return)

    print("\n" + "="*80 + "\n")

    print("Unparsable Query Example:")
    query_unparsable = "How much money will I have later?"
    fin_calc_ai.process_financial_query(query_unparsable)