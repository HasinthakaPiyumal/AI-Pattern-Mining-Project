import re
import io
import sys
import numpy as np

def _calculate_npv(initial_investment: float, cash_flows: list[float], discount_rate: float) -> float:
    npv = initial_investment
    for i, cash_flow in enumerate(cash_flows):
        npv += cash_flow / ((1 + discount_rate) ** (i + 1))
    return npv

def _calculate_irr(cash_flows: list[float]) -> float:
    try:
        return np.irr(cash_flows)
    except ValueError:
        return float('nan')

def _calculate_payback_period(initial_investment: float, cash_flows: list[float]) -> float:
    remaining_investment = initial_investment
    payback_period = 0.0
    
    for i, cash_flow in enumerate(cash_flows):
        if remaining_investment <= 0:
            break
        if remaining_investment > cash_flow:
            remaining_investment -= cash_flow
            payback_period += 1
        else:
            payback_period += remaining_investment / cash_flow
            remaining_investment = 0
            break

    if remaining_investment > 0:
        return float('inf')
    return payback_period

def _generate_code_from_query(query: str) -> str:
    query_lower = query.lower()
    generated_code = ""

    if "calculate npv" in query_lower:
        initial_investment_match = re.search(r"initial investment (\d+)", query_lower)
        cash_flows_match = re.search(r"cash flows \[([\d,\s]+)\]", query_lower)
        discount_rate_match = re.search(r"discount rate ([\d.]+)", query_lower)

        if initial_investment_match and cash_flows_match and discount_rate_match:
            initial_investment = float(initial_investment_match.group(1))
            cash_flows_str = cash_flows_match.group(1).replace(' ', '')
            cash_flows = [float(x) for x in cash_flows_str.split(',')]
            discount_rate = float(discount_rate_match.group(1))
            
            generated_code = (
                f"result = _calculate_npv({initial_investment}, {cash_flows}, {discount_rate})\n"\
                "print(result)"
            )

    elif "calculate irr" in query_lower:
        cash_flows_match = re.search(r"cash flows \[([\d,\s\-]+)\]", query_lower)
        if cash_flows_match:
            cash_flows_str = cash_flows_match.group(1).replace(' ', '')
            cash_flows = [float(x) for x in cash_flows_str.split(',')]
            generated_code = (
                f"result = _calculate_irr({cash_flows})\n"\
                "print(result)"
            )

    elif "calculate payback period" in query_lower:
        initial_investment_match = re.search(r"initial investment (\d+)", query_lower)
        cash_flows_match = re.search(r"cash flows \[([\d,\s]+)\]", query_lower)

        if initial_investment_match and cash_flows_match:
            initial_investment = float(initial_investment_match.group(1))
            cash_flows_str = cash_flows_match.group(1).replace(' ', '')
            cash_flows = [float(x) for x in cash_flows_str.split(',')]
            
            generated_code = (
                f"result = _calculate_payback_period({initial_investment}, {cash_flows})\n"\
                "print(result)"
            )

    return generated_code

def _execute_python_code(code_string: str) -> any:
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    
    execution_globals = {
        '__builtins__': None,
        'print': print,
        '_calculate_npv': _calculate_npv,
        '_calculate_irr': _calculate_irr,
        '_calculate_payback_period': _calculate_payback_period,
        'np': np
    }
    
    try:
        exec(code_string, execution_globals, {})
        output = redirected_output.getvalue().strip()
        if output:
            try:
                return float(output)
            except ValueError:
                return output
        return None
    finally:
        sys.stdout = old_stdout

def _synthesize_response(original_query: str, computational_result: any) -> str:
    response_template = ""

    if "npv" in original_query.lower():
        response_template = f"Based on your query regarding NPV calculation, the Net Present Value is: {computational_result:.2f}. "\
                            "This indicates the profitability of the project, considering the time value of money."
    elif "irr" in original_query.lower():
        if computational_result is not None and not np.isnan(computational_result):
            response_template = f"For your Internal Rate of Return (IRR) query, the calculated IRR is: {computational_result:.2%}. "\
                                "This represents the discount rate at which the net present value of all cash flows is zero."
        else:
            response_template = "Could not calculate the Internal Rate of Return (IRR) for the provided cash flows. "\
                                "Please check your inputs."
    elif "payback period" in original_query.lower():
        if computational_result == float('inf'):
            response_template = "The investment's payback period could not be determined or the investment is never paid back with the given cash flows."
        else:
            response_template = f"The calculated Payback Period for your investment is: {computational_result:.2f} years. "\
                                "This is the time it takes for an investment to generate cash flows that cover its initial cost."
    else:
        response_template = f"For your query: '{original_query}', the computational result is: {computational_result}. "\
                            "I can provide more detailed financial analysis if you specify the type of calculation."
    
    return response_template

def main():
    print("AI-Powered Financial Analyst Assistant (PAL Prompting)")
    print("--------------------------------------------------")
    
    while True:
        user_query = input("\nEnter your financial query (e.g., 'Calculate NPV for initial investment 10000, cash flows [3000, 4000, 5000], discount rate 0.1'):\n")
        if user_query.lower() == 'exit':
            break

        print(f"\nProcessing query: '{user_query}'...")

        generated_code = _generate_code_from_query(user_query)
        print("\n--- Generated Code ---")
        print(generated_code)

        if not generated_code:
            print("Could not generate code for the given query. Please try again with a different query.")
            continue

        print("\n--- Executing Code ---")
        try:
            execution_output = _execute_python_code(generated_code)
            print(f"Code executed successfully. Output: {execution_output}")
        except Exception as e:
            print(f"Error during code execution: {e}")
            continue

        final_answer = _synthesize_response(user_query, execution_output)
        print("\n--- Final Answer ---")
        print(final_answer)
        print("\n--------------------------------------------------")

    print("Exiting Financial Analyst Assistant. Goodbye!")

if __name__ == "__main__":
    main()