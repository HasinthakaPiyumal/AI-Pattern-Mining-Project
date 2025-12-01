import io
import sys
import contextlib
import math
import pandas as pd
# import yfinance as yf # Commented out for direct execution without external installs, but mentioned in explanation

def execute_python_code(code_string: str, global_vars: dict = None, local_vars: dict = None) -> dict:
    """
    Executes a given Python code string in a sandboxed environment and captures its output.
    WARNING: Executing arbitrary code from untrusted sources can be a security risk.
             For a production system, a more robust sandboxing mechanism (e.g., containers)
             should be implemented.

    Args:
        code_string: The Python code to execute as a string.
        global_vars: A dictionary for global variables accessible during execution.
        local_vars: A dictionary for local variables accessible during execution.

    Returns:
        A dictionary containing 'result' (return value or exception), 'stdout' (printed output),
        and 'error' (error message if any).
    """
    if global_vars is None:
        global_vars = {"math": math, "pd": pd} # Provide common modules
        # global_vars["yf"] = yf # If yfinance is installed and desired
    if local_vars is None:
        local_vars = {}

    stdout_capture = io.StringIO()
    exception_occurred = None
    execution_result = None

    try:
        with contextlib.redirect_stdout(stdout_capture):
            # Use a dummy function to capture a return value from exec if needed
            # For simplicity, we'll rely on global/local var updates or print statements
            exec(code_string, global_vars, local_vars)
            execution_result = local_vars.get('__result__', None) # A common pattern if the code assigns to __result__
    except Exception as e:
        exception_occurred = str(e)
        # Capture traceback as well if detailed error is needed
        # import traceback
        # exception_occurred = traceback.format_exc()
    
    return {
        "result": execution_result,
        "stdout": stdout_capture.getvalue(),
        "error": exception_occurred,
        "global_vars_after_exec": global_vars, # Useful for inspecting state changes
        "local_vars_after_exec": local_vars
    }

def generate_financial_code(query: str) -> str:
    """
    Simulates an LLM generating Python code for financial calculations based on a query.
    In a real PAL system, this would be an actual LLM call.
    """
    query_lower = query.lower()

    if "net present value" in query_lower or "npv" in query_lower:
        # Example: Calculate NPV
        # For a real LLM, it would parse the values from the query
        return """
def calculate_npv(initial_investment, cash_flows, discount_rate):
    npv = -initial_investment
    for i, cf in enumerate(cash_flows):
        npv += cf / ((1 + discount_rate)**(i + 1))
    return npv

# Assuming specific values from a parsed query
initial_investment = 100000
cash_flows = [30000, 40000, 50000]
discount_rate = 0.10

__result__ = calculate_npv(initial_investment, cash_flows, discount_rate)
print(f"Calculated NPV: {__result__:.2f}")
"""
    elif "compound annual growth rate" in query_lower or "cagr" in query_lower:
        # Example: Calculate CAGR for a single stock
        # In a real scenario, this would involve fetching data (e.g., yfinance)
        # and parsing the stock ticker and period from the query.
        return """
import math
import pandas as pd
# import yfinance as yf # Uncomment and install yfinance for real data fetching

def calculate_cagr(start_value, end_value, num_years):
    if num_years <= 0:
        return 0
    return ((end_value / start_value)**(1 / num_years)) - 1

# Simulate fetching data or use hardcoded values for demonstration
# For a real application:
# try:
#     data = yf.download("AAPL", start="2018-01-01", end="2023-01-01")['Adj Close']
#     start_price = data.iloc[0]
#     end_price = data.iloc[-1]
#     num_years = 5 # (data.index[-1].year - data.index[0].year)
# except Exception:
#     print("Could not fetch real-time data for AAPL. Using simulated data.")
start_price_aapl = 100.00 # Simulated start price
end_price_aapl = 175.00   # Simulated end price
num_years_aapl = 5

start_price_msft = 90.00 # Simulated start price
end_price_msft = 220.00  # Simulated end price
num_years_msft = 5

cagr_aapl = calculate_cagr(start_price_aapl, end_price_aapl, num_years_aapl)
cagr_msft = calculate_cagr(start_price_msft, end_price_msft, num_years_msft)

__result__ = {"AAPL_CAGR": cagr_aapl, "MSFT_CAGR": cagr_msft}
print(f"AAPL CAGR: {cagr_aapl:.4f}")
print(f"MSFT CAGR: {cagr_msft:.4f}")
"""
    elif "fibonacci" in query_lower: # An example of a non-financial algorithm
        return """
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a, end=" ")
        a, b = b, a + b
    return a # return the nth fib number

num_terms = 10
print(f"Fibonacci sequence up to {num_terms} terms:")
__result__ = fibonacci(num_terms)
print(f"\\n{num_terms}-th Fibonacci number: {__result__}")
"""
    else:
        return "print('Error: Could not generate code for the given financial query.')\n__result__ = None"

def formulate_natural_language_response(query: str, execution_result: dict) -> str:
    """
    Simulates an LLM integrating the execution result into a natural language response.
    """
    if execution_result["error"]:
        return (f"I encountered an error while processing your request: {execution_result['error']}\n"
                f"Please check the query or the underlying code.")

    if "net present value" in query.lower() or "npv" in query.lower():
        npv_value = execution_result["result"]
        if npv_value is not None:
            return (f"Based on my calculations, the Net Present Value (NPV) for your project "
                    f"is: ${npv_value:,.2f}. This was calculated by executing a program.")
        else:
            return "I have calculated the NPV, but the result was not explicitly returned. Here is the program output:\n" + execution_result["stdout"]
    
    elif "compound annual growth rate" in query.lower() or "cagr" in query.lower():
        cagr_values = execution_result["result"]
        if isinstance(cagr_values, dict):
            response_parts = ["Here are the Compound Annual Growth Rates (CAGR) based on my calculations:"]
            for ticker, cagr in cagr_values.items():
                response_parts.append(f"- {ticker}: {cagr:.2%}")
            
            # Simple comparison
            aapl_cagr = cagr_values.get("AAPL_CAGR", 0)
            msft_cagr = cagr_values.get("MSFT_CAGR", 0)
            if aapl_cagr > msft_cagr:
                response_parts.append("Based on these simulated values, Apple (AAPL) performed better.")
            elif msft_cagr > aapl_cagr:
                response_parts.append("Based on these simulated values, Microsoft (MSFT) performed better.")
            else:
                response_parts.append("Based on these simulated values, both performed similarly or data was insufficient for comparison.")
            
            return "\n".join(response_parts)
        else:
            return "I have calculated the CAGR, but the result was not explicitly returned. Here is the program output:\n" + execution_result["stdout"]
            
    elif "fibonacci" in query.lower():
        fib_number = execution_result["result"]
        stdout_output = execution_result["stdout"].strip()
        return (f"Here is the Fibonacci sequence generated by the program:\n{stdout_output}\n"
                f"The Nth Fibonacci number requested was: {fib_number}.")

    return f"I have processed your request. Here is the raw output from the program:\nResult: {execution_result['result']}\nStdout: {execution_result['stdout']}"


def run_financier_ai(query: str) -> str:
    """
    Orchestrates the Financier AI process:
    1. Generates Python code from the query (simulated LLM).
    2. Executes the generated code.
    3. Formulates a natural language response based on the execution result (simulated LLM).
    """
    print(f"\nUser Query: \"{query}\"")

    # Step 1: LLM Generates Code
    generated_code = generate_financial_code(query)
    print("\n--- Generated Python Code ---")
    print(generated_code)
    print("-----------------------------\n")

    # Step 2: Code Execution
    print("--- Executing Code ---")
    execution_output = execute_python_code(generated_code)
    print("----------------------\n")

    print(f"Code Execution Result: {execution_output['result']}")
    print(f"Code Stdout: {execution_output['stdout'].strip()}")
    if execution_output['error']:
        print(f"Code Error: {execution_output['error']}")
    
    # Step 3: Output Integration & Final Answer Formulation
    final_response = formulate_natural_language_response(query, execution_output)
    print("\n--- Financier AI Response ---")
    print(final_response)
    print("-----------------------------\n")
    return final_response

if __name__ == "__main__":
    # Example 1: NPV Calculation
    run_financier_ai("Calculate the net present value (NPV) for a project with initial investment $100,000, cash inflows of $30,000, $40,000, $50,000 over three years, and a discount rate of 10%.")

    # Example 2: CAGR Comparison (simulated data)
    run_financier_ai("Compare the compound annual growth rate (CAGR) of Apple (AAPL) and Microsoft (MSFT) stock over the last 5 years, assuming dividend reinvestment, and suggest which performed better.")

    # Example 3: General algorithmic task (Fibonacci)
    run_financier_ai("Generate the first 10 terms of the Fibonacci sequence and tell me the 10th number.")

    # Example 4: Unrecognized query
    run_financier_ai("What is the square root of 12345?")