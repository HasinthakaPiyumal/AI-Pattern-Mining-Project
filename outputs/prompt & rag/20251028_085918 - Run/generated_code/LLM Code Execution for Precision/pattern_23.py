import numpy as np
import sys
from io import StringIO

def simulate_llm_code_generation(user_query: str) -> str:
    if "net present value" in user_query.lower() or "npv" in user_query.lower():
        return """
import numpy as np
initial_investment = -10000.0
cash_flows = [3000.0, 4000.0, 5000.0, 6000.0]
discount_rate = 0.05
all_cash_flows = [initial_investment] + cash_flows
npv_result = np.npv(discount_rate, all_cash_flows)
print(f"NPV: {npv_result}")
        """
    elif "internal rate of return" in user_query.lower() or "irr" in user_query.lower():
        return """
import numpy as np
initial_investment = -10100.0
cash_flows = [3000.0, 4000.0, 5000.0, 3000.0]
all_cash_flows = [initial_investment] + cash_flows
irr_result = np.irr(all_cash_flows)
print(f"IRR: {irr_result}")
        """
    elif "simple interest" in user_query.lower():
        return """
principal = 1000.0
rate = 0.05
time = 3
simple_interest_result = principal * rate * time
print(f"Simple Interest: {simple_interest_result}")
        """
    else:
        return "print('I am sorry, I can only perform NPV, IRR, and simple interest calculations at the moment. Please refine your query.')"

def execute_generated_code(code: str) -> str:
    local_vars = {}
    old_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output
    try:
        exec(code, globals(), local_vars)
        exec_output = redirected_output.getvalue().strip()
        return exec_output
    except Exception as e:
        return f"Error during code execution: {e}"
    finally:
        sys.stdout = old_stdout

def simulate_llm_advice_generation(user_query: str, code_execution_result: str) -> str:
    if "net present value" in user_query.lower() or "npv" in user_query.lower():
        if "NPV:" in code_execution_result:
            try:
                npv_value = float(code_execution_result.split("NPV: ")[1].strip())
                if npv_value > 0:
                    return f"Based on your query and the calculated {code_execution_result}, the project has a positive NPV. This suggests the investment is likely to be profitable and should be considered. Remember to also assess other factors like risk and strategic fit."
                else:
                    return f"Based on your query and the calculated {code_execution_result}, the project has a non-positive NPV. This suggests the investment may not be profitable under the given conditions. You might want to reconsider or adjust the project parameters."
            except (ValueError, IndexError):
                return f"I calculated the NPV but had trouble parsing the result: {code_execution_result}. Please check your input and the generated code logic."
        else:
            return f"I calculated the NPV but encountered an issue: {code_execution_result}. Please check your input."
    elif "internal rate of return" in user_query.lower() or "irr" in user_query.lower():
        if "IRR:" in code_execution_result:
            try:
                irr_value = float(code_execution_result.split("IRR: ")[1].strip())
                return f"Based on your query and the calculated {code_execution_result}, you can compare this IRR to your required rate of return or hurdle rate. If the IRR is higher, the investment may be attractive. Be aware of multiple IRRs for non-conventional cash flows."
            except (ValueError, IndexError):
                return f"I calculated the IRR but had trouble parsing the result: {code_execution_result}. Please check your input and the generated code logic."
        else:
            return f"I calculated the IRR but encountered an issue: {code_execution_result}. Please check your input."
    elif "simple interest" in user_query.lower():
        if "Simple Interest:" in code_execution_result:
            return f"Based on your query, the {code_execution_result}. This is a straightforward calculation of interest earned over time without compounding."
        else:
            return f"I attempted to calculate simple interest but encountered an issue: {code_execution_result}. Please check your input."
    else:
        return f"I processed your request. Here's the raw calculation result: {code_execution_result}. Please note that I couldn't provide specific advice for this type of query yet, beyond the numerical output."

def financial_advisor_ai():
    print("Welcome to the Financial Advisor AI (PAL Demo)! Type 'exit' to quit.")
    while True:
        user_query = input("\nHow can I help you with your financial calculations today? ")
        if user_query.lower() == 'exit':
            break

        generated_code = simulate_llm_code_generation(user_query)
        print("\n--- Generated Python Code (Simulated LLM) ---")
        print(generated_code)

        execution_result = execute_generated_code(generated_code)
        print("\n--- Code Execution Result ---")
        print(execution_result)

        final_advice = simulate_llm_advice_generation(user_query, execution_result)
        print("\n--- Financial Advice (Simulated LLM) ---")
        print(final_advice)

if __name__ == "__main__":
    financial_advisor_ai()