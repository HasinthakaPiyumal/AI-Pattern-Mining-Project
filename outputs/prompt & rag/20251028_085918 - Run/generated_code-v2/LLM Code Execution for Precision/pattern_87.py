import io
import contextlib

def generate_financial_code(user_query: str) -> tuple[str, str]:
    """
    Simulates an LLM generating Python code for financial calculations.
    For demonstration, it hardcodes a specific calculation based on keywords.
    In a real PAL system, this would be an actual LLM call.
    """
    if "compound annual growth rate" in user_query.lower() and "stock" in user_query.lower():
        code = """
start_value = 1000  # Initial investment
end_value = 1610.51 # Final value after growth (example for ~10% CAGR over 5 years)
years = 5           # Number of years

# Calculate CAGR
cagr = ((end_value / start_value)**(1/years) - 1) * 100
print(f"The Compound Annual Growth Rate (CAGR) is: {cagr:.2f}%")
"""
        explanation = "Python code generated to calculate Compound Annual Growth Rate (CAGR)."
        return code, explanation
    elif "discounted cash flow" in user_query.lower() or "npv" in user_query.lower():
        code = """
cash_flows = [-100000, 30000, 40000, 50000, 20000] # Initial investment and subsequent cash flows
discount_rate = 0.10 # 10% discount rate

npv = 0
for i, cf in enumerate(cash_flows):
    npv += cf / ((1 + discount_rate)**i)

print(f"The Net Present Value (NPV) is: {npv:.2f}")
"""
        explanation = "Python code generated to calculate Net Present Value (NPV)."
        return code, explanation
    else:
        return "", "No specific financial calculation code generated. The LLM would attempt to answer directly or ask for clarification."

def execute_python_code(code: str) -> str:
    """
    Executes the given Python code in a controlled environment and captures its output.
    WARNING: Using exec() can be dangerous in production. A secure sandbox is crucial.
    """
    if not code:
        return "No code to execute."

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            exec(code, {'__builtins__': {}}, {})
        except Exception as e:
            return f"Error during code execution: {e}"
    return f.getvalue().strip()

def formulate_response(user_query: str, code_output: str, llm_explanation: str) -> str:
    """
    Combines the LLM's initial thoughts/query and the code execution result
    to formulate a comprehensive natural language answer.
    """
    response = f"Based on your query: \"{user_query}\"\n\n"
    if code_output and "No code to execute" not in code_output and "Error during code execution" not in code_output:
        response += "I've performed the necessary calculations using a precise computational engine.\n"
        response += f"Here are the results:\n{code_output}\n\n"
        response += "This approach ensures accuracy for complex financial computations."
    else:
        response += f"I processed your request. {llm_explanation}\n"
        if "Error during code execution" in code_output:
            response += f"There was an issue with the calculation: {code_output}\n"
        elif "No code to execute" in code_output:
            response += "The query did not require a specific code-based calculation."
        else:
            response += "I couldn't retrieve a numerical result for this specific query, or the query did not necessitate code execution."

    response += "\n\nIf you have further financial analysis needs or specific parameters, please let me know!"
    return response

def main():
    print("Welcome to the Smart Financial Analyst Assistant!")
    print("I can help you with complex financial calculations by generating and executing code.")
    print("Try asking about 'Compound Annual Growth Rate for a stock' or 'Net Present Value'.")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYour financial query: ")
        if user_input.lower() == 'exit':
            break

        generated_code, llm_explanation = generate_financial_code(user_input)

        if generated_code:
            print("\n--- LLM generated code (simulated) ---")
            print(generated_code.strip())
            print("--------------------------------------")

            execution_result = execute_python_code(generated_code)
            print("\n--- Code Execution Output ---")
            print(execution_result)
            print("---------------------------")
        else:
            execution_result = "No code was generated for this query."
            print(f"\n--- Assistant thought ---")
            print(llm_explanation)
            print(f"-------------------------")

        final_answer = formulate_response(user_input, execution_result, llm_explanation)
        print("\n--- Smart Financial Analyst Assistant's Answer ---")
        print(final_answer)
        print("--------------------------------------------------")

if __name__ == "__main__":
    main()