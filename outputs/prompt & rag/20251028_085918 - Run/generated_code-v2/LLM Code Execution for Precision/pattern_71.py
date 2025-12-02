import re
import io
import contextlib

def mock_llm_generate(prompt: str) -> str:
    """
    Simulates an LLM's response, which can either be a direct answer
    or generated Python code for calculations.
    """
    prompt_lower = prompt.lower()

    if "future value" in prompt_lower and "investment" in prompt_lower:
        # Example: Calculate the future value of an investment of $1000 at 5% annual interest for 10 years, compounded annually.
        # Simple regex to extract numbers for demonstration. A real LLM would be more robust.
        match = re.search(r'\$(\d+).*?(\d+)% annual interest.*?(\d+) years', prompt_lower)
        if match:
            principal = float(match.group(1))
            rate = float(match.group(2)) / 100
            time = int(match.group(3))
            return f"""<CODE>
principal = {principal}
rate = {rate}
time = {time}
future_value = principal * (1 + rate)**time
print(f"The future value of your investment will be: ${future_value:.2f}")
</CODE>"""
        else:
            return "Please provide the principal amount, interest rate, and number of years for the future value calculation."

    elif "monthly payment" in prompt_lower and "mortgage" in prompt_lower:
        # Example: What is the monthly payment for a $300,000 mortgage at 4% annual interest over 30 years?
        match = re.search(r'\$(\d{3}(?:,\d{3})*|\d+).*?(\d+)% annual interest.*?(\d+) years', prompt_lower)
        if match:
            principal = float(match.group(1).replace(',', ''))
            annual_rate = float(match.group(2)) / 100
            loan_term_years = int(match.group(3))

            return f"""<CODE>
principal = {principal}
annual_rate = {annual_rate}
loan_term_years = {loan_term_years}

monthly_rate = annual_rate / 12
number_of_payments = loan_term_years * 12

if monthly_rate == 0:
    monthly_payment = principal / number_of_payments
else:
    monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**number_of_payments) / (((1 + monthly_rate)**number_of_payments) - 1)

print(f"The estimated monthly mortgage payment is: ${monthly_payment:.2f}")
</CODE>"""
        else:
            return "Please provide the mortgage amount, annual interest rate, and loan term in years."

    elif "explain" in prompt_lower and "result" in prompt_lower:
        # This handles the follow-up explanation after code execution
        return f"Based on the calculation you requested, {{LLM_CALCULATION_RESULT_PLACEHOLDER}}. This result helps you understand your financial scenario better."

    else:
        # Default simple response for other queries
        return "I can help with financial calculations like future value of investments or mortgage payments. What would you like to calculate or know?"

def execute_python_code(code_string: str) -> str:
    """
    Executes a given Python code string and captures its stdout.
    Returns the captured output.
    """
    output_capture = io.StringIO()
    with contextlib.redirect_stdout(output_capture):
        try:
            exec(code_string, {'__builtins__': {}})
        except Exception as e:
            return f"Error during code execution: {e}"
    return output_capture.getvalue().strip()

def run_financial_advisor():
    """
    Main loop for the Smart Financial Advisor.
    """
    print("Welcome to the Smart Financial Advisor!\n")
    print("I can help you with financial calculations using code generation.\n")
    print("Try asking questions like: \"Calculate the future value of an investment of $1000 at 5% annual interest for 10 years.\"\n")
    print("Or: \"What is the monthly payment for a $300,000 mortgage at 4% annual interest over 30 years?\"\n")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Thank you for using the Smart Financial Advisor. Goodbye!")
            break

        llm_response = mock_llm_generate(user_query)

        code_match = re.search(r'<CODE>(.*?)</CODE>', llm_response, re.DOTALL)

        if code_match:
            generated_code = code_match.group(1).strip()
            print("\nAI (generating code for calculation...)\n")
            print("--- Generated Python Code ---")
            print(generated_code)
            print("-----------------------------")
            
            calculation_result = execute_python_code(generated_code)
            print(f"\nAI (executing code... Result): {calculation_result}\n")

            # Feed the result back to the LLM for a natural language explanation
            final_llm_prompt = f"Based on the following calculation result: '{calculation_result}'. Explain this result in simple terms: {user_query}"
            final_explanation = mock_llm_generate(f"explain result: {final_llm_prompt}")
            final_explanation = final_explanation.replace('{LLM_CALCULATION_RESULT_PLACEHOLDER}', calculation_result)
            print(f"AI: {final_explanation}\n")
        else:
            print(f"AI: {llm_response}\n")

if __name__ == "__main__":
    run_financial_advisor()