import io
import sys
import math

def simulate_llm_code_generation(prompt: str) -> str:
    """
    Simulates an LLM generating Python code based on a financial query.
    For demonstration, it extracts specific parameters and constructs a Python script
    to calculate future value with compound interest and monthly contributions.
    """
    # In a real PAL system, an LLM would parse the prompt and generate this code dynamically.
    # Here, we hardcode the parsing for a specific example to illustrate the concept.

    if "future value of an investment" in prompt.lower() and \
       "compound interest" in prompt.lower() and \
       "monthly contributions" in prompt.lower():

        # --- Parameter Extraction (Simulated) ---
        # A real LLM would use its language understanding to extract these values.
        # For this demo, we're assuming a predefined structure or specific values for illustration.
        principal = 10000.0
        annual_rate = 0.05
        years = 10.0
        monthly_contribution = 100.0

        # --- Code Generation by LLM ---
        # The LLM generates a self-contained Python script to perform the calculation.
        generated_code = f"""
import math

def calculate_future_value(principal, annual_rate, years, monthly_contribution):
    # Convert annual rate to monthly rate
    monthly_rate = annual_rate / 12
    # Total number of months
    n_months = years * 12

    # Future value of initial principal (compounded monthly)
    # FV_principal = P * (1 + r/n)^(nt)
    fv_principal = principal * (1 + monthly_rate)**n_months

    # Future value of an ordinary annuity (monthly contributions compounded monthly)
    # FV_annuity = PMT * [((1 + r)^n - 1) / r]
    fv_contributions = monthly_contribution * (((1 + monthly_rate)**n_months - 1) / monthly_rate)

    total_fv = fv_principal + fv_contributions
    return round(total_fv, 2)

# Parameters extracted from the user's prompt
principal = {principal}
annual_rate = {annual_rate}
years = {years}
monthly_contribution = {monthly_contribution}

# Perform the calculation
result = calculate_future_value(principal, annual_rate, years, monthly_contribution)
print(result)
"""
        return generated_code
    else:
        return "print(\"Error: Could not generate code for this specific financial request.\")"

def execute_python_code(code: str) -> str:
    """
    Executes the given Python code string in a controlled environment
    and captures its standard output.
    """
    output_capture = io.StringIO()
    original_stdout = sys.stdout
    try:
        sys.stdout = output_capture
        # Execute the code. Use a limited global/local scope for security if needed
        exec(code, {}, {})
        return output_capture.getvalue().strip()
    except Exception as e:
        return f"Execution Error: {e}"
    finally:
        sys.stdout = original_stdout # Ensure stdout is restored

def generate_llm_response(original_prompt: str, calculation_result: str) -> str:
    """
    Integrates the numerical calculation result back into a natural language response.
    """
    if "Execution Error" in calculation_result:
        return f"I apologize, but I encountered an error during the calculation: {calculation_result}. Please ensure all parameters are correctly specified."
    else:
        return f"Based on your inquiry about: \"{original_prompt}\", the calculated future value of your investment is approximately ${calculation_result}. This calculation considers your initial principal and monthly contributions, compounded over the specified period."

# --- Main Application Flow Simulation ---
if __name__ == "__main__":
    # 1. User provides a complex financial query
    user_query = "Calculate the future value of an investment with compound interest over 10 years at a 5% annual rate, with an initial principal of $10,000 and additional monthly contributions of $100."
    print(f"\nUser Query: {user_query}")

    # 2. LLM (simulated) generates Python code for the calculation
    generated_code = simulate_llm_code_generation(user_query)
    print(f"\n--- Generated Python Code (by LLM) ---\n{generated_code}\n---")

    # 3. Execute the generated Python code
    calculation_output = execute_python_code(generated_code)
    print(f"\n--- Code Execution Output ---\n{calculation_output}\n---")

    # 4. LLM integrates the result into a natural language response
    final_response = generate_llm_response(user_query, calculation_output)
    print(f"\nAI Financial Advisor Response: {final_response}\n")

    # Example with a different query (unhandled by simple sim LLM)
    user_query_2 = "What is the capital gains tax on selling a stock after 2 years?"
    print(f"\nUser Query: {user_query_2}")
    generated_code_2 = simulate_llm_code_generation(user_query_2)
    print(f"\n--- Generated Python Code (by LLM) ---\n{generated_code_2}\n---")
    calculation_output_2 = execute_python_code(generated_code_2)
    final_response_2 = generate_llm_response(user_query_2, calculation_output_2)
    print(f"\nAI Financial Advisor Response: {final_response_2}\n")
