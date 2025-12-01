import io
import contextlib

def _simulate_llm_code_generation(query: str) -> str:
    try:
        if "compound interest" in query.lower():
            parts = query.lower().split()
            principal_index = -1
            rate_index = -1
            years_index = -1

            for i, part in enumerate(parts):
                if part == "for" and i + 1 < len(parts) and parts[i+1].replace('$', '').replace(',', '').isdigit():
                    principal_index = i + 1
                elif part == "at" and i + 1 < len(parts) and parts[i+1].replace('%', '').replace('.', '', 1).isdigit():
                    rate_index = i + 1
                elif part == "years" and i - 1 >= 0 and parts[i-1].isdigit():
                    years_index = i - 1
            
            if principal_index != -1 and rate_index != -1 and years_index != -1:
                principal = float(parts[principal_index].replace('$', '').replace(',', ''))
                rate = float(parts[rate_index].replace('%', '')) / 100
                years = int(parts[years_index])
                return f"""
p = {principal}
r = {rate}
t = {years}
n = 1
future_value = p * (1 + r/n)**(n*t)
print(f"The future value with compound interest is: {{future_value:.2f}}")
"""

        elif "simple interest" in query.lower():
            parts = query.lower().split()
            principal_index = -1
            rate_index = -1
            years_index = -1

            for i, part in enumerate(parts):
                if part == "for" and i + 1 < len(parts) and parts[i+1].replace('$', '').replace(',', '').isdigit():
                    principal_index = i + 1
                elif part == "at" and i + 1 < len(parts) and parts[i+1].replace('%', '').replace('.', '', 1).isdigit():
                    rate_index = i + 1
                elif part == "years" and i - 1 >= 0 and parts[i-1].isdigit():
                    years_index = i - 1

            if principal_index != -1 and rate_index != -1 and years_index != -1:
                principal = float(parts[principal_index].replace('$', '').replace(',', ''))
                rate = float(parts[rate_index].replace('%', '')) / 100
                years = int(parts[years_index])
                return f"""
p = {principal}
r = {rate}
t = {years}
simple_interest = p * r * t
total_amount = p + simple_interest
print(f"The simple interest is: {{simple_interest:.2f}}")
print(f"The total amount is: {{total_amount:.2f}}")
"""

        elif "future value" in query.lower() and "annuity" not in query.lower():
            parts = query.lower().split()
            principal_index = -1
            rate_index = -1
            years_index = -1

            for i, part in enumerate(parts):
                if part == "value" and i + 2 < len(parts) and parts[i+2].replace('$', '').replace(',', '').isdigit():
                    principal_index = i + 2
                elif part == "at" and i + 1 < len(parts) and parts[i+1].replace('%', '').replace('.', '', 1).isdigit():
                    rate_index = i + 1
                elif (part == "years" or part == "year") and i - 1 >= 0 and parts[i-1].isdigit():
                    years_index = i - 1

            if principal_index != -1 and rate_index != -1 and years_index != -1:
                principal = float(parts[principal_index].replace('$', '').replace(',', ''))
                rate = float(parts[rate_index].replace('%', '')) / 100
                years = int(parts[years_index])
                return f"""
principal = {principal}
rate = {rate}
years = {years}
future_value = principal * (1 + rate)**years
print(f"The future value is: {{future_value:.2f}}")
"""
    except Exception:
        return "print('Error: Could not parse your financial query. Please ensure correct numbers and keywords.')"

    return "print('Sorry, I can only perform simple financial calculations like compound interest, simple interest, and future value. Please rephrase your query.')"

def _simulate_llm_response_generation(original_query: str, calculation_result: str) -> str:
    if "Error" in calculation_result or "Sorry" in calculation_result:
        return f"I encountered an issue processing your request: {calculation_result}"
    return f"Based on your query '{original_query}', here is the financial calculation:\n{calculation_result}\nIs there anything else I can help you with?"

def execute_python_code(code: str) -> str:
    output_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_capture):
            exec(code)
        return output_capture.getvalue().strip()
    except Exception as e:
        return f"Code execution error: {e}"

def financial_chatbot():
    print("Welcome to the Financial Advisor Chatbot!")
    print("I can help with simple financial calculations like compound interest, simple interest, and future value.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nHow can I help you with your finances today? ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break

        generated_code = _simulate_llm_code_generation(user_query)
        calculation_result = execute_python_code(generated_code)
        final_response = _simulate_llm_response_generation(user_query, calculation_result)

        print("\nChatbot:", final_response)

if __name__ == "__main__":
    financial_chatbot()