import io
import sys

def simulate_llm_code_generation(query: str) -> str:
    """Simulates an LLM generating Python code based on a financial query."""
    query_lower = query.lower()
    if "what is" in query_lower and "% of" in query_lower:
        parts = query_lower.split("what is")[-1].strip().split("% of")
        try:
            percentage = float(parts[0].strip())
            number = float(parts[1].strip().replace('?', ''))
            return f"result = ({percentage} / 100) * {number}; print(result)"
        except ValueError:
            return "print('Error: Could not parse percentage or number.')"
    elif "simple interest" in query_lower and "principal" in query_lower and "rate" in query_lower and "time" in query_lower:
        try:
            principal_str = query_lower.split("principal ")[1].split(',')[0].strip()
            rate_str = query_lower.split("rate ")[1].split(',')[0].strip().replace('%', '')
            time_str = query_lower.split("time ")[1].split(' ')[0].strip().replace('years', '').replace('year', '')

            principal = float(principal_str)
            rate = float(rate_str) / 100
            time = float(time_str)
            return f"interest = {principal} * {rate} * {time}; print(interest)"
        except (ValueError, IndexError):
            return "print('Error: Could not parse simple interest parameters.')"
    elif "compound interest" in query_lower and "principal" in query_lower and "rate" in query_lower and "time" in query_lower and "compounded" in query_lower:
        try:
            principal_str = query_lower.split("principal ")[1].split(',')[0].strip()
            rate_str = query_lower.split("rate ")[1].split(',')[0].strip().replace('%', '')
            time_str = query_lower.split("time ")[1].split(' ')[0].strip().replace('years', '').replace('year', '')
            n_str = query_lower.split("compounded ")[1].split(' ')[0].strip()

            principal = float(principal_str)
            rate = float(rate_str) / 100
            time = float(time_str)
            n = float(n_str)
            return f"amount = {principal} * (1 + {rate}/{n})**({n}*{time}); interest = amount - {principal}; print(interest)"
        except (ValueError, IndexError):
            return "print('Error: Could not parse compound interest parameters.')"
    else:
        return "print('I can only perform specific financial calculations for now (e.g., percentage, simple/compound interest). Please rephrase your query.')"

def execute_python_code(code: str) -> str:
    """Executes Python code in an isolated environment and captures its output."""
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        exec(code)
    except Exception as e:
        return f"Execution Error: {e}"
    finally:
        sys.stdout = old_stdout  # Restore original stdout
    return redirected_output.getvalue().strip()

def simulate_llm_response_generation(original_query: str, code_output: str) -> str:
    """Simulates an LLM integrating code output into a natural language response."""
    if "Error" in code_output or "Execution Error" in code_output:
        return f"I encountered an error while trying to process your request: {code_output}"

    query_lower = original_query.lower()
    if "what is" in query_lower and "% of" in query_lower:
        return f"Based on my calculations, {original_query} is {code_output}."
    elif "simple interest" in query_lower:
        return f"The simple interest for your specified parameters is {code_output}."
    elif "compound interest" in query_lower:
        return f"The compound interest for your specified parameters is {code_output}."
    else:
        return f"Here is the result of my computation: {code_output}. " \
               f"If this doesn't fully answer your {original_query} query, please provide more context."

def main():
    print("\nWelcome to the Smart Financial Analyst Assistant!")
    print("I can help with specific financial calculations using program-aided reasoning.")
    print("Try queries like: 'What is 15% of 200?' or 'Calculate simple interest for principal 1000, rate 5%, time 2 years.'")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYour financial query: ")
        if user_query.lower() == 'exit':
            print("Thank you for using the Smart Financial Analyst Assistant. Goodbye!")
            break

        # Step 1: LLM generates Python code
        generated_code = simulate_llm_code_generation(user_query)
        print(f"\n[Assistant - Generated Code]: {generated_code}")

        # Step 2: Execute the generated code
        execution_result = execute_python_code(generated_code)
        print(f"[Assistant - Code Output]: {execution_result}")

        # Step 3: LLM formulates natural language response
        final_response = simulate_llm_response_generation(user_query, execution_result)
        print(f"\n[Assistant - Response]: {final_response}")

if __name__ == "__main__":
    main()