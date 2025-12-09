import io
import sys
import re
import statistics

def simulate_llm_code_generation(prompt: str) -> str:
    if "standard deviation" in prompt.lower() and "portfolio" in prompt.lower():
        match = re.search(r'\[([\d.,\s-]+)\]', prompt)
        if match:
            returns_str = match.group(1)
            return f"import statistics\nreturns = [{returns_str}]\nstd_dev = statistics.stdev(returns)\nprint(f'Standard Deviation: {{std_dev}}')"
        else:
            return "print('Error: Could not parse portfolio returns from the prompt.')"
    elif "mean" in prompt.lower() and "portfolio" in prompt.lower():
        match = re.search(r'\[([\d.,\s-]+)\]', prompt)
        if match:
            returns_str = match.group(1)
            return f"import statistics\nreturns = [{returns_str}]\nmean_return = statistics.mean(returns)\nprint(f'Mean Return: {{mean_return}}')"
        else:
            return "print('Error: Could not parse portfolio returns from the prompt.')"
    elif "sum" in prompt.lower() and "numbers" in prompt.lower():
        match = re.search(r'\[([\d.,\s-]+)\]', prompt)
        if match:
            numbers_str = match.group(1)
            return f"numbers = [{numbers_str}]\ntotal_sum = sum(numbers)\nprint(f'Sum: {{total_sum}}')"
        else:
            return "print('Error: Could not parse numbers from the prompt.')"
    return "print('Sorry, I can only generate code for basic financial calculations like standard deviation or mean, or sum of numbers.')"

def execute_python_code(code: str) -> str:
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        exec(code, {})
        output = redirected_output.getvalue().strip()
    except Exception as e:
        output = f"Execution Error: {e}"
    finally:
        sys.stdout = old_stdout
    return output

def simulate_llm_interpret_and_advise(results: str, original_prompt: str) -> str:
    if "Standard Deviation:" in results:
        std_dev = float(results.split(" ")[-1])
        if std_dev > 0.05:
            return f"Based on your portfolio returns, the calculated Standard Deviation is {std_dev:.4f}. This indicates a relatively higher level of volatility in your portfolio. You might want to consider diversifying or re-evaluating your risk tolerance."
        else:
            return f"Based on your portfolio returns, the calculated Standard Deviation is {std_dev:.4f}. This suggests a relatively stable portfolio with lower volatility. Good job on managing risk!"
    elif "Mean Return:" in results:
        mean_return = float(results.split(" ")[-1])
        return f"Your portfolio's Mean Return is {mean_return:.4f}. This represents the average return you can expect from your investments over the period analyzed."
    elif "Sum:" in results:
        total_sum = float(results.split(" ")[-1])
        return f"The sum of the numbers you provided is {total_sum}."
    elif "Error:" in results or "Execution Error:" in results:
        return f"I encountered an issue during calculation: {results}. Please rephrase your request or check the input data."
    return f"I have processed the calculation and the result is: {results}."

def run_assistant():
    print("Welcome to the Financial Portfolio Optimization Assistant!\n")
    print("I can help you with basic financial calculations like standard deviation, mean, or sum of numbers.\n")
    print("Examples: \n - 'Calculate the standard deviation of my portfolio with these returns: [0.05, -0.02, 0.03]'\n - 'What is the mean return for my portfolio with these values: [0.01, 0.02, -0.015]'\n - 'Sum these numbers: [10, 20, 30.5]'\n")

    while True:
        user_input = input("Enter your financial query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        print("\nAssistant (simulating LLM code generation)...\n")
        generated_code = simulate_llm_code_generation(user_input)
        print(f"Generated Code:\n{generated_code}\n")

        print("Assistant (executing code)...\n")
        execution_results = execute_python_code(generated_code)
        print(f"Execution Results:\n{execution_results}\n")

        print("Assistant (simulating LLM interpretation and advice)...\n")
        final_advice = simulate_llm_interpret_and_advise(execution_results, user_input)
        print(f"Final Advice:\n{final_advice}\n")

    print("Thank you for using the Financial Portfolio Optimization Assistant. Goodbye!")

if __name__ == "__main__":
    run_assistant()