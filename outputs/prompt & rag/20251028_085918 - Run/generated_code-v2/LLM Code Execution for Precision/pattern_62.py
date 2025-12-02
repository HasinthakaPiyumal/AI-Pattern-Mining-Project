import io
import sys
import pandas as pd
import numpy as np

def prompt_generator(query: str) -> str:
    """Generates a prompt for the LLM based on the user's query."""
    prompt = f"""Generate Python code to answer the following financial question. The code should print the final result. You can use pandas and numpy if needed.

Financial Question: {query}

Python Code:
"""
    return prompt

def mock_llm_code_generator(prompt: str) -> str:
    """Simulates an LLM generating Python code based on a prompt."""
    if "compound interest" in prompt.lower() or "interest rate" in prompt.lower():
        return """principal = 1000
rate = 0.05
time = 10
amount = principal * (1 + rate)**time
compound_interest = amount - principal
print(f"Compound Interest: {compound_interest:.2f}")
"""
    elif "stock portfolio" in prompt.lower() or "returns" in prompt.lower():
        return """# Assuming a simple scenario with two stocks
stock_data = {
    'Stock_A': [100, 105, 110, 115, 120],
    'Stock_B': [50, 52, 55, 58, 60]
}
df = pd.DataFrame(stock_data)

# Calculate daily returns
returns = df.pct_change().dropna()

# Assuming equal weighting for simplicity
portfolio_returns = returns.mean(axis=1)

print(f"Average daily portfolio returns:\n{portfolio_returns.mean():.4f}")
"""
    elif "present value" in prompt.lower() or "discount" in prompt.lower():
        return """future_value = 10000
discount_rate = 0.07
periods = 5
present_value = future_value / (1 + discount_rate)**periods
print(f"Present Value: {present_value:.2f}")
"""
    else:
        return """print("I need more specific financial details to generate relevant code.")
"""

def code_execution_engine(code: str) -> str:
    """Executes the given Python code in a controlled environment and captures output."""
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        # A very basic sandbox. For production, a more robust sandboxing solution is required.
        exec(code, {'pd': pd, 'np': np}, {})
        output = redirected_output.getvalue()
    except Exception as e:
        output = f"Error during code execution: {e}"
    finally:
        sys.stdout = old_stdout  # Restore stdout
    return output

def result_synthesizer(execution_output: str) -> str:
    """Synthesizes a human-readable explanation from the code execution output."""
    if "Error during code execution" in execution_output:
        return f"I encountered an error while processing your request:\n{execution_output}"
    elif "I need more specific financial details" in execution_output:
        return "I couldn't generate specific financial code for your query. Please provide more details."
    else:
        return f"Here are the results of your financial analysis:\n{execution_output}"

def main():
    print("\nAI-powered Financial Analyst Assistant (PAL Prompting Demo)")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nEnter your financial query: ")
        if user_query.lower() == 'exit':
            break

        # 1. Prompt Generation
        llm_prompt = prompt_generator(user_query)
        print(f"\n[DEBUG] LLM Prompt:\n{llm_prompt.strip()}\n")

        # 2. LLM Integration (Mock)
        generated_code = mock_llm_code_generator(llm_prompt)
        print(f"[DEBUG] Generated Code:\n{generated_code.strip()}\n")

        # 3. Code Execution
        execution_result = code_execution_engine(generated_code)
        print(f"[DEBUG] Code Execution Output:\n{execution_result.strip()}\n")

        # 4. Result Synthesizer
        final_answer = result_synthesizer(execution_result)
        print(f"\n{'-'*50}\nFinal Answer:\n{final_answer.strip()}\n{'-'*50}")

if __name__ == "__main__":
    main()