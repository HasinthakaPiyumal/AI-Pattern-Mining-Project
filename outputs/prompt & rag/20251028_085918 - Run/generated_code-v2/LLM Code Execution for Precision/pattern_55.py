import os
import sys
from io import StringIO

from llm_code_generator import generate_code
from code_executor import execute_code
from llm_result_summarizer import summarize_results

def main():
    print("Intelligent Financial Data Analyst - PAL Prompting Example\n")

    user_query = "Calculate the 3-year compound annual growth rate (CAGR) for Apple's (AAPL) revenue."
    print(f"User Query: {user_query}\n")

    print("1. LLM as a Code Generator: Generating Python code...")
    generated_code = generate_code(user_query)
    print("Generated Code:\n---\n" + generated_code + "\n---\n")

    print("2. Code Execution Engine: Executing generated code...")
    # Capture stdout during code execution
    old_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output
    
    execution_result, execution_error = execute_code(generated_code)
    
    sys.stdout = old_stdout # Restore stdout
    captured_output = redirected_output.getvalue()

    if execution_error:
        print(f"Code Execution Error: {execution_error}")
        return
    
    print(f"Code Execution Output (from print statements):\n---\n{captured_output}--- ")
    print(f"Code Execution Raw Result: {execution_result}\n")

    print("3. LLM as a Natural Language Reasoner/Summarizer: Summarizing results...")
    final_summary = summarize_results(user_query, execution_result, captured_output)
    print("Final Summary:\n---\n" + final_summary + "\n---\n")

if __name__ == "__main__":
    main()