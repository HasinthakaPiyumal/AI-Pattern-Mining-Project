import gradio as gr
import subprocess
import re

def simulate_llm_code_generation(query):
    query_lower = query.lower()
    if "future value" in query_lower and "compounded" in query_lower:
        match = re.search(r"future value of \$(\d+,?\d*\.?\d*) at (\d+\.?\d*)% annual interest compounded for (\d+) years", query_lower)
        if match:
            principal = float(match.group(1).replace(',', ''))
            rate = float(match.group(2)) / 100
            years = int(match.group(3))
            return f"future_value = {principal} * (1 + {rate})**{years}; print(f'Calculated Future Value: {{future_value:.2f}}')"
    elif "simple interest" in query_lower:
        match = re.search(r"simple interest on \$(\d+,?\d*\.?\d*) for (\d+) years at (\d+\.?\d*)%", query_lower)
        if match:
            principal = float(match.group(1).replace(',', ''))
            years = int(match.group(2))
            rate = float(match.group(3)) / 100
            return f"simple_interest = {principal} * {rate} * {years}; print(f'Calculated Simple Interest: {{simple_interest:.2f}}')"
    return "print('Error: Could not generate code for this query. Please try a different financial calculation.')"

def execute_python_code(code_string):
    try:
        process = subprocess.run(['python', '-c', code_string], capture_output=True, text=True, check=True, timeout=5)
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Code Execution Error: {e.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Code Execution Error: Timeout"
    except Exception as e:
        return f"An unexpected error occurred during code execution: {str(e)}"

def integrate_results_and_nlg(original_query, code_output):
    if "Calculated Future Value:" in code_output:
        value_match = re.search(r"Calculated Future Value: (\d+\.?\d*)", code_output)
        if value_match:
            future_value = float(value_match.group(1))
            return f"Based on your request, the calculated future value is approximately ${future_value:.2f}."
    elif "Calculated Simple Interest:" in code_output:
        value_match = re.search(r"Calculated Simple Interest: (\d+\.?\d*)", code_output)
        if value_match:
            simple_interest = float(value_match.group(1))
            return f"For your query, the simple interest calculated is approximately ${simple_interest:.2f}."
    elif "Error:" in code_output or "Code Execution Error:" in code_output:
        return f"I encountered an issue processing your request: {code_output}. Please ensure your query is clear and specific for financial calculations."
    return f"I have processed your request, but I couldn't fully interpret the result into a natural language explanation. Here's the raw output: {code_output}"

def financial_advisor_ai(user_query):
    generated_code = simulate_llm_code_generation(user_query)
    if "Error:" in generated_code: # Check for simulation errors directly from code generation
        return generated_code.replace("print('", "").replace("')", "")
    
    executed_output = execute_python_code(generated_code)
    final_response = integrate_results_and_nlg(user_query, executed_output)
    return final_response

if __name__ == "__main__":
    interface = gr.Interface(
        fn=financial_advisor_ai,
        inputs=gr.Textbox(lines=2, placeholder="Ask me a financial question, e.g., 'Calculate the future value of $10000 at 5% annual interest compounded for 10 years.'"),
        outputs="text",
        title="Personal Financial Advisor AI (PAL Prompting Demo)",
        description="This AI uses Program-Aided Language Models (PAL) to generate and execute Python code for precise financial calculations, then provides a natural language explanation."
    )
    interface.launch()
