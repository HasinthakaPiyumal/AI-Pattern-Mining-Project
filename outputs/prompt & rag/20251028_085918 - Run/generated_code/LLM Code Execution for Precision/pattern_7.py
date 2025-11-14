
import gradio as gr
import io
import contextlib

# Note: In a real application, you would import and configure an LLM client (e.g., openai)
# and handle API keys securely (e.g., using environment variables).
# For this demonstration, LLM calls are simulated to make the code runnable without actual API keys.

def financial_assistant(user_query: str) -> str:
    """
    Main function for the financial assistant, demonstrating the PAL pattern
    by simulating LLM interactions and code execution.
    """

    # --- 1. Simulate LLM generating Python code based on the user query ---
    # In a real application, this would be an actual LLM API call:
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": "You are a financial analyst assistant. Generate Python code to solve financial queries. Output only the Python code. Ensure imports are included and results are printed."},
    #         {"role": "user", "content": f"User query: {user_query}\nGenerate Python code to calculate..."}
    #     ],
    #     temperature=0.0
    # )
    # generated_code = response.choices[0].message.content

    generated_code = ""
    if "portfolio return" in user_query.lower():
        generated_code = """
import pandas as pd

# Simulate portfolio data for demonstration
data = {
    'asset': ['AAPL', 'GOOG', 'MSFT'],
    'weights': [0.4, 0.3, 0.3],
    'returns': [0.15, 0.10, 0.12] # Example returns
}
portfolio_df = pd.DataFrame(data)

# Calculate portfolio return
portfolio_return = (portfolio_df['weights'] * portfolio_df['returns']).sum()
print(f"Calculated Portfolio Return: {portfolio_return:.4f}")
"""
    elif "bond yield" in user_query.lower():
        generated_code = """
# Simulate bond parameters: Face Value, Coupon Rate, Current Price, Years to Maturity
face_value = 1000
coupon_rate = 0.05  # 5% annual coupon
current_price = 950
years_to_maturity = 10

# Simplified approximation for Yield to Maturity (YTM)
# For a more accurate YTM, numerical methods or financial libraries are typically used.
annual_coupon_payment = face_value * coupon_rate
ytm_approx = (annual_coupon_payment + (face_value - current_price) / years_to_maturity) / ((face_value + current_price) / 2)
print(f"Approximate Bond Yield to Maturity: {ytm_approx:.4f}")
"""
    elif "fibonacci" in user_query.lower(): # A non-financial example to show algorithmic execution
        generated_code = """
def fibonacci(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    print(f"Fibonacci sequence up to {n} terms: {result}")

print("Generating Fibonacci sequence up to 10 terms:")
fibonacci(10)
"""
    else:
        return "I can only process queries related to 'portfolio return', 'bond yield', or 'fibonacci' for this demonstration. Please try one of these."
    # --- End LLM Code Generation Simulation ---

    # --- 2. Execute the generated Python code safely ---
    captured_output = io.StringIO()
    code_execution_successful = True
    try:
        # Use contextlib.redirect_stdout to capture print statements from exec
        with contextlib.redirect_stdout(captured_output):
            # Execute the code in a restricted environment for safety
            # Note: A real-world application might use a more robust sandboxing solution
            exec(generated_code, {'__builtins__': None}, {})
        code_output = captured_output.getvalue().strip()
    except Exception as e:
        code_output = f"Error during code execution: {type(e).__name__}: {e}"
        code_execution_successful = False

    # --- 3. Simulate LLM synthesizing the result into a natural language explanation ---
    # In a real application, this would be another LLM API call:
    # final_response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful financial analyst assistant. Explain the financial results clearly and concisely."},`
    #         {"role": "user", "content": f"User query: {user_query}\n\nPython code executed:\n```python\n{generated_code}\n```\n\nExecution Output:\n```\n{code_output}\n```\n\nBased on this, provide a concise financial analysis and answer the user's original query."}
    #     ],
    #     temperature=0.7
    # )
    # final_explanation = final_response.choices[0].message.content

    final_explanation_intro = f"Based on your query: \"{user_query}\"\n\n"
    final_explanation_code_info = f"The AI assistant generated and executed the following Python code to address your request:\n```python\n{generated_code}\n```\n\n"
    final_explanation_output_info = f"The execution yielded the following result:\n```\n{code_output}\n```\n\n"

    # Provide context-specific explanations based on the simulated output
    if code_execution_successful:
        if "Calculated Portfolio Return:" in code_output:
            final_explanation_summary = "This calculation represents the weighted average return of your portfolio, offering a clear indicator of its overall performance based on the simulated asset weights and individual returns. This is a fundamental metric for portfolio analysis."
        elif "Approximate Bond Yield to Maturity:" in code_output:
            final_explanation_summary = "The approximate Bond Yield to Maturity (YTM) indicates the total estimated return an investor can expect if they hold the bond until maturity. This value is crucial for assessing bond investment attractiveness, especially when comparing against other fixed-income securities."
        elif "Fibonacci sequence" in code_output:
            final_explanation_summary = "The Fibonacci sequence is a classic mathematical series where each number is the sum of the two preceding ones. This demonstrates the system's capability to generate and execute code for general algorithmic tasks beyond financial specific computations, showcasing the versatility of the PAL pattern."
        else:
            final_explanation_summary = "The system successfully processed your request using program-aided computation. Please review the detailed code and its output above for the specific results."
    else:
        final_explanation_summary = "An error occurred during the execution of the generated code. Please review the error message above for details. This might indicate an issue with the query or the generated code's syntax/logic."

    return final_explanation_intro + final_explanation_code_info + final_explanation_output_info + final_explanation_summary

# --- Gradio Interface Setup ---
iface = gr.Interface(
    fn=financial_assistant,
    inputs=gr.Textbox(lines=3, placeholder="Ask a financial question, e.g., 'Calculate my portfolio return', 'What is the approximate bond yield?', or 'Generate fibonacci sequence up to 10 terms'"),
    outputs=gr.Textbox(label="Financial Analysis Result", lines=15),
    title="AI-Powered Financial Analyst Assistant (PAL Pattern Demo)",
    description="This assistant leverages the Program-Aided Language Models (PAL) Prompting pattern. It interprets your natural language query, generates Python code to perform precise calculations (like financial analysis), executes that code, and then provides a natural language explanation of the results."
)

# To run the Gradio app, uncomment the following line and execute the script:
# if __name__ == "__main__":
#    iface.launch()
