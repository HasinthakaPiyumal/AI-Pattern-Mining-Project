import gradio as gr
import re

# --- Financial Calculation Module (Example functions) ---
def calculate_future_value(principal, annual_contribution, years, annual_rate):
    future_value = principal
    for _ in range(years):
        future_value = future_value * (1 + annual_rate)
        future_value += annual_contribution
    return future_value

def calculate_compound_interest(principal, annual_rate, years, compound_per_year=1):
    return principal * (1 + annual_rate / compound_per_year)**(compound_per_year * years)

# --- Mock LLM and Code Execution Environment ---
class MockLLM:
    def __init__(self):
        pass

    def generate_response(self, prompt):
        # Simulate LLM understanding and code generation for specific financial queries
        if "future value" in prompt.lower() and "annual contribution" in prompt.lower():
            # Example: "Calculate the future value of my retirement savings with an annual contribution of $5000 for 20 years at a 7% annual return, and then compare it to a scenario with a 9% return"
            match = re.search(r"annual contribution of \$(\d+) for (\d+) years at a (\d+)% annual return.*compare it to a scenario with a (\d+)% return", prompt.lower())
            if match:
                contribution = int(match.group(1))
                years = int(match.group(2))
                rate1 = float(match.group(3)) / 100
                rate2 = float(match.group(4)) / 100

                code = f"""
results = []
value1 = calculate_future_value(0, {contribution}, {years}, {rate1})
results.append(f"Future value at {rate1*100}% return: ${{value1:,.2f}}")
value2 = calculate_future_value(0, {contribution}, {years}, {rate2})
results.append(f"Future value at {rate2*100}% return: ${{value2:,.2f}}")
output_result = "\n".join(results)
"""
                return {"action": "execute_code", "code": code, "pre_text": "Let me calculate that for you...", "post_text": "Based on these calculations, I recommend reviewing your investment strategy.", "type": "future_value_comparison"}
        elif "compound interest" in prompt.lower():
            match = re.search(r"compound interest for a principal of \$(\d+) at (\d+)% over (\d+) years", prompt.lower())
            if match:
                principal = int(match.group(1))
                rate = float(match.group(2)) / 100
                years = int(match.group(3))
                code = f"""
result = calculate_compound_interest({principal}, {rate}, {years})
output_result = f"The compound interest for your investment is: ${{result:,.2f}}"
"""
                return {"action": "execute_code", "code": code, "pre_text": "Calculating compound interest...", "post_text": "Consider these figures for your financial planning.", "type": "compound_interest"}

        return {"action": "respond_directly", "response": "I can help with general financial advice. For complex calculations, please try to be specific about the values you want to use."}

def execute_python_code(code):
    # A simplified and UNSAFE code execution environment for demonstration.
    # In a production environment, this requires strict sandboxing and security measures.
    local_vars = {
        "calculate_future_value": calculate_future_value,
        "calculate_compound_interest": calculate_compound_interest,
        "output_result": ""
    }
    try:
        exec(code, {"__builtins__": {}}, local_vars)
        return local_vars.get("output_result", "No specific output_result variable found in executed code.")
    except Exception as e:
        return f"Error during code execution: {e}"

mock_llm = MockLLM()

# --- Gradio Interface Logic ---
def smart_financial_advisor(user_query):
    llm_output = mock_llm.generate_response(user_query)

    if llm_output["action"] == "execute_code":
        pre_text = llm_output.get("pre_text", "")
        code_result = execute_python_code(llm_output["code"])
        post_text = llm_output.get("post_text", "")

        final_response = f"{pre_text}\n\n{code_result}\n\n{post_text}"
    else:
        final_response = llm_output["response"]

    return final_response

# --- Gradio UI ---
iface = gr.Interface(
    fn=smart_financial_advisor,
    inputs=gr.Textbox(lines=5, label="Ask your financial question:", placeholder="e.g., Calculate the future value of my retirement savings with an annual contribution of $5000 for 20 years at a 7% annual return, and then compare it to a scenario with a 9% return"),
    outputs=gr.Textbox(label="Smart Financial Advisor Response:"),
    title="Smart Financial Advisor (PAL Prompting Demo)",
    description="This AI advisor leverages Program-Aided Language Models (PAL) to perform precise financial calculations and provide tailored advice. Try asking for future value or compound interest calculations."
)

if __name__ == "__main__":
    iface.launch()