import streamlit as st
import subprocess
import re

def simulate_llm_code_generation(query: str) -> dict:
    """Simulates LLM generating code and a response with a placeholder."""
    query_lower = query.lower()
    if "future value" in query_lower and "investment" in query_lower and "interest" in query_lower:
        # Example: "Calculate the future value of an investment of $1000 at 5% annual interest for 10 years, compounded annually."
        match_principal = re.search(r'\$(\d+,?\d*\.?\d*)', query_lower)
        match_rate = re.search(r'(\d+\.?\d*)\%', query_lower)
        match_time = re.search(r'for (\d+) years', query_lower)

        principal = float(match_principal.group(1).replace(',', '')) if match_principal else 1000.0
        rate = float(match_rate.group(1)) / 100 if match_rate else 0.05
        time = int(match_time.group(1)) if match_time else 10

        python_code = f"""
principal = {principal}
rate = {rate}
time = {time}
future_value = principal * (1 + rate)**time
print(f"{{future_value:.2f}}")
"""
        llm_response_template = "Based on your input, the future value of your investment is approximately {result}."
        return {"code": python_code, "response_template": llm_response_template}
    elif "compound interest" in query_lower:
        # Example: "Calculate the compound interest for $5000 at 4% for 7 years."
        match_principal = re.search(r'\$(\d+,?\d*\.?\d*)', query_lower)
        match_rate = re.search(r'(\d+\.?\d*)\%', query_lower)
        match_time = re.search(r'for (\d+) years', query_lower)

        principal = float(match_principal.group(1).replace(',', '')) if match_principal else 5000.0
        rate = float(match_rate.group(1)) / 100 if match_rate else 0.04
        time = int(match_time.group(1)) if match_time else 7

        python_code = f"""
principal = {principal}
rate = {rate}
time = {time}
compound_interest = principal * ((1 + rate)**time - 1)
print(f"{{compound_interest:.2f}}")
"""
        llm_response_template = "The calculated compound interest is {result}."
        return {"code": python_code, "response_template": llm_response_template}
    else:
        return {"code": "", "response_template": "I can provide financial calculations. Please ask about future value or compound interest."}

def execute_code_in_sandbox(code: str) -> str:
    """Executes Python code in a sandboxed environment using subprocess."""
    if not code.strip():
        return ""
    try:
        # Using a separate Python interpreter process for basic sandboxing.
        # In a real application, consider more robust sandboxing (e.g., Docker, a dedicated microservice).
        process = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=5  # Timeout to prevent infinite loops
        )
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error during code execution: {e.stderr}"
    except subprocess.TimeoutExpired:
        return "Code execution timed out."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def process_financial_query(query: str) -> str:
    """Orchestrates the LLM interaction and code execution for a financial query."""
    llm_output = simulate_llm_code_generation(query)
    generated_code = llm_output["code"]
    response_template = llm_output["response_template"]

    if generated_code:
        code_output = execute_code_in_sandbox(generated_code)
        if "Error" in code_output or "timed out" in code_output:
            final_response = f"I encountered an issue with the calculation: {code_output}"
        else:
            final_response = response_template.format(result=code_output)
    else:
        final_response = response_template # If no code was generated, just use the template
    
    return final_response

# Streamlit Frontend
st.title("Financial Advisory & Portfolio Optimization Tool (PAL Prompting Demo)")
st.write("Ask me a financial question, and I'll use program-aided reasoning to provide a precise answer.")

user_query = st.text_area("Enter your financial question:", "Calculate the future value of an investment of $10000 with an annual interest rate of 5% over 10 years.")

if st.button("Get Advice"):
    if user_query:
        with st.spinner("Thinking and calculating..."):
            response = process_financial_query(user_query)
            st.success("Here is your advice:")
            st.write(response)
    else:
        st.warning("Please enter a query.")

st.markdown("""
---
**How it works (PAL Prompting):**
1.  You ask a financial question.
2.  A simulated LLM determines if computation is needed.
3.  If so, it generates Python code for the calculation.
4.  The code is executed securely.
5.  The numerical result is integrated back into a natural language response.
""")
