import streamlit as st
import subprocess
import sys

def run_python_code(code: str) -> str:
    """Executes Python code in a subprocess and returns its stdout."""
    try:
        # Write the code to a temporary file
        with open("temp_script.py", "w") as f:
            f.write(code)
        
        # Execute the script using subprocess
        result = subprocess.run(
            [sys.executable, "temp_script.py"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error during code execution: {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def simulate_llm_with_pal(query: str) -> str:
    """Simulates an LLM that generates and executes code for calculations."""
    query_lower = query.lower()
    response = ""
    
    # Example: Simple interest calculation
    if "calculate simple interest" in query_lower:
        try:
            parts = query_lower.split()
            principal = float(parts[parts.index("principal") + 1]) if "principal" in parts else 1000
            rate = float(parts[parts.index("rate") + 1]) if "rate" in parts else 0.05
            time = float(parts[parts.index("time") + 1]) if "time" in parts else 1

            python_code = f"""
def calculate_simple_interest(principal, rate, time):
    return principal * rate * time

principal = {principal}
rate = {rate}
time = {time}
interest = calculate_simple_interest(principal, rate, time)
print(f"Simple Interest: {{interest:.2f}}")
"""
            st.info(f"LLM generated code for simple interest:\n```python\n{python_code}\n```")
            calculation_result = run_python_code(python_code)
            response = f"Based on your query, the simple interest is: {calculation_result}"
        except Exception as e:
            response = f"I need more details to calculate simple interest. Please provide principal, rate, and time. Error: {e}"
    
    # Example: Compound interest calculation
    elif "calculate compound interest" in query_lower:
        try:
            parts = query_lower.split()
            principal = float(parts[parts.index("principal") + 1]) if "principal" in parts else 1000
            rate = float(parts[parts.index("rate") + 1]) if "rate" in parts else 0.05
            time = float(parts[parts.index("time") + 1]) if "time" in parts else 1
            compounding_frequency = float(parts[parts.index("compounded") + 1]) if "compounded" in parts else 1 # e.g., 1 for annually

            python_code = f"""
def calculate_compound_interest(principal, rate, time, n):
    return principal * ((1 + rate/n)**(n*time)) - principal

principal = {principal}
rate = {rate}
time = {time}
compounding_frequency = {compounding_frequency}
interest = calculate_compound_interest(principal, rate, time, compounding_frequency)
print(f"Compound Interest: {{interest:.2f}}")
"""
            st.info(f"LLM generated code for compound interest:\n```python\n{python_code}\n```")
            calculation_result = run_python_code(python_code)
            response = f"Based on your query, the compound interest is: {calculation_result}"
        except Exception as e:
            response = f"I need more details to calculate compound interest. Please provide principal, rate, time, and compounding frequency. Error: {e}"

    # Generic response if no specific calculation is detected
    else:
        response = f"Hello! I'm your Financial Advisory Chatbot. I can help with various financial queries. For example, try asking 'calculate simple interest with principal 5000 rate 0.06 time 3' or 'calculate compound interest principal 1000 rate 0.05 time 2 compounded 4'."
    
    return response

# Streamlit UI
st.title("Financial Advisory Chatbot with PAL")
st.markdown("This chatbot uses Program-Aided Language (PAL) Prompting to perform precise financial calculations by generating and executing Python code.")

user_query = st.text_input("Ask a financial question:", "calculate simple interest with principal 5000 rate 0.06 time 3")

if st.button("Get Advice"):
    if user_query:
        with st.spinner("Processing your query..."):
            llm_response = simulate_llm_with_pal(user_query)
            st.write("**Chatbot's Response:**")
            st.write(llm_response)
    else:
        st.warning("Please enter a question.")
