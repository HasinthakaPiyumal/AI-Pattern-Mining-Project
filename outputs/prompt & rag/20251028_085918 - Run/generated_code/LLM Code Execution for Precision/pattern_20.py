import streamlit as st
import openai
import pandas as pd
import numpy as np
import io
import contextlib
import os

# Set your OpenAI API key
# It's recommended to set this as an environment variable or use Streamlit secrets
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Placeholder for OpenAI API key if not using environment variable or Streamlit secrets
# In a real application, use st.secrets["OPENAI_API_KEY"] or os.environ.get("OPENAI_API_KEY")
# For demonstration, we'll allow direct input or a placeholder if not found.

def execute_code_safely(code_string: str, global_vars: dict = None, local_vars: dict = None) -> (str, str):
    """
    Safely executes a given Python code string and captures its output.
    Returns a tuple of (stdout_output, error_output).
    """
    if global_vars is None:
        global_vars = {"pd": pd, "np": np, "st": st} # Provide common libraries
    if local_vars is None:
        local_vars = {}

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    error_message = ""

    with contextlib.redirect_stdout(stdout_buffer):
        with contextlib.redirect_stderr(stderr_buffer):
            try:
                exec(code_string, global_vars, local_vars)
            except Exception as e:
                error_message = str(e)
    
    return stdout_buffer.getvalue(), stderr_buffer.getvalue() + error_message

def call_llm(prompt: str, model: str = "gpt-4", api_key: str = None) -> str:
    """
    Calls the OpenAI LLM with the given prompt.
    """
    if not api_key:
        return "Error: OpenAI API key is not set. Please enter your API key."
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant that can generate and interpret Python code for financial analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except openai.AuthenticationError:
        return "Error: Invalid OpenAI API key. Please check your key."
    except openai.APITimeoutError:
        return "Error: OpenAI API request timed out."
    except openai.APIConnectionError as e:
        return f"Error: Could not connect to OpenAI API: {e}"
    except openai.RateLimitError:
        return "Error: OpenAI API rate limit exceeded. Please wait and try again."
    except Exception as e:
        return f"An unexpected error occurred with the LLM call: {e}"

def generate_financial_code(financial_task: str, user_inputs: dict, api_key: str) -> str:
    """
    Prompts the LLM to generate Python code for a given financial task.
    """
    prompt = f"""Generate a Python code snippet to perform the following financial analysis:
    {financial_task}

    Here are the user inputs: {user_inputs}

    The code should use pandas and numpy for calculations if suitable. 
    Print the final numerical result(s) of the calculation clearly.
    Do not include any explanations or extra text, just the pure Python code block.
    Example for Discounted Cash Flow (DCF):
    ```python
    # Example inputs
    initial_investment = {user_inputs.get("initial_investment", 100000)}
    cash_flows = {user_inputs.get("cash_flows", [20000, 25000, 30000, 35000, 40000])}
    discount_rate = {user_inputs.get("discount_rate", 0.10)}

    # DCF Calculation
    npv = 0
    for i, cash_flow in enumerate(cash_flows):
        npv += cash_flow / (1 + discount_rate)**(i + 1)
    
    dcf = npv - initial_investment
    print(f"Discounted Cash Flow (DCF): {dcf:.2f}")
    ```
    Ensure the code is self-contained and runnable.
    Wrap the code in ```python ``` block.
    """
    return call_llm(prompt, api_key=api_key)

def interpret_results(financial_task: str, generated_code: str, execution_output: str, error_output: str, api_key: str) -> str:
    """
    Prompts the LLM to interpret the execution results and provide recommendations.
    """
    prompt = f"""I performed a financial analysis task: {financial_task}

    Here is the Python code that was generated and executed:
    ```python
    {generated_code}
    ```

    Here is the output from executing the code:
    {execution_output}

    Here is any error output from execution:
    {error_output}

    Please interpret these results. Explain what the numbers mean for an investment decision and provide clear recommendations.
    If there were errors, explain what went wrong and how it impacts the analysis.
    """
    return call_llm(prompt, api_key=api_key)

st.set_page_config(layout="wide", page_title="AI Financial Advisor (CAR Pattern)")
st.title("📈 AI Financial Advisor (Code-Assisted Reasoning)")
st.markdown("This tool leverages an LLM to generate and execute Python code for financial calculations, providing precise results and interpretations.")

# Sidebar for API Key and Model Selection
st.sidebar.header("Configuration")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.sidebar.warning("Please enter your OpenAI API key to use the application.")

financial_model = st.sidebar.selectbox(
    "Select LLM Model",
    ("gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"),
    index=0,
    help="Choose the OpenAI model for code generation and interpretation."
)


st.header("1. Define Your Financial Analysis Task")
financial_task = st.text_area(
    "Describe the financial analysis you want to perform (e.g., 'Calculate the Net Present Value (NPV) of a project', 'Determine the portfolio risk for given assets', 'Calculate Black-Scholes option price').",
    "Calculate the Discounted Cash Flow (DCF) for an investment project.",
    height=100
)

st.header("2. Provide Financial Parameters")
# Dynamic input fields based on common financial tasks or a generic JSON input
user_inputs_raw = st.text_area(
    "Enter relevant financial parameters as a JSON dictionary (e.g., initial_investment, cash_flows, discount_rate, stock_ticker, etc.)",
    """
{
    "initial_investment": 100000,
    "cash_flows": [20000, 25000, 30000, 35000, 40000],
    "discount_rate": 0.10,
    "years": 5
}
""",
    height=200
)

user_inputs = {}
try:
    user_inputs = json.loads(user_inputs_raw)
except json.JSONDecodeError:
    st.error("Invalid JSON input for financial parameters. Please ensure it's a valid JSON dictionary.")

if st.button("Run Financial Analysis"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not financial_task:
        st.error("Please describe the financial analysis task.")
    elif not user_inputs:
        st.error("Please provide valid financial parameters.")
    else:
        with st.spinner("Step 1: Generating Python code..."):
            generated_code_raw = generate_financial_code(financial_task, user_inputs, openai_api_key)
            
            # Extract code block from LLM response
            if "```python" in generated_code_raw and "```" in generated_code_raw:
                start_index = generated_code_raw.find("```python") + len("```python")
                end_index = generated_code_raw.rfind("```")
                generated_code = generated_code_raw[start_index:end_index].strip()
            else:
                generated_code = generated_code_raw.strip() # Assume it's just code if no markdown block
                st.warning("LLM did not wrap code in ```python block. Attempting to execute raw response.")

            st.subheader("3. Generated Code")
            st.code(generated_code, language="python")

        with st.spinner("Step 2: Executing generated code..."):
            if "Error:" in generated_code_raw or "Error:" in generated_code: # Check if LLM returned an error directly
                st.error(f"LLM failed to generate valid code: {generated_code_raw}")
                execution_output = ""
                error_output = generated_code_raw # Treat LLM error as execution error
            else:
                execution_output, error_output = execute_code_safely(generated_code)
            
            st.subheader("4. Code Execution Output")
            if execution_output:
                st.text("Standard Output:")
                st.code(execution_output, language="text")
            if error_output:
                st.text("Error Output:")
                st.error(error_output)

        with st.spinner("Step 3: Interpreting results and providing recommendations..."):
            interpretation = interpret_results(financial_task, generated_code, execution_output, error_output, openai_api_key)
            st.subheader("5. AI Financial Interpretation & Recommendation")
            st.write(interpretation)

# Optional: Add a section for predefined examples
st.sidebar.markdown("--- ")
st.sidebar.subheader("Examples")
if st.sidebar.button("Example: DCF Calculation"):
    st.session_state.financial_task = "Calculate the Discounted Cash Flow (DCF) for an investment project."
    st.session_state.user_inputs_raw = """
{
    "initial_investment": 150000,
    "cash_flows": [30000, 35000, 40000, 45000, 50000],
    "discount_rate": 0.12,
    "years": 5
}
"""
    st.rerun()

if st.sidebar.button("Example: Simple Interest"):
    st.session_state.financial_task = "Calculate the simple interest for a loan."
    st.session_state.user_inputs_raw = """
{
    "principal": 10000,
    "rate": 0.05,
    "time_years": 3
}
"""
    st.rerun()

# Pre-fill inputs if examples are clicked
if "financial_task" in st.session_state:
    st.session_state["financial_task_key"] = st.session_state.financial_task
    del st.session_state.financial_task
if "user_inputs_raw" in st.session_state:
    st.session_state["user_inputs_raw_key"] = st.session_state.user_inputs_raw
    del st.session_state.user_inputs_raw

# Add keys to widgets for proper state management with reruns
financial_task = st.text_area(
    "Describe the financial analysis you want to perform (e.g., 'Calculate the Net Present Value (NPV) of a project', 'Determine the portfolio risk for given assets', 'Calculate Black-Scholes option price').",
    value=st.session_state.get("financial_task_key", "Calculate the Discounted Cash Flow (DCF) for an investment project."),
    height=100,
    key="financial_task_key"
)

user_inputs_raw = st.text_area(
    "Enter relevant financial parameters as a JSON dictionary (e.g., initial_investment, cash_flows, discount_rate, stock_ticker, etc.)",
    value=st.session_state.get("user_inputs_raw_key", """
{
    "initial_investment": 100000,
    "cash_flows": [20000, 25000, 30000, 35000, 40000],
    "discount_rate": 0.10,
    "years": 5
}
"""),
    height=200,
    key="user_inputs_raw_key"
)
