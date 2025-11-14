import streamlit as st
import openai
import yfinance as yf
import pandas as pd
import sys
import io
import re

# --- OpenAI LLM Interaction Functions ---

def generate_code_with_llm(user_query: str, openai_api_key: str) -> str:
    """
    Instructs the LLM to generate Python code for financial calculations.
    """
    client = openai.OpenAI(api_key=openai_api_key)

    prompt = f"""
    You are a financial Python code generator. Given the user's request, generate a Python script that calculates the requested financial metric using 'yfinance' and 'pandas'. Print only the final numerical result. Do not include any explanations or extra text. If data retrieval fails or the calculation is not possible, print 'Error: Could not retrieve data or perform calculation.'.

    Here's an example for CAGR for AAPL over 5 years:

    import yfinance as yf
    import pandas as pd

    ticker = 'AAPL'
    years = 5

    data = yf.download(ticker, period=f'{{years}}y', progress=False)
    if not data.empty:
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        # Ensure we have at least 2 data points for calculation
        if len(data) > 1:
            # Adjust years if actual data period is shorter/longer than requested
            actual_days = (data.index[-1] - data.index[0]).days
            actual_years = actual_days / 365.25 if actual_days > 0 else years # Fallback if only one day or issue
            if actual_years == 0: actual_years = years # Prevent division by zero if only one day

            cagr = (end_price / start_price)**(1/actual_years) - 1
            print(cagr)
        else:
            print('Error: Insufficient data for calculation.')
    else:
        print('Error: Could not retrieve data.')

    User request: {user_query}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # gpt-4 or gpt-4o would be more reliable for code generation
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates Python code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        code = response.choices[0].message.content.strip()
        # Basic sanitization: sometimes LLM adds markdown wrappers
        if code.startswith("```python") and code.endswith("```"):
            code = code[len("```python\n"):-len("\n```")].strip()
        return code
    except openai.APIError as e:
        st.error(f"OpenAI API error during code generation: {e}")
        return "print('Error: OpenAI API failed to generate code.')"
    except Exception as e:
        st.error(f"An unexpected error occurred during code generation: {e}")
        return "print('Error: Unexpected error during code generation.')"

def summarize_result_with_llm(original_query: str, numerical_result: str, openai_api_key: str) -> str:
    """
    Instructs the LLM to summarize the numerical result in natural language.
    """
    client = openai.OpenAI(api_key=openai_api_key)

    prompt = f"""
    You are a financial assistant. A user asked: '{original_query}'. The calculation resulted in: '{numerical_result}'. Explain this result clearly and concisely to the user in natural language. If the result indicates an error (e.g., 'Error: ...'), simply state that the calculation could not be performed and provide a brief reason based on the error message.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains financial results."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except openai.APIError as e:
        st.error(f"OpenAI API error during result summarization: {e}")
        return "Error: Could not summarize the result due to an API issue."
    except Exception as e:
        st.error(f"An unexpected error occurred during result summarization: {e}")
        return "Error: Unexpected error during result summarization."

# --- Code Execution Module ---

def execute_python_code(code_string: str) -> str:
    """
    Executes the given Python code string and captures its stdout.
    Returns the captured output as a string.
    """
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    try:
        # Using exec in a limited global/local scope for demonstration.
        # In a production environment, a more secure sandboxing mechanism
        # (e.g., a separate process with strict resource limits or a container)
        # should be used to prevent malicious code execution.
        exec_globals = {'yfinance': yf, 'pandas': pd, 'pd': pd, 'yf': yf}
        exec(code_string, exec_globals, exec_globals)
        output = redirected_output.getvalue().strip()
    except Exception as e:
        output = f"Execution Error: {e}"
    finally:
        sys.stdout = old_stdout  # Restore original stdout
    return output

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Financial PAL Assistant", layout="centered")
    st.title("📈 Financial Analysis Assistant (PAL Prompting)")
    st.markdown(
        "This assistant uses a Large Language Model (LLM) to generate and execute Python code for precise financial calculations, "
        "then summarizes the results in natural language. Powered by Program-Aided Language Models (PAL) prompting."
    )

    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key to proceed.")
        return

    user_query = st.text_input("Enter your financial query (e.g., 'What is the 5-year CAGR for Microsoft stock?')")

    if st.button("Analyze"):
        if not user_query:
            st.error("Please enter a financial query.")
            return

        st.divider()
        st.subheader("Processing your request...")

        # Phase 1: LLM generates Python code
        with st.spinner("Generating Python code for calculation..."):
            generated_code = generate_code_with_llm(user_query, openai_api_key)
            st.success("Code generation complete!")
            st.markdown("##### Generated Python Code:")
            st.code(generated_code, language="python")

        # Phase 2: Execute the generated code
        numerical_result = ""
        with st.spinner("Executing the generated Python code..."):
            execution_output = execute_python_code(generated_code)
            st.success("Code execution complete!")
            st.markdown("##### Code Execution Output:")
            st.text(execution_output)
            numerical_result = execution_output # Capture for summarization

        # Phase 3: LLM summarizes the numerical result
        with st.spinner("Summarizing the financial results..."):
            final_answer = summarize_result_with_llm(user_query, numerical_result, openai_api_key)
            st.success("Summary complete!")
            st.markdown("##### Financial Analysis Result:")
            st.markdown(final_answer)

if __name__ == "__main__":
    main()
