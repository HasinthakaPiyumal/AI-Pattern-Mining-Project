import streamlit as st
import pandas as pd
import io
import sys
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_python_code(query: str, df_head: str, df_columns: list) -> str:
    prompt = f"""You are a financial analysis assistant.
The user will provide a query about financial data and the head and columns of a pandas DataFrame named `df` containing financial information.
Your task is to generate Python code using pandas to perform the requested analysis.
The code should only output the final result using a print statement. Do not include any explanations or extra text in your code.
Make sure the code is executable and directly addresses the user's query using the provided DataFrame `df`.
Assume the DataFrame `df` is already loaded and available in the execution environment.

DataFrame head:
{df_head}

DataFrame columns:
{df_columns}

User query: "{query}"

Generated Python code:
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates Python code for financial analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

def generate_explanation(query: str, generated_code: str, execution_output: str) -> str:
    prompt = f"""You are a financial analyst specializing in explaining complex financial calculations.
The user asked a query, and Python code was generated and executed to get a numerical result.
Your task is to explain the results of the financial analysis in natural language, provide insights, and offer recommendations.

User Query: "{query}"
Generated Python Code:
```python
{generated_code}
```
Execution Output:
{execution_output}

Based on the above, provide a comprehensive explanation, interpretation, and recommendations:
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst providing explanations and recommendations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

def execute_code(code: str, dataframe: pd.DataFrame) -> str:
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        exec_globals = {'df': dataframe, 'pd': pd}
        exec(code, exec_globals)
        output = redirected_output.getvalue()
    except Exception as e:
        output = f"Error during code execution: {e}"
    finally:
        sys.stdout = old_stdout
    return output

st.set_page_config(layout="wide", page_title="Financial Analysis and Forecasting Tool")

st.title("Financial Statement Analysis and Forecasting Tool")

uploaded_file = st.file_uploader("Upload your financial data (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        st.write("---")

        analyst_query = st.text_area("Enter your financial analysis query:")

        if st.button("Analyze"):
            if analyst_query:
                with st.spinner("Generating Python code..."):
                    df_head_str = df.head().to_string()
                    df_columns_list = df.columns.tolist()
                    generated_code = generate_python_code(analyst_query, df_head_str, df_columns_list)
                    st.subheader("Generated Python Code")
                    st.code(generated_code, language="python")

                with st.spinner("Executing generated code..."):
                    execution_output = execute_code(generated_code, df)
                    st.subheader("Code Execution Output")
                    st.code(execution_output)

                if "Error during code execution" not in execution_output:
                    with st.spinner("Generating explanation and recommendations..."):
                        llm_explanation = generate_explanation(analyst_query, generated_code, execution_output)
                        st.subheader("LLM Explanation and Recommendations")
                        st.write(llm_explanation)
                else:
                    st.error("Code execution failed. Cannot generate explanation.")
            else:
                st.warning("Please enter a query to analyze.")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}. Please ensure it's a valid CSV.")