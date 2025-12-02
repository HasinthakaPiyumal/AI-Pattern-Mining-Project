import streamlit as st
import pandas as pd
import io
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None
    return df

def execute_financial_code(code_string, dataframe):
    local_vars = {"df": dataframe, "pd": pd}
    output_capture = io.StringIO()
    try:
        import sys
        old_stdout = sys.stdout
        sys.stdout = output_capture
        exec(code_string, {"__builtins__": None, "pd": pd}, local_vars)
        sys.stdout = old_stdout
        result = output_capture.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        result = f"Error during code execution: {e}"
    return result

st.title("Financial Report Generator and Analyzer (PAL Prompting)")

if "dataframe" not in st.session_state:
    st.session_state.dataframe = None

uploaded_file = st.file_uploader("Upload your financial data (CSV or Excel)", type=["csv", "xls", "xlsx"])

if uploaded_file is not None and st.session_state.dataframe is None:
    st.session_state.dataframe = load_data(uploaded_file)
    if st.session_state.dataframe is not None:
        st.success("Financial data loaded successfully!")
        st.subheader("First 5 rows of your data:")
        st.write(st.session_state.dataframe.head())

user_query = st.text_area("Ask a question about your financial data (e.g., \"What is the total revenue?\", \"Calculate net profit.\")")

if st.button("Analyze") and st.session_state.dataframe is not None and user_query:
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    df_schema = st.session_state.dataframe.columns.tolist()
    st.info(f"DataFrame columns: {df_schema}")

    code_gen_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an expert Python programmer.\n            You are given a pandas DataFrame named `df` with the following columns: {df_columns}.\n            Your task is to write Python code using pandas to answer the user's question.\n            The code should print the final answer to stdout. Do not explain the code.\n            Only provide the Python code block, e.g., print(df['column'].sum()).\n            If the question involves a calculation, print the calculated value directly.\n            If the question asks for a summary or trend, print the relevant pandas output.\n            Ensure the code is self-contained and executable.\n            """),
            ("human", "{query}")
        ]
    )
    code_gen_chain = code_gen_prompt_template | llm | StrOutputParser()

    st.subheader("Generating Python Code...")
    generated_code = code_gen_chain.invoke({"df_columns": df_schema, "query": user_query})
    st.code(generated_code, language="python")

    st.subheader("Executing Code...")
    execution_output = execute_financial_code(generated_code, st.session_state.dataframe)
    st.write("Code Execution Output:")
    st.text(execution_output)

    answer_gen_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a financial analyst assistant.\n            The user asked a question about their financial data.\n            You generated and executed Python code to answer it.\n            Here is the original question: {original_query}\n            Here is the generated Python code:\n            ```python\n            {generated_code}\n            ```\n            Here is the output from executing the code:\n            {execution_output}\n            Based on the original question and the code execution output, provide a concise and clear natural language answer and insights for the small business owner.\n            Explain the results simply.\n            """),
            ("human", "Generate the natural language explanation.")
        ]
    )
    answer_gen_chain = answer_gen_prompt_template | llm | StrOutputParser()

    st.subheader("Generating Natural Language Answer...")
    final_answer = answer_gen_chain.invoke({
        "original_query": user_query,
        "generated_code": generated_code,
        "execution_output": execution_output
    })
    st.success("Analysis Complete!")
    st.markdown("### Your Financial Insight:")
    st.write(final_answer)
elif st.button("Analyze"):
    if st.session_state.dataframe is None:
        st.warning("Please upload a financial data file first.")
    elif not user_query:
        st.warning("Please enter a query.")