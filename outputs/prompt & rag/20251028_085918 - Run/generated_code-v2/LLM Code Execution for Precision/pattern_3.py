import streamlit as st
import pandas as pd
import io
import subprocess
import json

st.title("Automated Financial Report Generation and Analysis")
st.write("Upload your financial data (CSV) and ask natural language questions.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
user_query = st.text_input("Enter your financial query (e.g., \"Calculate my total revenue\")")

def call_llm_for_code_generation(query, df_head_info):
    if "total revenue" in query.lower():
        code = """
import pandas as pd
import json

def calculate_total_revenue(df):
    if 'Revenue' in df.columns:
        total_revenue = df['Revenue'].sum()
        return json.dumps({"result": total_revenue, "metric": "Total Revenue"})
    else:
        return json.dumps({"error": "Revenue column not found. Please ensure your CSV has a 'Revenue' column."})
"""
        return code, "calculate_total_revenue"
    elif "profit margin" in query.lower() and "revenue" in query.lower() and "expenses" in query.lower():
        code = """
import pandas as pd
import json

def calculate_profit_margin(df):
    if 'Revenue' in df.columns and 'Expenses' in df.columns:
        total_revenue = df['Revenue'].sum()
        total_expenses = df['Expenses'].sum()
        if total_revenue == 0:
            return json.dumps({"result": 0.0, "metric": "Profit Margin (Revenue is zero)"})
        profit_margin = ((total_revenue - total_expenses) / total_revenue) * 100
        return json.dumps({"result": profit_margin, "metric": "Profit Margin"})
    else:
        return json.dumps({"error": "Required columns (Revenue, Expenses) not found. Please ensure your CSV has both."})
"""
        return code, "calculate_profit_margin"
    else:
        return None, None

def call_llm_for_response_generation(query, code_output):
    if code_output:
        try:
            output_data = json.loads(code_output)
            if "error" in output_data:
                return f"I encountered an error during calculation: {output_data['error']}"
            elif "result" in output_data and "metric" in output_data:
                result_val = output_data["result"]
                metric_name = output_data["metric"]
                if "profit margin" in metric_name.lower():
                    return f"Based on your data, the {metric_name} is {result_val:.2f}%."
                elif "total revenue" in metric_name.lower():
                    return f"Based on your data, your {metric_name} is ${result_val:,.2f}."
                else:
                    return f"The calculation for '{query}' resulted in: {metric_name} = {result_val}."
        except json.JSONDecodeError:
            return f"The code execution produced an unexpected output: {code_output}"
    return f"I couldn't generate a specific financial insight for your query: '{query}'. Please try rephrasing or provide more details."

def execute_generated_code(code, function_name, df):
    df_json = df.to_json(orient='records')
    
    script_content = f"""
{code}

if __name__ == "__main__":
    import sys
    import pandas as pd
    import json
    
    input_data_json = sys.stdin.read()
    df_data = pd.read_json(input_data_json)
    
    try:
        result = {function_name}(df_data)
        print(result)
    except Exception as e:
        print(json.dumps({{"error": str(e)}}))
"""
    
    try:
        process = subprocess.run(
            ["python", "-c", script_content],
            input=df_json,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": f"Code execution failed with exit code {e.returncode}: {e.stderr.strip()}"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Code execution timed out."})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during code execution setup: {str(e)}"}) 

if uploaded_file is not None:
    st.subheader("Uploaded Data Preview")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if user_query:
        st.subheader("Processing Query...")
        
        df_head_info = df.head().to_dict('records')
        
        generated_code, function_to_call = call_llm_for_code_generation(user_query, df_head_info)
        
        if generated_code and function_to_call:
            st.text("LLM Generated Code:")
            st.code(generated_code, language="python")
            
            st.text("Executing Code...")
            code_execution_output = execute_generated_code(generated_code, function_to_call, df)
            
            st.text("Code Execution Raw Output:")
            st.write(code_execution_output)
            
            final_response = call_llm_for_response_generation(user_query, code_execution_output)
            
            st.subheader("Financial Insight:")
            st.write(final_response)
        else:
            st.error("I couldn't generate code for your request. Please try a different query or ensure your data has relevant columns (e.g., 'Revenue', 'Expenses').")
else:
    st.info("Please upload a CSV file to begin your financial analysis.")