import pandas as pd
import subprocess
import json
import os

# Placeholder for LLM interaction (replace with actual API calls if an API key is available)
class MockLLM:
    def generate_code(self, prompt, financial_data_str):
        # Simulate LLM generating Python code based on prompt and data
        # For this example, we'll hardcode a simple calculation example.
        # In a real scenario, the LLM would dynamically create this code.
        if "calculate profit margin" in prompt.lower():
            code = f"""
import pandas as pd
import json

financial_data = json.loads('''{financial_data_str}''')
df = pd.DataFrame(financial_data)

# Ensure 'Revenue' and 'CostOfGoodsSold' columns exist and are numeric
if 'Revenue' in df.columns and 'CostOfGoodsSold' in df.columns:
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
    df['CostOfGoodsSold'] = pd.to_numeric(df['CostOfGoodsSold'], errors='coerce')
    
    # Calculate Gross Profit
    df['GrossProfit'] = df['Revenue'] - df['CostOfGoodsSold']
    
    # Calculate Gross Profit Margin
    df['GrossProfitMargin'] = (df['GrossProfit'] / df['Revenue']) * 100
    
    # Print relevant results
    print(f"Annual Revenue: {{df['Revenue'].sum():,.2f}}")
    print(f"Annual Cost of Goods Sold: {{df['CostOfGoodsSold'].sum():,.2f}}")
    print(f"Annual Gross Profit: {{df['GrossProfit'].sum():,.2f}}")
    print(f"Average Gross Profit Margin: {{df['GrossProfitMargin'].mean():.2f}}%")
else:
    print("Error: 'Revenue' or 'CostOfGoodsSold' columns not found in data.")
"""
            return code
        elif "calculate current ratio" in prompt.lower():
             code = f"""
import pandas as pd
import json

financial_data = json.loads('''{financial_data_str}''')
df = pd.DataFrame(financial_data)

# Assuming 'CurrentAssets' and 'CurrentLiabilities' exist
if 'CurrentAssets' in df.columns and 'CurrentLiabilities' in df.columns:
    df['CurrentAssets'] = pd.to_numeric(df['CurrentAssets'], errors='coerce')
    df['CurrentLiabilities'] = pd.to_numeric(df['CurrentLiabilities'], errors='coerce')

    total_current_assets = df['CurrentAssets'].sum()
    total_current_liabilities = df['CurrentLiabilities'].sum()

    if total_current_liabilities > 0:
        current_ratio = total_current_assets / total_current_liabilities
        print(f"Total Current Assets: {{total_current_assets:,.2f}}")
        print(f"Total Current Liabilities: {{total_current_liabilities:,.2f}}")
        print(f"Current Ratio: {{current_ratio:.2f}}")
    else:
        print("Error: Total Current Liabilities are zero, cannot calculate Current Ratio.")
else:
    print("Error: 'CurrentAssets' or 'CurrentLiabilities' columns not found in data.")
"""
             return code
        return "print('No specific financial calculation code generated for this prompt.')"

    def generate_report(self, analysis_results, original_query):
        # Simulate LLM generating a natural language report
        report = f"""
        Financial Analysis Report based on your query: "{original_query}"

        ---
        Calculations and Key Metrics:
        {analysis_results}
        ---

        Interpretation:
        Based on the precise calculations performed by our financial engine, here's an interpretation:
        - The company's profitability (if profit margin was calculated) indicates...
        - Its short-term liquidity (if current ratio was calculated) appears to be...
        - Further analysis might involve...

        Disclaimer: This report is based on the provided data and automated calculations. For critical decisions, consult a financial expert.
        """
        return report

# Initialize Mock LLM
mock_llm = MockLLM()

def load_mock_financial_data():
    """
    Loads mock financial data representing a simplified income statement
    and balance sheet data.
    """
    data = {
        'Date': ['2022-12-31', '2021-12-31', '2020-12-31'],
        'Revenue': [1000000, 950000, 900000],
        'CostOfGoodsSold': [600000, 580000, 550000],
        'OperatingExpenses': [200000, 190000, 180000],
        'CurrentAssets': [400000, 380000, 350000],
        'CurrentLiabilities': [200000, 190000, 175000],
        'TotalAssets': [1500000, 1400000, 1300000],
        'TotalLiabilities': [800000, 750000, 700000]
    }
    return pd.DataFrame(data)

def execute_python_code(code: str) -> str:
    """
    Executes the given Python code in a subprocess and captures its stdout.
    """
    temp_file_name = "generated_financial_script.py"
    try:
        with open(temp_file_name, "w") as f:
            f.write(code)

        result = subprocess.run(
            ["python", temp_file_name],
            capture_output=True,
            text=True,
            check=True,
            timeout=30 # Set a timeout for execution
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing code: {e.stderr}"
    except FileNotFoundError:
        return "Error: Python interpreter not found."
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out."
    finally:
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name) # Clean up the temporary file

def run_financial_analysis(query: str):
    """
    Orchestrates the PAL prompting process for financial analysis.
    """
    print(f"User Query: {query}\n")

    # 1. Load Financial Data
    financial_df = load_mock_financial_data()
    financial_data_json = financial_df.to_json(orient="records")
    print("Mock Financial Data Loaded and converted to JSON for LLM.\n")

    # 2. LLM generates Python code for analysis
    print("Prompting LLM to generate Python code for analysis...")
    generated_code = mock_llm.generate_code(query, financial_data_json)
    print("\n--- Generated Python Code ---\n")
    print(generated_code)
    print("\n-----------------------------\n")

    # 3. Execute the generated Python code
    print("Executing generated Python code...")
    execution_output = execute_python_code(generated_code)
    print("\n--- Code Execution Output ---\n")
    print(execution_output)
    print("\n-----------------------------\n")

    # 4. LLM generates a natural language report based on the results
    print("Prompting LLM to generate natural language report...")
    final_report = mock_llm.generate_report(execution_output, query)
    print("\n--- Final Financial Analysis Report ---\n")
    print(final_report)
    print("\n-------------------------------------\n")
    return final_report

if __name__ == "__main__":
    # Example usage:
    analysis_query_profit_margin = "Please calculate the annual gross profit and average gross profit margin for the company."
    run_financial_analysis(analysis_query_profit_margin)

    print("\n" + "="*80 + "\n")

    analysis_query_current_ratio = "What is the company's current ratio, indicating its short-term liquidity?"
    run_financial_analysis(analysis_query_current_ratio)