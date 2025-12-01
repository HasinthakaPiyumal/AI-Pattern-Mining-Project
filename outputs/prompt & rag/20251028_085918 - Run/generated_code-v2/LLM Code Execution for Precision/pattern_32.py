import pandas as pd

def mock_llm_generate_code(prompt: str) -> str:
    """Simulates an LLM generating Python code based on a financial analysis prompt."""
    if "ROI" in prompt:
        return "def calculate_roi(initial_investment, current_value):\n    return ((current_value - initial_investment) / initial_investment) * 100\n\ninitial = 1000\ncurrent = 1250\n_execution_output = {'ROI': calculate_roi(initial, current)}"
    elif "average closing price" in prompt:
        return "_execution_output = {'average_price': data['Close'].mean()}"
    elif "trend analysis" in prompt:
        return "_execution_output = {'trend_summary': 'Data shows a general upward trend based on recent closing prices.'}"
    else:
        return "_execution_output = {'error': 'Could not generate relevant code for the request.'}"

def execute_financial_code(code_string: str, data: pd.DataFrame = None) -> dict:
    """Safely executes generated Python code and captures its output."""
    execution_globals = {}
    execution_locals = {'data': data, '_execution_output': {}}
    try:
        exec(code_string, execution_globals, execution_locals)
        return execution_locals.get('_execution_output', {'error': 'Code executed but no output variable found.'})
    except Exception as e:
        return {'error': f"Code execution failed: {e}"}

def mock_llm_generate_report(original_request: str, computational_results: dict) -> str:
    """Simulates an LLM generating a financial report based on the request and computational results."""
    report = f"Financial Analysis Report for: '{original_request}'\n\n"
    
    if computational_results and not computational_results.get('error'):
        report += "Computational Results:\n"
        for key, value in computational_results.items():
            report += f"- {key.replace('_', ' ').title()}: {value:.2f}%\n" if 'ROI' in key else f"- {key.replace('_', ' ').title()}: {value}\n"
        report += "\n"
        
        if 'ROI' in computational_results:
            report += f"Recommendation: Based on the calculated ROI of {computational_results['ROI']:.2f}%, this investment shows a positive return. Further analysis of market conditions and risk factors is recommended.\n"
        elif 'average_price' in computational_results:
            report += f"Market Insight: The average closing price of {computational_results['average_price']:.2f} provides a baseline for evaluating recent price movements.\n"
        elif 'trend_summary' in computational_results:
            report += f"Trend Analysis: {computational_results['trend_summary']}\n"
        else:
            report += "Overall, the computational results provide valuable insights for decision-making.\n"
    else:
        report += f"No specific computational results or an error occurred during execution: {computational_results.get('error', 'Unknown error')}. The report is based solely on the initial request.\n"
        report += "Recommendation: Due to lack of precise computations, a general market overview suggests careful consideration.\n"

    report += "\nDisclaimer: This report is for informational purposes only and does not constitute financial advice."
    return report

if __name__ == "__main__":
    # --- Conceptual User Interface / Main Flow ---

    # Sample Financial Data (e.g., historical stock prices)
    sample_financial_data = pd.DataFrame({
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
        'Open': [100.0, 102.5, 101.0, 103.0, 104.5],
        'High': [103.0, 104.0, 102.5, 105.0, 106.0],
        'Low': [99.5, 100.0, 100.5, 102.0, 103.0],
        'Close': [102.0, 101.5, 102.0, 104.0, 105.5],
        'Volume': [1000, 1200, 1100, 1300, 1400]
    })

    print("--- Financial Data Analysis System (PAL Prompting Demo) ---\n")

    # --- Scenario 1: Calculate ROI ---
    analysis_request_1 = "Calculate the Return on Investment for an initial investment of 1000 which is now worth 1250."
    print(f"User Request 1: {analysis_request_1}")

    # LLM generates code
    generated_code_1 = mock_llm_generate_code(analysis_request_1)
    print(f"Generated Code 1: {generated_code_1}\n")

    # Execute code
    computational_results_1 = execute_financial_code(generated_code_1)
    print(f"Computational Results 1: {computational_results_1}\n")

    # LLM generates report
    final_report_1 = mock_llm_generate_report(analysis_request_1, computational_results_1)
    print(f"Final Report 1:\n{final_report_1}\n{'='*80}\n")

    # --- Scenario 2: Average Closing Price ---
    analysis_request_2 = "What is the average closing price from the provided financial data?"
    print(f"User Request 2: {analysis_request_2}")

    # LLM generates code
    generated_code_2 = mock_llm_generate_code(analysis_request_2)
    print(f"Generated Code 2: {generated_code_2}\n")

    # Execute code with data
    computational_results_2 = execute_financial_code(generated_code_2, sample_financial_data)
    print(f"Computational Results 2: {computational_results_2}\n")

    # LLM generates report
    final_report_2 = mock_llm_generate_report(analysis_request_2, computational_results_2)
    print(f"Final Report 2:\n{final_report_2}\n{'='*80}\n")

    # --- Scenario 3: Trend Analysis (more qualitative) ---
    analysis_request_3 = "Perform a general trend analysis on the financial data."
    print(f"User Request 3: {analysis_request_3}")

    # LLM generates code (or a qualitative statement)
    generated_code_3 = mock_llm_generate_code(analysis_request_3)
    print(f"Generated Code 3: {generated_code_3}\n")

    # Execute code
    computational_results_3 = execute_financial_code(generated_code_3, sample_financial_data)
    print(f"Computational Results 3: {computational_results_3}\n")

    # LLM generates report
    final_report_3 = mock_llm_generate_report(analysis_request_3, computational_results_3)
    print(f"Final Report 3:\n{final_report_3}\n{'='*80}\n")