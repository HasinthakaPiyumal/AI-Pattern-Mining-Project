import pandas as pd
import numpy as np
import io
import sys
import json

class FinancialAnalystCAR:
    def __init__(self):
        self.financial_data = self._generate_sample_financial_data()
        self.llm_knowledge_base = {
            "moving average": {
                "description": "Calculates the simple moving average for a specified column over a given window.",
                "code_template": "df['{column}'].rolling(window={window}).mean().rename('{} {} MA'.format(df['{column}'].name, {window}))"
            },
            "daily returns": {
                "description": "Calculates the daily percentage change for a specified column.",
                "code_template": "df['{column}'].pct_change().rename('{} Daily Returns'.format(df['{column}'].name))"
            },
            "z-score anomalies": {
                "description": "Calculates the Z-score for a specified column to identify potential anomalies.",
                "code_template": "((df['{column}'] - df['{column}'].mean()) / df['{column}'].std()).rename('{} Z-Score'.format(df['{column}'].name))"
            },
            "compound annual growth rate": {
                "description": "Calculates the Compound Annual Growth Rate (CAGR) for an investment.",
                "code_template": "((df['{column}'].iloc[-1] / df['{column}'].iloc[0])**(1/(len(df)-1))) - 1"
            },
            "std dev": {
                "description": "Calculates the standard deviation for a specified column.",
                "code_template": "df['{column}'].std().rename('{} Standard Deviation'.format(df['{column}'].name))"
            }
        }

    def _generate_sample_financial_data(self):
        """Generates a sample pandas DataFrame for financial data."""
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        np.random.seed(42)
        data = {
            'Open': np.random.uniform(100, 150, 100).cumsum() / 10 + 100,
            'High': np.random.uniform(105, 155, 100).cumsum() / 10 + 100,
            'Low': np.random.uniform(95, 145, 100).cumsum() / 10 + 100,
            'Close': np.random.uniform(100, 150, 100).cumsum() / 10 + 100,
            'Volume': np.random.randint(100000, 1000000, 100),
            'Investment': (np.random.rand(100) - 0.5).cumsum() + 1000
        }
        df = pd.DataFrame(data, index=dates)
        df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 5, 100)
        df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 5, 100)
        df['Close'] = df['Close'].apply(lambda x: max(x, 0.01)) # Ensure no zero or negative close prices for returns
        df['Investment'] = df['Investment'].apply(lambda x: max(x, 100)) # Ensure positive investment
        return df.round(2)

    def _simulate_llm_code_generation(self, query: str) -> tuple[str, str, dict]:
        """Simulates an LLM generating Python code based on a natural language query.
        In a real scenario, this would involve an actual LLM API call.
        Returns (generated_code, result_variable_name, metadata).
        """
        query_lower = query.lower()
        code_to_execute = ""
        result_var = "calculated_result"
        metadata = {}

        if "moving average" in query_lower and ("close" in query_lower or "price" in query_lower):
            window = 10
            if "over" in query_lower:
                try:
                    window_str = query_lower.split("over ")[1].split(" periods")[0].strip()
                    window = int(window_str)
                except (IndexError, ValueError): pass # Use default if parsing fails
            column = 'Close'
            code_to_execute = self.llm_knowledge_base["moving average"]["code_template"].format(column=column, window=window)
            metadata['operation'] = f"{window}-day Moving Average"
            metadata['column'] = column
        elif "daily returns" in query_lower and ("close" in query_lower or "price" in query_lower):
            column = 'Close'
            code_to_execute = self.llm_knowledge_base["daily returns"]["code_template"].format(column=column)
            metadata['operation'] = "Daily Returns"
            metadata['column'] = column
        elif "anomaly" in query_lower and "z-score" in query_lower and "volume" in query_lower:
            column = 'Volume'
            code_to_execute = self.llm_knowledge_base["z-score anomalies"]["code_template"].format(column=column)
            metadata['operation'] = "Z-Score Anomaly Detection"
            metadata['column'] = column
        elif "compound annual growth rate" in query_lower or "cagr" in query_lower and "investment" in query_lower:
            column = 'Investment'
            code_to_execute = self.llm_knowledge_base["compound annual growth rate"]["code_template"].format(column=column)
            metadata['operation'] = "Compound Annual Growth Rate"
            metadata['column'] = column
            result_var = "cagr_result"
            # For CAGR, we assign the result to a simple variable, not a series
            code_to_execute = f"{result_var} = " + code_to_execute
        elif "standard deviation" in query_lower and ("close" in query_lower or "price" in query_lower):
            column = 'Close'
            code_to_execute = self.llm_knowledge_base["std dev"]["code_template"].format(column=column)
            metadata['operation'] = "Standard Deviation"
            metadata['column'] = column
        else:
            return "", "", {"error": "Could not generate relevant code for the query."}

        # Wrap the expression for series/dataframe results
        if not code_to_execute.startswith(f"{result_var} = ") and result_var != "cagr_result":
             code_to_execute = f"{result_var} = " + code_to_execute

        return code_to_execute, result_var, metadata

    def _execute_python_code(self, code_string: str, df: pd.DataFrame) -> dict:
        """Safely executes generated Python code using an external interpreter (exec).
        Returns a dictionary containing the results.
        """
        local_vars = {'df': df, 'pd': pd, 'np': np}
        global_vars = {'pd': pd, 'np': np} # Limit global scope for safety

        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            exec(code_string, global_vars, local_vars)
            output = redirected_output.getvalue()
            # Extract the assigned variable if it exists
            if 'calculated_result' in local_vars:
                result = local_vars['calculated_result']
                if isinstance(result, (pd.Series, pd.DataFrame)):
                    return {'result': result.to_dict(), 'type': 'pandas', 'stdout': output}
                else:
                    return {'result': result, 'type': type(result).__name__, 'stdout': output}
            elif 'cagr_result' in local_vars: # Specific handling for CAGR which is a scalar
                return {'result': local_vars['cagr_result'], 'type': 'float', 'stdout': output}
            else:
                return {'result': None, 'type': 'None', 'stdout': output, 'warning': 'No specific result variable found.'}
        except Exception as e:
            return {'error': str(e), 'stdout': redirected_output.getvalue()}
        finally:
            sys.stdout = old_stdout

    def generate_financial_report(self, natural_language_query: str) -> dict:
        """Generates a financial report by leveraging code-assisted reasoning.
        The LLM generates code, which is executed, and results are integrated.
        """
        print(f"\nUser Query: {natural_language_query}")

        # Step 1: LLM generates code based on the query
        generated_code, result_var, metadata = self._simulate_llm_code_generation(natural_language_query)
        if "error" in metadata:
            return {"status": "error", "message": metadata["error"]}
        if not generated_code:
            return {"status": "error", "message": "Could not generate code for the given query."}

        print(f"Generated Python Code:\n```python\n{generated_code}\n```")

        # Step 2: Execute the generated code
        execution_results = self._execute_python_code(generated_code, self.financial_data)

        if "error" in execution_results:
            return {"status": "error", "message": f"Error executing code: {execution_results['error']}"}

        # Step 3: LLM (simulated) interprets the precise results and formulates the report
        report_parts = []
        report_parts.append("Financial Report Summary:\n")
        report_parts.append(f"Based on your query: '{natural_language_query}'\n")

        calculated_value = execution_results.get('result')
        operation = metadata.get('operation', 'calculation')
        column = metadata.get('column', 'data')

        if calculated_value is not None:
            if isinstance(calculated_value, dict) and execution_results.get('type') == 'pandas':
                # Convert dict back to Series/DataFrame for easier handling
                if column and len(calculated_value) == len(self.financial_data):
                    result_series = pd.Series(calculated_value, index=self.financial_data.index, name=f'{column} {operation}')
                    report_parts.append(f"Operation: {operation} on '{column}'\n")
                    report_parts.append("-------------------------------------\n")
                    report_parts.append(f"First 5 results:\n{result_series.head().to_string()}\n")
                    report_parts.append(f"Last 5 results:\n{result_series.tail().to_string()}\n")

                    if "anomaly" in operation.lower():
                        # Simple anomaly detection logic for Z-score
                        threshold = 2.0
                        anomalies = result_series[abs(result_series) > threshold]
                        if not anomalies.empty:
                            report_parts.append(f"\nPotential Anomalies (Z-score > {threshold}):\n")
                            for date, value in anomalies.items():
                                report_parts.append(f"  Date: {date.strftime('%Y-%m-%d')}, Value: {self.financial_data.loc[date, column]:.2f}, Z-Score: {value:.2f}\n")
                            report_parts.append(f"\nRecommendation: Investigate these periods for unusual activity in '{column}'.\n")
                        else:
                            report_parts.append("\nNo significant anomalies detected based on the Z-score threshold.\n")
                else:
                    report_parts.append(f"Detailed results (first 10 items):\n{json.dumps(list(calculated_value.items())[:10], indent=2)}\n")
            elif execution_results.get('type') == 'float': # For CAGR and other scalar results
                 report_parts.append(f"Operation: {operation} on '{column}'\n")
                 report_parts.append(f"-------------------------------------\n")
                 report_parts.append(f"Calculated {operation}: {calculated_value:.4f}\n")
                 if operation == "Compound Annual Growth Rate":
                     report_parts.append(f"This indicates an average annual growth rate of {calculated_value:.2%} for the investment over the period.\n")
                 elif operation == "Standard Deviation":
                     report_parts.append(f"The standard deviation of {calculated_value:.2f} for '{column}' indicates the volatility or dispersion of its values.\n")
            else:
                report_parts.append(f"Calculated Value: {calculated_value}\n")
        else:
            report_parts.append("No specific numerical result was extracted.")

        if execution_results.get('stdout'):
            report_parts.append(f"\nInterpreter Output:\n{execution_results['stdout']}")

        return {"status": "success", "report": "".join(report_parts)}

# Example Usage:
if __name__ == "__main__":
    analyst = FinancialAnalystCAR()

    queries = [
        "Calculate the 10-day moving average for the Close price.",
        "What are the daily returns for the Close price?",
        "Detect anomalies in Volume using Z-score.",
        "Calculate the Compound Annual Growth Rate for the Investment amount.",
        "What is the standard deviation of the Close price?",
        "Show me something random."
    ]

    for query in queries:
        report = analyst.generate_financial_report(query)
        print("\n=======================================================")
        if report["status"] == "success":
            print(report["report"])
        else:
            print(f"Error generating report: {report['message']}")
        print("=======================================================")

    # Demonstrate the generated data
    print("\n--- Sample Financial Data (Head) ---")
    print(analyst.financial_data.head().to_string())
    print("\n--- Sample Financial Data (Tail) ---")
    print(analyst.financial_data.tail().to_string())
