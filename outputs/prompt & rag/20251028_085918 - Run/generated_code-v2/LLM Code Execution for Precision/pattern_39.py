
import json
import io
import sys

# Mocking external modules for demonstration purposes
# In a real application, these would be separate files/services

class MockLLM:
    """A mock LLM class that simulates code generation and report generation."""
    def generate_code(self, query: str, available_data: dict) -> str:
        """Simulates an LLM generating Python code based on a query.
        For this demo, it returns a hardcoded example.
        """
        print(f"MockLLM: Generating code for query: '{query}'")
        # Example: Simulating a query for calculating average stock price
        if "average price" in query.lower() or "mean price" in query.lower():
            return (
                "import pandas as pd\n\n"\
                "df = pd.DataFrame(financial_data['stocks'])\n"\
                "result = {'average_price': df['price'].mean()}\n"\
                "print(json.dumps(result))"
            )
        elif "portfolio value" in query.lower():
             return (
                "import pandas as pd\n\n"\
                "df_holdings = pd.DataFrame(financial_data['portfolio_holdings'])\n"\
                "df_stocks = pd.DataFrame(financial_data['stocks'])\n"\
                "merged_df = pd.merge(df_holdings, df_stocks, on='symbol', how='left')\n"\
                "merged_df['value'] = merged_df['shares'] * merged_df['price']\n"\
                "total_portfolio_value = merged_df['value'].sum()\n"\
                "result = {'total_portfolio_value': total_portfolio_value}\n"\
                "print(json.dumps(result))"
            )
        else:
            return (
                "# No specific code generation logic for this query in mock\n"\
                "result = {'error': 'Could not generate specific code for this query in mock.'}\n"\
                "print(json.dumps(result))"
            )

    def generate_report(self, query: str, code_executed: str, execution_result: dict) -> str:
        """Simulates an LLM generating a human-readable report.
        """
        print(f"MockLLM: Generating report for query: '{query}' with result: {execution_result}")
        if "average_price" in execution_result and execution_result["average_price"] is not None:
            avg_price = execution_result["average_price"]
            return f"Based on your request, the average stock price is approximately ${avg_price:.2f}. This was calculated by executing the following Python code:\n```python\n{code_executed}\n```\n"
        elif "total_portfolio_value" in execution_result and execution_result["total_portfolio_value"] is not None:
            portfolio_value = execution_result["total_portfolio_value"]
            return f"Your total portfolio value is approximately ${portfolio_value:.2f}. This was determined by calculating the value of each holding and summing them up, using the following Python script:\n```python\n{code_executed}\n```\n"
        elif "error" in execution_result:
            return f"I encountered an error while trying to fulfill your request: {execution_result['error']}\nCode attempted:\n```python\n{code_executed}\n```\nPlease try rephrasing your query."
        else:
            return f"I processed your request, but the result was not interpretable for a specific report. Raw output: {execution_result}\nCode executed:\n```python\n{code_executed}\n```\n"


class SafeCodeExecutor:
    """A class to safely execute generated Python code."""
    def execute_code(self, code: str, globals_dict: dict = None) -> dict:
        """Executes the given Python code in a restricted environment.
        Captures stdout and stderr. Returns results as a dictionary.
        """
        print("SafeCodeExecutor: Executing generated code...")
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = io.StringIO()
        redirected_error = io.StringIO()

        sys.stdout = redirected_output
        sys.stderr = redirected_error

        exec_globals = globals_dict if globals_dict is not None else {}
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)
            stdout_value = redirected_output.getvalue()
            stderr_value = redirected_error.getvalue()

            # Attempt to parse JSON output from the executed code
            try:
                # Assuming the code prints a JSON object as its result
                result = json.loads(stdout_value)
            except json.JSONDecodeError:
                result = {"raw_output": stdout_value.strip() if stdout_value else "No direct JSON output.", "error_parsing_json": True}

            return {
                "success": True,
                "result": result,
                "stdout": stdout_value,
                "stderr": stderr_value
            }
        except Exception as e:
            stderr_value = redirected_error.getvalue()
            return {
                "success": False,
                "error": str(e),
                "stdout": redirected_output.getvalue(),
                "stderr": stderr_value
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class FinancialData:
    """A class to provide mock financial data."""
    def get_data(self) -> dict:
        """Returns mock financial data.
        In a real application, this would fetch data from APIs/databases.
        """
        print("FinancialData: Providing mock data.")
        return {
            "stocks": [
                {"symbol": "AAPL", "price": 170.50, "volume": 10000000},
                {"symbol": "GOOG", "price": 135.20, "volume": 5000000},
                {"symbol": "MSFT", "price": 350.10, "volume": 7000000},
                {"symbol": "AMZN", "price": 145.75, "volume": 8000000},
            ],
            "portfolio_holdings": [
                {"symbol": "AAPL", "shares": 10},
                {"symbol": "GOOG", "shares": 5},
                {"symbol": "MSFT", "shares": 12},
            ],
            "economic_indicators": {"inflation_rate": 0.03, "gdp_growth": 0.02}
        }


class FinancialAnalystAssistant:
    """Main orchestrator for the AI-powered Financial Analyst Assistant."""
    def __init__(self):
        self.llm = MockLLM()
        self.executor = SafeCodeExecutor()
        self.financial_data_provider = FinancialData()

    def process_query(self, query: str) -> str:
        """Processes a natural language query from the user.
        """
        print(f"\nAssistant: Processing user query: '{query}'")

        # 1. Get financial data for the LLM to understand context
        financial_data = self.financial_data_provider.get_data()

        # 2. LLM generates Python code based on the query and available data
        generated_code = self.llm.generate_code(query, financial_data)
        print(f"\nGenerated Code:\n```python\n{generated_code}\n```")

        # 3. Execute the generated code safely
        # Pass financial_data to the execution environment
        execution_result = self.executor.execute_code(generated_code, globals_dict={'financial_data': financial_data})

        print(f"\nCode Execution Result: {execution_result}")

        # 4. LLM generates a human-readable report based on execution results
        report_output = {}
        if execution_result['success'] and 'result' in execution_result:
            report_output = execution_result['result']
        elif not execution_result['success']:
            report_output = {'error': execution_result['error'], 'stderr': execution_result['stderr']}
        else:
            report_output = {'error': 'Unexpected execution result structure.'}

        final_report = self.llm.generate_report(query, generated_code, report_output)

        return final_report


if __name__ == "__main__":
    assistant = FinancialAnalystAssistant()

    # Example Queries
    query1 = "What is the average price of all stocks?"
    report1 = assistant.process_query(query1)
    print(f"\n--- Final Report 1 ---\n{report1}")

    print("\n" + "="*50 + "\n")

    query2 = "Calculate the total value of my portfolio holdings."
    report2 = assistant.process_query(query2)
    print(f"\n--- Final Report 2 ---\n{report2}")

    print("\n" + "="*50 + "\n")

    query3 = "What is the current inflation rate?" # This will hit the mock LLM's generic error handling
    report3 = assistant.process_query(query3)
    print(f"\n--- Final Report 3 ---\n{report3}")

    print("\n" + "="*50 + "\n")

    query4 = "Show me the square root of 25." # This will also hit generic error, as no specific code gen logic
    report4 = assistant.process_query(query4)
    print(f"\n--- Final Report 4 ---\n{report4}")


