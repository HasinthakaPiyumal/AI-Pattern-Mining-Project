
import subprocess
import json

class FinancialAssistant:
    def __init__(self, llm_model_name="SimulatedLLM"):
        self.llm_model_name = llm_model_name

    def _simulate_llm_code_generation(self, prompt: str) -> str:
        """Simulates an LLM generating Python code based on the prompt.
        In a real application, this would involve an actual LLM API call.
        For demonstration, it returns a hardcoded or simple dynamic code snippet.
        """
        if "DCF valuation" in prompt:
            return """
# Python code for a simplified DCF valuation
def calculate_dcf(free_cash_flows, discount_rate, growth_rate_terminal, terminal_year):
    npv = 0
    for i, fcf in enumerate(free_cash_flows):
        npv += fcf / ((1 + discount_rate)**(i + 1))
    
    terminal_value = free_cash_flows[-1] * (1 + growth_rate_terminal) / (discount_rate - growth_rate_terminal)
    npv += terminal_value / ((1 + discount_rate)**terminal_year)
    return npv

import numpy as np

# Example data for demonstration
free_cash_flows_example = [100, 110, 120, 130, 140]
discount_rate_example = 0.10
growth_rate_terminal_example = 0.03
terminal_year_example = len(free_cash_flows_example)

dcf_result = calculate_dcf(free_cash_flows_example, discount_rate_example, growth_rate_terminal_example, terminal_year_example)
print(f"{{\"result_type\": \"DCF Valuation\", \"value\": {dcf_result:.2f}}}")
"""
        elif "financial ratios" in prompt:
            return """
# Python code for calculating basic financial ratios
def calculate_ratios(revenue, cost_of_goods_sold, operating_expenses, net_income, total_assets, total_liabilities, equity):
    gross_profit = revenue - cost_of_goods_sold
    gross_margin = (gross_profit / revenue) * 100 if revenue else 0
    operating_income = gross_profit - operating_expenses
    operating_margin = (operating_income / revenue) * 100 if revenue else 0
    net_profit_margin = (net_income / revenue) * 100 if revenue else 0
    debt_to_equity = total_liabilities / equity if equity else 0
    return {
        "gross_margin": f"{gross_margin:.2f}%",
        "operating_margin": f"{operating_margin:.2f}%",
        "net_profit_margin": f"{net_profit_margin:.2f}%",
        "debt_to_equity": f"{debt_to_equity:.2f}"
    }

# Example data
rev, cogs, opex, net_inc = 1000000, 400000, 300000, 200000
assets, liabilities, eq = 5000000, 2000000, 3000000

ratios = calculate_ratios(rev, cogs, opex, net_inc, assets, liabilities, eq)
print(json.dumps(ratios))
"""
        else:
            return f"print(\"No specific code generated for: {prompt}. Please try a more specific financial query.\")"

    def _execute_python_code(self, code: str) -> str:
        """Executes the given Python code in a subprocess and returns its stdout.
        Handles potential errors during execution.
        """
        try:
            # It's crucial to execute untrusted code in a secure, sandboxed environment
            # For this example, we're using a simple subprocess call, which is NOT PRODUCTION-READY for untrusted code.
            # A real application would use containers, secure sandboxes (e.g., gVisor, WebAssembly), or dedicated execution services.
            process = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                check=True, # Raise an exception for non-zero exit codes
                timeout=30 # Prevent long-running processes
            )
            return process.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Error during code execution: {e.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out."
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    def _simulate_llm_response_generation(self, original_query: str, code_output: str) -> str:
        """Simulates an LLM formulating a natural language response based on the original query and code output.
        In a real application, this would involve another LLM API call.
        """
        if "Error" in code_output:
            return f"I encountered an issue processing your request regarding '{original_query}'. The computation failed with the following message: {code_output}. Please ensure your query is well-formed or try a different approach."
        
        try:
            # Attempt to parse as JSON if it looks like structured output
            parsed_output = json.loads(code_output)
            if isinstance(parsed_output, dict):
                response_parts = []
                if parsed_output.get("result_type") == "DCF Valuation":
                    response_parts.append(f"Based on the Discounted Cash Flow (DCF) valuation, the calculated intrinsic value is approximately ${parsed_output['value']}.")
                    response_parts.append(f"This was derived using standard DCF methodology with the provided input parameters.")
                elif "gross_margin" in parsed_output and "net_profit_margin" in parsed_output:
                    response_parts.append(f"Here are the financial ratios you requested:")
                    for key, value in parsed_output.items():
                        response_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
                    response_parts.append("These ratios provide insights into the company's profitability and leverage.")
                else:
                    response_parts.append(f"I have completed the computational task for '{original_query}'. The structured result is: {json.dumps(parsed_output, indent=2)}")
                return "\n".join(response_parts)
        except json.JSONDecodeError:
            # If not JSON, treat as plain text output
            pass

        return f"I have processed your request regarding '{original_query}'. The computational engine returned the following result:\n\n{code_output}\n\nBased on this, I can provide further insights if you specify what you'd like to analyze next."

    def analyze_financial_query(self, query: str) -> str:
        """Main method to process a financial query using LLM for code generation and execution.
        """
        print(f"\n--- Processing Query: {query} ---")
        print("1. LLM (Simulated) generating code...")
        generated_code = self._simulate_llm_code_generation(query)
        print("\n--- Generated Code ---")
        print(generated_code)
        print("----------------------")

        if "No specific code generated" in generated_code:
            return self._simulate_llm_response_generation(query, generated_code)

        print("2. Executing generated Python code...")
        code_execution_output = self._execute_python_code(generated_code)
        print("\n--- Code Execution Output ---")
        print(code_execution_output)
        print("---------------------------")

        print("3. LLM (Simulated) formulating final response...")
        final_response = self._simulate_llm_response_generation(query, code_execution_output)
        print("\n--- Final Response ---")
        print(final_response)
        print("----------------------")
        return final_response

if __name__ == "__main__":
    assistant = FinancialAssistant()

    # Example 1: DCF Valuation
    assistant.analyze_financial_query("Perform a Discounted Cash Flow (DCF) valuation for a company with given free cash flows.")

    # Example 2: Financial Ratios
    assistant.analyze_financial_query("Calculate key financial ratios like gross margin and debt-to-equity.")
    
    # Example 3: Unrecognized query
    assistant.analyze_financial_query("What's the best stock to buy next week?")

    # Example 4: Query that might lead to execution error (simulated by LLM not providing proper code for some cases)
    # (This example is more conceptual as the current sim_llm_code_generation is robust for the specific cases)
    # To simulate an error, one could modify the generated code to be invalid.
    # For instance, if the LLM generated 'print(1/0)'