import re
import io
import sys

class FinancialAnalystAssistant:
    def __init__(self):
        self.history = [] # For context, if needed in a more advanced version

    def _generate_code(self, query: str) -> str:
        """
        Simulates the LLM generating Python code based on the query.
        In a real PAL system, this would be a sophisticated LLM prompt.
        """
        if "P/E ratio" in query and ("EPS" in query or "earnings per share" in query) and ("stock price" in query or "current price" in query):
            eps_match = re.search(r"EPS of ([\d.]+)|earnings per share of ([\d.]+)", query, re.IGNORECASE)
            price_match = re.search(r"stock price of ([\d.]+)|current price of ([\d.]+)", query, re.IGNORECASE)

            eps = float(eps_match.group(1) or eps_match.group(2)) if eps_match and (eps_match.group(1) or eps_match.group(2)) else None
            price = float(price_match.group(1) or price_match.group(2)) if price_match and (price_match.group(1) or price_match.group(2)) else None

            if eps is not None and price is not None:
                return f"""
def calculate_pe(earnings_per_share, current_stock_price):
    if earnings_per_share == 0:
        return float(\'inf\') # Handle division by zero gracefully
    return current_stock_price / earnings_per_share

eps_val = {eps}
stock_price_val = {price}
pe_ratio = calculate_pe(eps_val, stock_price_val)
print(f"__PAL_RESULT__:PE_RATIO_VALUE:{{pe_ratio:.2f}}")
"""
        elif "cash flow projection" in query or "project cash flow" in query:
            initial_cf_match = re.search(r"starting from ([\d.]+)", query, re.IGNORECASE)
            growth_rate_match = re.search(r"([\d.]+)% growth rate", query, re.IGNORECASE)
            num_years_match = re.search(r"over the next (\d+) years", query, re.IGNORECASE)

            initial_cf = float(initial_cf_match.group(1)) if initial_cf_match else 1000.0
            growth_rate = float(growth_rate_match.group(1)) / 100 if growth_rate_match else 0.10 # Default to 10%
            num_years = int(num_years_match.group(1)) if num_years_match else 3 # Default to 3 years

            code_lines = []
            code_lines.append(f"current_cf = {initial_cf}")
            for i in range(1, num_years + 1):
                if i > 1:
                    code_lines.append(f"current_cf = current_cf * (1 + {growth_rate})")
                code_lines.append(f"print(f\"__PAL_RESULT__:CASH_FLOW_Y{{i}}:{{current_cf:.2f}}\")")
            return "\n".join(code_lines)
        else:
            return "" # No code generated if not recognized

    def _execute_code(self, code: str) -> dict:
        """
        Executes the generated Python code in a sandboxed environment
        and captures its output.
        IMPORTANT: In a production environment, executing arbitrary code
        from an LLM requires robust sandboxing for security.
        This is a simplified example for demonstration.
        """
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        results = {}
        try:
            exec(code, {"__builtins__": {}}) # Limited builtins for basic safety
            output = redirected_output.getvalue()
            # Parse output for specific results markers
            for line in output.splitlines():
                if line.startswith("__PAL_RESULT__"):
                    key_value = line.replace("__PAL_RESULT__:", "").strip()
                    if ":" in key_value:
                        key, value = key_value.split(":", 1)
                        results[key.strip()] = value.strip()
        except Exception as e:
            results["ERROR"] = str(e)
            results["RAW_OUTPUT"] = redirected_output.getvalue()
        finally:
            sys.stdout = old_stdout # Restore stdout
        return results

    def _formulate_response(self, query: str, code_results: dict) -> str:
        """
        Simulates the LLM integrating computational results into a natural language response.
        """
        if "ERROR" in code_results:
            return f"I encountered an error while processing your request: {code_results["ERROR"]}. Raw output: {code_results.get("RAW_OUTPUT", "N/A")}"

        response_parts = []
        if "PE_RATIO_VALUE" in code_results:
            pe_ratio = code_results["PE_RATIO_VALUE"]
            response_parts.append(f"Based on my calculations, the P/E ratio is approximately {pe_ratio}.")
            if pe_ratio == "inf":
                response_parts.append("This indicates that the company has zero or negative earnings.")
            else:
                response_parts.append("This figure can be used for financial analysis and comparison with industry peers.")
        
        cash_flow_keys = [k for k in code_results if k.startswith("CASH_FLOW_Y")]
        if cash_flow_keys:
            response_parts.append("Here are the projected cash flows:")
            # Sort keys to ensure years are in order
            sorted_cash_flow_keys = sorted(cash_flow_keys, key=lambda x: int(x.split("Y")[1]))
            for k in sorted_cash_flow_keys:
                response_parts.append(f"Year {k.split("Y")[1]}: ${code_results[k]}")
            response_parts.append("These projections are based on the assumed initial values and growth rates and should be reviewed with further financial data.")

        if not response_parts and code_results: # If results but no specific handler
            response_parts.append("I have processed the request and found the following results:")
            for k, v in code_results.items():
                response_parts.append(f"- {k}: {v}")

        if not response_parts: # If no code was executed or no results found
            return f"I couldn\"t generate a precise answer for \"{query}\". Please rephrase or provide more context."

        return " ".join(response_parts)

    def answer_query(self, query: str) -> str:
        """
        Main method to process a user query using the PAL prompting approach.
        """
        # 1. LLM (simulated) decides if code generation is needed and generates it.
        generated_code = self._generate_code(query)

        if generated_code:
            print(f"\n--- LLM Generated Code ---\n{generated_code}\n--------------------------")
            # 2. Execute the generated code.
            code_results = self._execute_code(generated_code)
            print(f"--- Code Execution Results ---\n{code_results}\n------------------------------")
            # 3. LLM (simulated) integrates results into a natural language response.
            final_response = self._formulate_response(query, code_results)
        else:
            final_response = f"I\"m sorry, I cannot perform that calculation at the moment. My capabilities are focused on financial computations like P/E ratios or cash flow projections. Please provide a query related to these areas."

        self.history.append({"query": query, "response": final_response})
        return final_response
