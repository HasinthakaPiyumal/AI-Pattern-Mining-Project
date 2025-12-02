import io
import contextlib
import re

class SmartFinancialAdvisor:
    def __init__(self):
        pass

    def _generate_code_from_query(self, query: str) -> str:
        if "investment" in query.lower() and "compounded" in query.lower() and "value" in query.lower():
            principal = 0.0
            rate = 0.0
            time = 0
            n = 0

            principal_match = re.search(r"\$(\d+(?:,\d+)*)", query)
            if principal_match:
                principal = float(principal_match.group(1).replace(",", ""))

            rate_match = re.search(r"(\d+(?:\.\d+)?)%", query)
            if rate_match:
                rate = float(rate_match.group(1)) / 100

            time_match = re.search(r"(\d+)\s*year", query, re.IGNORECASE)
            if time_match:
                time = int(time_match.group(1))

            if "quarterly" in query.lower():
                n = 4
            elif "annually" in query.lower():
                n = 1
            elif "monthly" in query.lower():
                n = 12
            elif "daily" in query.lower():
                n = 365
            else:
                n = 1

            if principal and rate and time and n:
                code_snippet = f"""
def calculate_compound_interest(principal, rate, time, n):
    amount = principal * (1 + (rate / n))**(n * time)
    return amount

result = calculate_compound_interest({principal}, {rate}, {time}, {n})
print(f"Future Value: {{result:.2f}}")
"""
                return code_snippet.strip()
        
        return "print(\"Sorry, I can only calculate compound interest for now. Please ask a relevant question.\")"

    def _execute_code(self, code: str) -> str:
        output_capture = io.StringIO()
        with contextlib.redirect_stdout(output_capture):
            try:
                exec(code, {"__builtins__": None}, {})
            except Exception as e:
                return f"Error during code execution: {e}"
        return output_capture.getvalue().strip()

    def _formulate_advice(self, query: str, calculation_result: str) -> str:
        if "Future Value:" in calculation_result:
            try:
                future_value = calculation_result.split("Future Value: ")[1]
                return f"Based on your query: \"{query}\", your investment is projected to grow to approximately {future_value}. This indicates a solid growth over the period, leveraging the power of compound interest for accurate calculation."
            except IndexError:
                return f"I processed your query: \"{query}\". The calculation resulted in: {calculation_result}. Please rephrase if the output is not clear."
        elif "Error during code execution" in calculation_result:
            return f"I encountered an error while processing your request: {calculation_result}. Please check your input."
        else:
            return f"I processed your query: \"{query}\". The result was: {calculation_result}. For more detailed financial advice, please provide specific parameters or consult a human expert."

    def process_query(self, query: str) -> str:
        generated_code = self._generate_code_from_query(query)
        execution_output = self._execute_code(generated_code)
        final_advice = self._formulate_advice(query, execution_output)
        return final_advice
