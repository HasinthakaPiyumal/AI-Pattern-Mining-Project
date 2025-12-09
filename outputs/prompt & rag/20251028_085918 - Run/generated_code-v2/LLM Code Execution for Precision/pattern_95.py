
import io
import contextlib
import re

class FinancialAssistant:
    def ask_financial_question(self, question: str) -> str:
        """
        Processes a financial question, generates and executes code, and formulates a natural language response.
        """
        print(f"User Question: {question}")

        # Step 1: Simulate LLM generating Python code
        generated_code = self._generate_python_code(question)
        print(f"\nGenerated Python Code:\n{generated_code}")

        if not generated_code:
            return "I couldn't generate a suitable program for your question. Please try rephrasing or ask a question related to CAGR, Simple Interest, or Future Value."

        # Step 2: Execute the generated Python code
        execution_output, error = self._execute_code(generated_code)
        print(f"\nCode Execution Output:\n{execution_output}")
        if error:
            print(f"Code Execution Error:\n{error}")
            return f"An error occurred during calculation: {error}\nI attempted to use the following code:\n```python\n{generated_code.strip()}\n```\nPlease check the input values or the nature of the question."

        # Step 3: Simulate LLM formulating the final response
        final_response = self._formulate_response(question, generated_code, execution_output)
        return final_response

    def _generate_python_code(self, question: str) -> str:
        """
        Simulates an LLM generating Python code based on the financial question.
        In a real PAL system, this would be an actual LLM call to synthesize code.
        """
        question_lower = question.lower()

        if "cagr" in question_lower and "initial investment" in question_lower and "final value" in question_lower and "years" in question_lower:
            # Example: "Calculate the CAGR for an initial investment of 1000, final value of 2000 over 5 years."
            try:
                # Extract numbers - very basic parsing for demonstration. Assumes order: initial, final, years
                numbers = [float(s) for s in re.findall(r'-?\d+\.?\d*', question)]
                if len(numbers) >= 3:
                    initial_value, final_value, years = numbers[0], numbers[1], numbers[2]
                    return f"""
initial_value = {initial_value}
final_value = {final_value}
years = {years}

if years == 0:
    raise ValueError("Number of years cannot be zero for CAGR calculation.")
cagr = ((final_value / initial_value) ** (1 / years)) - 1
result = cagr * 100
print(f"CAGR: {{result:.2f}}%")
"""
            except Exception as e:
                print(f"Error parsing CAGR question or generating code: {e}")
                return ""
        elif "simple interest" in question_lower and "principal" in question_lower and "rate" in question_lower and "time" in question_lower:
            # Example: "What is the simple interest on a principal of 5000 at a rate of 3% for 2 years?"
            try:
                # Assuming order: principal, rate, time
                numbers = [float(s) for s in re.findall(r'-?\d+\.?\d*', question)]
                if len(numbers) >= 3:
                    principal, rate, time = numbers[0], numbers[1], numbers[2]
                    rate_decimal = rate / 100  # Convert percentage to decimal
                    return f"""
principal = {principal}
rate = {rate_decimal}
time = {time}
simple_interest = principal * rate * time
result = simple_interest
print(f"Simple Interest: ${{result:.2f}}")
"""
            except Exception as e:
                print(f"Error parsing Simple Interest question or generating code: {e}")
                return ""
        elif "future value" in question_lower and "present value" in question_lower and "rate" in question_lower and "periods" in question_lower:
             # Example: "Calculate the future value of 1000 with an annual rate of 5% over 10 periods, compounded annually."
            try:
                # Assuming order: present value, rate, periods
                numbers = [float(s) for s in re.findall(r'-?\d+\.?\d*', question)]
                if len(numbers) >= 3:
                    present_value, rate, periods = numbers[0], numbers[1], numbers[2]
                    rate_decimal = rate / 100
                    return f"""
present_value = {present_value}
rate = {rate_decimal}
periods = {periods}
future_value = present_value * (1 + rate) ** periods
result = future_value
print(f"Future Value: ${{result:.2f}}")
"""
            except Exception as e:
                print(f"Error parsing Future Value question or generating code: {e}")
                return ""
        return "" # No matching pattern

    def _execute_code(self, code: str):
        """
        Executes the given Python code in a sandboxed environment and captures output and errors.
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        local_scope = {}
        error = None

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                # Use exec with a limited scope for safety
                exec(code, {"__builtins__": {}}, local_scope)
        except Exception as e:
            error = str(e)
            stderr_content = stderr_capture.getvalue()
            if stderr_content:
                error += f"\nStderr: {stderr_content.strip()}"
        
        return stdout_capture.getvalue(), error

    def _formulate_response(self, question: str, generated_code: str, execution_output: str) -> str:
        """
        Simulates an LLM formulating a natural language response based on the question and code execution result.
        In a real PAL system, this would be an actual LLM call that incorporates the numerical result.
        """
        response = f"Based on your question: '{question}', I performed a financial calculation.\n"
        if execution_output.strip():
            response += f"The result of the calculation is: {execution_output.strip()}\n"
        else:
            response += "The calculation was performed, but no direct output was captured from the code's print statements. This might indicate an issue or that the code did not produce a printable result.\n"
        response += "This was achieved by generating and executing the following Python code:\n"
        response += "```python\n"
        response += generated_code.strip() + "\n"
        response += "```\n"
        response += "Please let me know if you have other financial questions requiring precise calculations."
        return response

if __name__ == "__main__":
    assistant = FinancialAssistant()

    print("--- Test Case 1: CAGR Calculation ---")
    cagr_question = "Calculate the Compound Annual Growth Rate for an initial investment of 1000, which grew to 2000 over 5 years."
    print(assistant.ask_financial_question(cagr_question))
    print("\n" + "="*80 + "\n")

    print("--- Test Case 2: Simple Interest Calculation ---")
    simple_interest_question = "What is the simple interest on a principal of 5000 at an annual rate of 3% for 2 years?"
    print(assistant.ask_financial_question(simple_interest_question))
    print("\n" + "="*80 + "\n")

    print("--- Test Case 3: Future Value Calculation ---")
    future_value_question = "Calculate the future value of 1000 with an annual rate of 5% over 10 periods, compounded annually."
    print(assistant.ask_financial_question(future_value_question))
    print("\n" + "="*80 + "\n")

    print("--- Test Case 4: Unrecognized Question ---")
    unrecognized_question = "What is the capital gains tax on selling a stock at 150 that was bought at 100?"
    print(assistant.ask_financial_question(unrecognized_question))
    print("\n" + "="*80 + "\n")

    print("--- Test Case 5: Code with a Calculation Error (e.g., division by zero for CAGR) ---")
    error_question = "Calculate the CAGR for an initial investment of 1000, final value of 2000 over 0 years."
    print(assistant.ask_financial_question(error_question))
    print("\n" + "="*80 + "\n")
