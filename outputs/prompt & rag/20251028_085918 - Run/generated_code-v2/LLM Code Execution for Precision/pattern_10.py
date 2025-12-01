
from fastapi import FastAPI
from pydantic import BaseModel
import io
import contextlib
import time

# --- models.py ---

class FinancialAdviceRequest(BaseModel):
    query: str

class FinancialAdviceResponse(BaseModel):
    advice: str

# --- financial_calculators.py ---

class FinancialCalculators:
    @staticmethod
    def future_value(principal: float, rate: float, periods: int) -> float:
        return principal * (1 + rate)**periods

    @staticmethod
    def present_value(future_value: float, rate: float, periods: int) -> float:
        return future_value / (1 + rate)**periods

    @staticmethod
    def loan_payment(principal: float, annual_rate: float, years: int) -> float:
        monthly_rate = annual_rate / 12
        n_payments = years * 12
        if monthly_rate == 0:
            return principal / n_payments
        return principal * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)

    @staticmethod
    def calculate_tax(income: float, tax_rate: float) -> float:
        return income * tax_rate

# --- code_executor.py ---

class CodeExecutionEngine:
    def execute_code(self, code: str, globals_dict: dict, timeout: int = 5) -> (str, str):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        local_vars = {}
        exec_globals = {
            **globals_dict,
            "__builtins__": {},
        }

        start_time = time.time()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    exec(code, exec_globals, local_vars)
            execution_output = stdout_capture.getvalue().strip()
            error_output = stderr_capture.getvalue().strip()
            return execution_output, error_output
        except Exception as e:
            return "", str(e)

# --- llm_service.py ---

class LLMService:
    def __init__(self):
        # In a real application, initialize your LLM client here (e.g., openai.OpenAI())
        pass

    def _construct_prompt(self, user_query: str) -> str:
        # This is a simplified prompt. A real prompt would be much more detailed
        # and include examples and specific function signatures.
        available_functions = """
        You have access to the following financial calculation functions:
        - FinancialCalculators.future_value(principal: float, rate: float, periods: int) -> float
        - FinancialCalculators.present_value(future_value: float, rate: float, periods: int) -> float
        - FinancialCalculators.loan_payment(principal: float, annual_rate: float, years: int) -> float
        - FinancialCalculators.calculate_tax(income: float, tax_rate: float) -> float

        Your task is to generate Python code to solve the user's financial query, then provide a natural language explanation of the result.
        Ensure your code prints the final numerical result using 'print()'.
        Respond in the format: <explanation>\n```python\n<code>\n```
        """
        return f"""{available_functions}
        User query: {user_query}
        """

    def _mock_llm_call(self, prompt: str) -> str:
        # Placeholder for actual LLM API call
        # This mock simulates an LLM response with code and explanation.
        if "future value of" in prompt.lower():
            return "The future value of your investment is calculated below.\n```python\nprint(FinancialCalculators.future_value(principal=1000, rate=0.05, periods=10))\n```"
        elif "monthly payment" in prompt.lower():
            return "Here is your estimated monthly loan payment.\n```python\nprint(FinancialCalculators.loan_payment(principal=200000, annual_rate=0.04, years=30))\n```"
        elif "tax on income" in prompt.lower():
            return "Your estimated tax liability is:\n```python\nprint(FinancialCalculators.calculate_tax(income=60000, tax_rate=0.25))\n```"
        else:
            return "I can help with that. Please specify more details.\n```python\n# No specific calculation code generated for this query.\nprint('No specific calculation performed.')\n```"

    def get_financial_advice(self, user_query: str) -> (str, str):
        prompt = self._construct_prompt(user_query)
        llm_response = self._mock_llm_call(prompt) # Replace with actual LLM API call

        explanation_parts = []
        code_parts = []
        in_code_block = False

        for line in llm_response.splitlines():
            if line.strip() == "```python":
                in_code_block = True
                continue
            elif line.strip() == "```":
                in_code_block = False
                continue

            if in_code_block:
                code_parts.append(line)
            else:
                explanation_parts.append(line)

        explanation = " ".join(explanation_parts).strip()
        code = "\n".join(code_parts).strip()

        return explanation, code

# --- main.py ---

app = FastAPI()
llm_service = LLMService()
code_executor = CodeExecutionEngine()

@app.post("/advise", response_model=FinancialAdviceResponse)
async def get_advice(request: FinancialAdviceRequest):
    initial_explanation, generated_code = llm_service.get_financial_advice(request.query)

    execution_globals = {"FinancialCalculators": FinancialCalculators}
    
    # Execute the generated code
    execution_output, error_output = code_executor.execute_code(generated_code, execution_globals)

    final_advice = initial_explanation
    if execution_output:
        final_advice += f"\n\nCalculation Result: {execution_output}"
    if error_output:
        final_advice += f"\n\nError during calculation: {error_output}"

    return FinancialAdviceResponse(advice=final_advice)

# To run this application:
# 1. Save the code as personal_finance_advisor.py
# 2. Install dependencies: pip install fastapi uvicorn pydantic
# 3. Run from your terminal: uvicorn personal_finance_advisor:app --reload
# 4. Access the API at http://127.0.0.1:8000/docs
