import subprocess
import tempfile
import os
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. Financial Tools/API Integration (Content of financial_tools.py) ---
FINANCIAL_TOOLS_CODE = """
import math

def calculate_future_value(principal: float, rate: float, periods: int) -> float:
    # FV = P * (1 + r)^n
    return principal * (1 + rate)**periods

def calculate_risk_metric(returns: list[float]) -> float:
    if not returns:
        return 0.0
    n = len(returns)
    mean = sum(returns) / n
    variance = sum([(x - mean)**2 for x in returns]) / n
    return math.sqrt(variance) # Standard Deviation

def mock_stock_price(symbol: str) -> float:
    # Mock function to simulate fetching a stock price
    prices = {"AAPL": 170.0, "MSFT": 280.0, "GOOG": 150.0}
    return prices.get(symbol.upper(), 100.0) # Default if not found

def calculate_portfolio_value(holdings: dict) -> float:
    total_value = 0.0
    for symbol, quantity in holdings.items():
        total_value += mock_stock_price(symbol) * quantity
    return total_value

"""

# --- 2. Code Execution Environment ---
def execute_python_code(code_to_execute: str) -> tuple[int, str, str]:
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        tools_file_path = os.path.join(temp_dir, "financial_tools.py")
        script_file_path = os.path.join(temp_dir, "generated_script.py")

        with open(tools_file_path, "w") as f:
            f.write(FINANCIAL_TOOLS_CODE)
        
        with open(script_file_path, "w") as f:
            f.write(code_to_execute)

        process = subprocess.run(
            ["python", script_file_path],
            capture_output=True,
            text=True,
            cwd=temp_dir # Execute in the temporary directory to allow relative imports
        )
        return process.returncode, process.stdout, process.stderr
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# --- 3. LLM Agent (Program Generator & Advisor - Mocked) ---
class LLM_Agent:
    def __init__(self, llm_model_name: str = "mock-llm"):
        self.llm_model_name = llm_model_name

    def generate_code_from_query(self, query: str) -> str:
        # In a real scenario, this would call an actual LLM API
        # and prompt it to generate Python code based on the query.
        # The LLM would be instructed to use functions from 'financial_tools'.
        if "future value" in query.lower():
            # Example: "What's the future value of $10,000 invested at 5% for 10 years?"
            return """
import financial_tools
principal = 10000.0
rate = 0.05
periods = 10
fv = financial_tools.calculate_future_value(principal, rate, periods)
print(f"Future Value: {fv:.2f}")
"""
        elif "risk of a portfolio" in query.lower():
            # Example: "Analyze the risk of a portfolio with Apple and Microsoft stock."
            return """
import financial_tools
# Mock returns for demonstration
returns = [-0.01, 0.02, 0.005, -0.003, 0.015]
risk = financial_tools.calculate_risk_metric(returns)
print(f"Portfolio Risk (Standard Deviation): {risk:.4f}")
"""
        elif "portfolio value" in query.lower():
            # Example: "What is the current value of a portfolio with 10 shares of AAPL and 5 shares of MSFT?"
            return """
import financial_tools
holdings = {"AAPL": 10, "MSFT": 5}
value = financial_tools.calculate_portfolio_value(holdings)
print(f"Current Portfolio Value: {value:.2f}")
"""
        else:
            return "print(\"Could not generate specific code for the query.\")"

    def formulate_advice_from_output(self, query: str, code_output: str) -> str:
        # In a real scenario, this would call an actual LLM API
        # to interpret the code output and provide natural language advice.
        if "future value" in query.lower() and "Future Value:" in code_output:
            fv_str = code_output.split("Future Value:")[1].strip()
            return f"Based on your investment details, the projected future value is {fv_str}. This is a basic calculation and doesn't account for taxes, inflation, or fees."
        elif "risk of a portfolio" in query.lower() and "Portfolio Risk (Standard Deviation):" in code_output:
            risk_str = code_output.split("Portfolio Risk (Standard Deviation):")[1].strip()
            return f"The calculated standard deviation for the portfolio returns is {risk_str}. This indicates the volatility of the portfolio; a higher number suggests higher risk."
        elif "portfolio value" in query.lower() and "Current Portfolio Value:" in code_output:
            value_str = code_output.split("Current Portfolio Value:")[1].strip()
            return f"The estimated current value of your portfolio is {value_str}, based on mocked stock prices. Real-time data and market conditions may vary."
        else:
            return f"I processed your request.\nCode Output: {code_output}\nFurther advice would require more advanced analysis."


# --- 4. User Interface (FastAPI) ---
app = FastAPI(
    title="Financial Advisory PAL Assistant",
    description="A Financial Advisory and Portfolio Optimization Assistant leveraging Program-Aided Language Models (PAL) Prompting."
)

class QueryRequest(BaseModel):
    user_query: str

llm_agent = LLM_Agent()

@app.post("/advise")
async def get_financial_advice(request: QueryRequest):
    try:
        # Step 1: LLM Agent generates code
        generated_code = llm_agent.generate_code_from_query(request.user_query)
        
        # Step 2: Execute the generated code
        return_code, stdout, stderr = execute_python_code(generated_code)

        if return_code != 0:
            # Handle execution errors
            error_message = f"Code execution failed with error: {stderr}"
            print(error_message)
            # Allow LLM to attempt to provide advice even with execution error, or raise HTTPException
            advice = llm_agent.formulate_advice_from_output(request.user_query, f"Error during execution: {stderr}")
            return {"query": request.user_query, "generated_code": generated_code, "code_output": stderr, "advice": advice, "status": "failed"}

        # Step 3: LLM Agent formulates advice from output
        financial_advice = llm_agent.formulate_advice_from_output(request.user_query, stdout)

        return {
            "query": request.user_query,
            "generated_code": generated_code,
            "code_output": stdout,
            "advice": financial_advice,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

