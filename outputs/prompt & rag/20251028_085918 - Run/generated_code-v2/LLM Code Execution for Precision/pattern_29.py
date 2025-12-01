import json
import sys
import io

# Assuming prompts.py and portfolio_optimizer.py are in the same directory
from prompts import SYSTEM_PROMPT_CODE_GENERATION, SYSTEM_PROMPT_NATURAL_LANGUAGE_EXPLANATION

# This is a placeholder for an actual LLM API call
def call_llm(prompt: str) -> str:
    """
    Simulates an LLM API call. In a real application, this would interact with a service
    like OpenAI GPT, Google Gemini, etc., to get a response.
    For demonstration, it will return a predefined code snippet.
    """
    print("\n--- Simulating LLM for Code Generation ---")
    # In a real scenario, this prompt would be sent to the LLM and it would return Python code.
    # For this demonstration, we'll return a hardcoded example of generated code.
    if "SYSTEM_PROMPT_CODE_GENERATION" in prompt:
        return """import numpy as np
import pandas as pd
from portfolio_optimizer import calculate_portfolio_metrics, black_scholes_call_price

def run_analysis():
    # Example user input parameters (these would come from a user interface)
    expected_returns = np.array([0.08, 0.12, 0.05]) # e.g., for Stocks, Bonds, Real Estate
    volatilities = np.array([0.15, 0.08, 0.10])
    weights = np.array([0.6, 0.3, 0.1]) # Initial allocation

    # Calculate initial portfolio metrics
    initial_portfolio_return, initial_portfolio_volatility = calculate_portfolio_metrics(expected_returns, volatilities, weights)

    # Simulate a more optimal allocation (e.g., from a more advanced optimization algorithm not shown here)
    # For a real PAL, the LLM might try different weights or call a dedicated optimizer library.
    optimized_weights = np.array([0.5, 0.35, 0.15])
    optimized_portfolio_return, optimized_portfolio_volatility = calculate_portfolio_metrics(expected_returns, volatilities, optimized_weights)

    # Example: Black-Scholes for an option (if user holds options)
    stock_price = 100
    strike_price = 105
    time_to_expiration = 0.5
    risk_free_rate = 0.05
    stock_volatility = 0.2
    option_price = black_scholes_call_price(stock_price, strike_price, time_to_expiration, risk_free_rate, stock_volatility)

    return {
        "initial_portfolio_return": initial_portfolio_return,
        "initial_portfolio_volatility": initial_portfolio_volatility,
        "optimized_portfolio_return": optimized_portfolio_return,
        "optimized_portfolio_volatility": optimized_portfolio_volatility,
        "recommended_weights": optimized_weights.tolist(),
        "option_value_example": option_price
    }
"""
    elif "SYSTEM_PROMPT_NATURAL_LANGUAGE_EXPLANATION" in prompt:
        print("\n--- Simulating LLM for Natural Language Explanation ---")
        # In a real scenario, the LLM would analyze the 'analysis_results_json' and generate text.
        # For this demonstration, we'll return a hardcoded explanation.
        return """Based on our analysis, your initial portfolio shows an expected annual return of approximately 9.3% with a volatility of 9.2%. 

We recommend adjusting your portfolio to achieve an expected annual return of 9.75% with a slightly increased volatility of 9.4%. This adjustment involves shifting your allocation to 50% Stocks, 35% Bonds, and 15% Real Estate. This modest change aims to improve your returns while maintaining a manageable risk profile. 

For any options in your portfolio, an example calculation shows a call option value of $3.68, which can be used for valuation purposes. 

Always consider your personal financial situation and consult with a professional advisor before making investment decisions.
"""
    return ""

def execute_generated_code(code: str) -> dict:
    """
    Executes the generated Python code in a sandboxed environment and captures its output.
    In a real system, this would involve a secure execution environment (e.g., Docker container, `exec` with restrictions).
    For this demonstration, we'll use a basic `exec` with output capturing.
    """
    print("\n--- Executing Generated Python Code ---")
    execution_globals = {'__builtins__': {}}
    execution_locals = {}
    
    # Redirect stdout to capture print statements (if any, though none expected from run_analysis)
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    try:
        # Execute the code, making portfolio_optimizer functions available if needed by the LLM generated code
        exec(code, {'np': np, 'pd': pd, 'calculate_portfolio_metrics': portfolio_optimizer.calculate_portfolio_metrics, 'black_scholes_call_price': portfolio_optimizer.black_scholes_call_price}, execution_locals)
        
        # Call the run_analysis function defined in the executed code
        if 'run_analysis' in execution_locals and callable(execution_locals['run_analysis']):
            results = execution_locals['run_analysis']()
            print(f"Code execution successful. Results: {results}")
            return results
        else:
            print("Error: 'run_analysis' function not found or not callable in generated code.")
            return {"error": "'run_analysis' function not found or not callable."}
    except Exception as e:
        print(f"Error during code execution: {e}")
        return {"error": str(e), "stdout": redirected_output.getvalue()}
    finally:
        sys.stdout = old_stdout # Restore stdout

def main():
    print("\n--- Financial Portfolio Optimizer with PAL Prompting ---")
    print("Please provide your financial goals, risk tolerance, and current holdings.")
    print("Example: 'I want to grow my wealth aggressively with a high risk tolerance. My current portfolio is 70% stocks, 20% bonds, 10% real estate.'")
    user_input = input("Your input: ")

    # Step 1: LLM generates Python code for financial analysis
    code_generation_prompt = f"{SYSTEM_PROMPT_CODE_GENERATION}\n\nUser Input: {user_input}"
    generated_code = call_llm(code_generation_prompt)
    print(f"\nGenerated Code:\n```python\n{generated_code}\n```")

    # Step 2: Execute the generated code
    analysis_results = execute_generated_code(generated_code)
    
    if "error" in analysis_results:
        print(f"\nError in analysis: {analysis_results['error']}")
        return

    # Step 3: LLM generates natural language explanation and recommendations
    analysis_results_json = json.dumps(analysis_results, indent=2)
    explanation_prompt = f"{SYSTEM_PROMPT_NATURAL_LANGUAGE_EXPLANATION}\n\nFinancial Analysis Results:\n{analysis_results_json}\n\nUser Input: {user_input}"
    final_recommendation = call_llm(explanation_prompt)
    
    print("\n--- Final Financial Recommendation --- ")
    print(final_recommendation)

if __name__ == "__main__":
    # Ensure numpy is available for the simulated execution environment if LLM uses it.
    # In a real setup, dependencies would be managed.
    try:
        import numpy as np
        import pandas as pd # pandas is often used with numpy for financial data
    except ImportError:
        print("Error: numpy and pandas are required. Please install them using 'pip install numpy pandas'.")
        sys.exit(1)
    
    # The portfolio_optimizer module needs to be available in the execution environment.
    # Since it's a local file, it will be imported directly in this script's scope
    # and then passed into the exec's globals for the generated code to access.

    main()