import io
import contextlib
import json

# Simulate an LLM generating Python code based on a prompt
def simulate_llm_code_generation(user_prompt):
    """
    Simulates an LLM generating Python code for financial calculations.
    In a real application, this would be an actual LLM API call.
    """
    if "moderate risk" in user_prompt.lower() and "$10,000" in user_prompt:
        generated_code = """
def optimize_portfolio(capital):
    # Simplified allocation for moderate risk: 60% stocks, 30% bonds, 10% cash
    stocks = capital * 0.60
    bonds = capital * 0.30
    cash = capital * 0.10
    
    print(json.dumps({
        "stocks": round(stocks, 2),
        "bonds": round(bonds, 2),
        "cash": round(cash, 2),
        "total_invested": round(stocks + bonds + cash, 2)
    }))

optimize_portfolio(10000)
"""
    elif "high risk" in user_prompt.lower() and "$50,000" in user_prompt:
        generated_code = """
def optimize_portfolio(capital):
    # Simplified allocation for high risk: 80% stocks, 15% bonds, 5% cash
    stocks = capital * 0.80
    bonds = capital * 0.15
    cash = capital * 0.05
    
    print(json.dumps({
        "stocks": round(stocks, 2),
        "bonds": round(bonds, 2),
        "cash": round(cash, 2),
        "total_invested": round(stocks + bonds + cash, 2)
    }))

optimize_portfolio(50000)
"""
    else:
        generated_code = """
# No specific optimization code generated for this prompt.
# A real LLM would generate more dynamic code based on varied inputs.
print(json.dumps({"error": "Could not generate specific portfolio optimization code for the given prompt."}))
"""
    return generated_code

# Execute the generated Python code
def execute_generated_code(code_string):
    """
    Executes the given Python code string and captures its standard output.
    """
    # Redirect stdout to capture print statements
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            # Using a separate dictionary for exec locals to avoid interfering with global scope
            exec_globals = {"json": json} # Provide json module if the generated code uses it
            exec(code_string, exec_globals)
        except Exception as e:
            return {"error": f"Error during code execution: {e}"}
    
    output = f.getvalue()
    try:
        # Assuming the generated code prints a JSON object
        return json.loads(output)
    except json.JSONDecodeError:
        return {"raw_output": output.strip(), "error": "Could not parse JSON output from executed code."}

# Generate natural language investment advice based on calculation results
def generate_advice(optimization_results, user_prompt):
    """
    Formulates natural language investment advice based on the numerical results
    from the executed code.
    """
    if "error" in optimization_results:
        return f"I encountered an issue: {optimization_results['error']}. Please try refining your request."
    
    if "stocks" in optimization_results:
        advice = f"Based on your request for {user_prompt}, here is a suggested portfolio allocation:\n\n"
        advice += f"- **Stocks**: ${optimization_results['stocks']:.2f}\n"
        advice += f"- **Bonds**: ${optimization_results['bonds']:.2f}\n"
        advice += f"- **Cash**: ${optimization_results['cash']:.2f}\n\n"
        advice += f"This allocation totals ${optimization_results['total_invested']:.2f}. "
        advice += "This is a simplified recommendation; always consult a financial advisor for personalized plans."
        return advice
    else:
        return "I processed your request, but the optimization results were not in the expected format to generate specific advice."

# Main application flow
def main():
    print("Welcome to the PAL Financial Portfolio Optimizer!")
    print("This tool simulates an LLM generating and executing code for investment recommendations.")
    
    while True:
        user_input = input("\nEnter your investment query (e.g., 'Suggest a portfolio for a moderate risk investor with $10,000.', or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        print("\n--- Simulating LLM Code Generation ---")
        generated_code = simulate_llm_code_generation(user_input)
        print("Generated Code:\n```python")
        print(generated_code.strip())
        print("```")
        
        print("\n--- Executing Generated Code ---")
        execution_results = execute_generated_code(generated_code)
        print(f"Execution Results: {execution_results}")
        
        print("\n--- Generating Investment Advice ---")
        investment_advice = generate_advice(execution_results, user_input)
        print(investment_advice)

if __name__ == "__main__":
    main()