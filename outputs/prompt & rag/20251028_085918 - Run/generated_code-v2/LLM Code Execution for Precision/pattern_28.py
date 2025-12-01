
import io
import sys

def generate_financial_code(query):
    """
    Simulates an LLM generating Python code for financial calculations.
    In a real PAL setup, an actual LLM would generate this code dynamically.
    """
    query_lower = query.lower()
    if "portfolio optimization" in query_lower:
        code = """
# Simplified simulation of portfolio optimization
# In a real application, this would involve fetching real market data
# and using libraries like NumPy/Pandas for complex calculations.

asset_returns_daily = [0.0005, 0.0007, 0.0004] # Example average daily returns for 3 assets
asset_risks_daily = [0.008, 0.012, 0.007]   # Example average daily risk (std dev proxy)
weights = [1/3, 1/3, 1/3]                     # Equal weights
num_trading_days_year = 252

# Calculate weighted average return and annualize
portfolio_return_annual = sum(r * w for r, w in zip(asset_returns_daily, weights)) * num_trading_days_year
# Calculate weighted average risk (very simplified, not true portfolio std dev) and annualize
portfolio_risk_annual = sum(r * w for r, w in zip(asset_risks_daily, weights)) * (num_trading_days_year**0.5) # Approximate annualization

print(f"Calculated Annualized Portfolio Return (Simplified): {portfolio_return_annual:.4f}")
print(f"Calculated Annualized Portfolio Risk (Simplified): {portfolio_risk_annual:.4f}")
"""
    elif "risk assessment" in query_lower and "stock" in query_lower:
        stock_symbol = "N/A" # Placeholder, ideally extracted from query
        # A simple way to extract a potential stock symbol
        words = query_lower.split()
        try:
            stock_index = words.index("stock")
            if stock_index + 1 < len(words):
                stock_symbol = words[stock_index + 1].upper().replace('?', '').replace('.', '')
        except ValueError:
            pass

        code = f"""
# Simplified simulation of stock risk (volatility)
# In a real scenario, this would involve fetching historical data via an API
# and calculating standard deviation.

simulated_annualized_volatility = 0.20 + (hash('{stock_symbol}') % 10) / 100.0 # Example placeholder value

print(f"Annualized Volatility for {stock_symbol} (Simplified): {simulated_annualized_volatility:.4f}")
"""
    elif "future value" in query_lower:
        # Simple extraction of numbers for demonstration, robust parsing needed in real app
        principal = 10000.0 # Default example initial investment
        rate = 0.05         # Default example annual interest rate (5%)
        years = 10          # Default example number of years
        
        # Attempt to parse numbers from query
        import re
        numbers = [float(s) for s in re.findall(r'\d+\.?\d*', query)]
        if len(numbers) >= 3:
            principal = numbers[0]
            rate = numbers[1] / 100.0 if numbers[1] > 1 else numbers[1] # Assume if >1, it's a percentage
            years = int(numbers[2])
        elif len(numbers) == 2:
            principal = numbers[0]
            rate = numbers[1] / 100.0 if numbers[1] > 1 else numbers[1]
        elif len(numbers) == 1:
            principal = numbers[0]

        code = f"""
principal = {principal}
rate = {rate}
years = {years}

future_value = principal * (1 + rate)**years
print(f"Future Value after {years} years: {future_value:.2f}")
"""
    else:
        code = "print('Sorry, I cannot generate code for this specific financial query yet, or it would require external libraries not supported in this simplified demo.')"
    return code

def execute_generated_code(code):
    """
    Executes the dynamically generated Python code in a controlled environment
    and captures its standard output.
    """
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    # Restrict built-ins to a safe subset for execution
    safe_builtins = {
        'print': print, 'len': len, 'range': range,
        'float': float, 'int': int, 'str': str,
        'list': list, 'dict': dict, 'tuple': tuple,
        'set': set, 'sum': sum, 'max': max, 'min': min,
        'abs': abs, 'round': round, 'zip': zip, 're': __import__('re') # Include re if needed for parsing within exec
    }
    
    try:
        # Execute the code. Use a restricted global dictionary.
        exec(code, {'__builtins__': safe_builtins})
        output = redirected_output.getvalue()
    except Exception as e:
        output = f"Error during code execution: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout # Restore stdout
    return output

def interpret_and_advise(query, code_output):
    """
    Interprets the numerical output from the executed code and formulates
    natural language financial advice.
    """
    advice = "Based on your query and the computations:\n\n"
    advice += f"Computational results:\n{code_output}\n"

    query_lower = query.lower()

    # Simple interpretation logic based on expected output patterns
    if "portfolio optimization" in query_lower:
        if "Annualized Portfolio Return" in code_output and "Annualized Portfolio Risk" in code_output:
            try:
                return_str = code_output.split("Annualized Portfolio Return (Simplified): ")[1].split("\n")[0]
                risk_str = code_output.split("Annualized Portfolio Risk (Simplified): ")[1].split("\n")[0]
                return_val = float(return_str)
                risk_val = float(risk_str)
                advice += f"This simplified portfolio indicates an annualized return of {return_val*100:.2f}% and an annualized risk (standard deviation proxy) of {risk_val*100:.2f}%. Please remember these are highly simplified figures. In a real scenario, actual market data and advanced financial models would be used for precise optimization and risk assessment."
            except (ValueError, IndexError):
                advice += "Could not parse simplified numerical results for portfolio optimization from code output."
        else:
            advice += "The simplified portfolio optimization code was executed, but expected metrics were not found in the output. Please review the output above."
    elif "risk assessment" in query_lower and "stock" in query_lower:
        if "Annualized Volatility for" in code_output:
            try:
                # Extract stock symbol and volatility from the output string
                match = re.search(r"Annualized Volatility for (.*?)\(Simplified\): (\\d+\\.\\d+)", code_output)
                if match:
                    symbol = match.group(1)
                    volatility_val = float(match.group(2))
                    advice += f"For {symbol}, the simplified annualized volatility is estimated at {volatility_val*100:.2f}%. This is a basic indicator of price fluctuation. For a comprehensive risk assessment, detailed historical data and financial metrics are essential."
                else:
                    advice += "Could not parse simplified numerical results for stock risk assessment from code output."
            except (ValueError, IndexError):
                advice += "Could not parse simplified numerical results for stock risk assessment from code output."
        else:
            advice += "The simplified stock risk assessment code was executed, but expected volatility metrics were not found in the output. Please review the output above."
    elif "future value" in query_lower:
        if "Future Value after" in code_output:
            try:
                fv_match = re.search(r"Future Value after .*? years: (\\d+\\.\\d+)", code_output)
                if fv_match:
                    fv_val = float(fv_match.group(1))
                    advice += f"Based on your inputs, your investment is projected to grow to approximately ${fv_val:.2f} after the specified period. This is a powerful concept for long-term financial planning."
                else:
                    advice += "Could not parse simplified numerical results for future value from code output."
            except (ValueError, IndexError):
                advice += "Could not parse simplified numerical results for future value from code output."
        else:
            advice += "The future value code was executed, but the result was not clearly identified in the output. Please review the output above."
    else:
        advice += "I've executed the code, but my interpretation capabilities for this specific result are limited. Please review the raw output."

    return advice

def financial_advisor_assistant():
    """
    Main function to run the interactive PAL Financial Advisory Assistant.
    """
    print("\nWelcome to the PAL Financial Advisory Assistant (Simplified Demo)! ")
    print("I can assist with basic portfolio optimization, stock risk assessment, and future value calculations by generating and executing Python code.")
    print("**Important:** Due to demonstration constraints, calculations are simplified and do not use external financial libraries or real-time data.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nHow can I assist you with your finances today? ")
        if user_query.lower() == 'exit':
            break

        print("\nThinking and generating code based on your query...")
        generated_code = generate_financial_code(user_query)
        print("\n--- Generated Python Code ---")
        print(generated_code)
        print("-----------------------------")

        print("\nExecuting generated code in a controlled environment...")
        execution_output = execute_generated_code(generated_code)
        print("\n--- Code Execution Output ---")
        print(execution_output)
        print("-----------------------------")

        print("\nInterpreting results and formulating financial advice...")
        final_advice = interpret_and_advise(user_query, execution_output)
        print("\n--- Financial Advice ---")
        print(final_advice)
        print("------------------------")

if __name__ == "__main__":
    financial_advisor_assistant()
