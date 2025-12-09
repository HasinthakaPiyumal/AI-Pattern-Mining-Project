import math

def code_generation_module(query):
    query_lower = query.lower()
    generated_code = ""
    if "future value" in query_lower and "investment" in query_lower:
        # Simulate parsing numbers from query - simplified
        try:
            principal = float(input("Enter initial investment (principal): "))
            rate = float(input("Enter annual interest rate (e.g., 0.05 for 5%): "))
            years = int(input("Enter number of years: "))
            compounding_periods = int(input("Enter compounding periods per year (e.g., 1 for annually, 12 for monthly): "))

            generated_code = f"""
result = principal * math.pow((1 + (rate / compounding_periods)), (compounding_periods * years))
final_result = f"The future value of your investment is: {{result:.2f}}"
"""
        except ValueError:
            generated_code = "print(\"Could not parse numerical inputs for future value calculation. Please provide valid numbers.\")"

    elif "risk and return" in query_lower and "portfolio" in query_lower:
        # Simulate some basic risk/return calculation
        try:
            num_assets = int(input("Enter number of assets in portfolio: "))
            total_return = 0
            total_risk = 0
            for i in range(num_assets):
                asset_return = float(input(f"Enter expected return for asset {i+1} (e.g., 0.1 for 10%): "))
                asset_risk = float(input(f"Enter risk (standard deviation) for asset {i+1} (e.g., 0.15): "))
                total_return += asset_return
                total_risk += asset_risk # Simplified, not actual portfolio risk
            
            avg_return = total_return / num_assets if num_assets > 0 else 0
            avg_risk = total_risk / num_assets if num_assets > 0 else 0

            generated_code = f"""
result = {{'avg_return': {avg_return}, 'avg_risk': {avg_risk}}}
final_result = f"Your portfolio\'s average expected return is {{result['avg_return']:.2%}} and average risk is {{result['avg_risk']:.2f}} (standard deviation)."
"""
        except ValueError:
            generated_code = "print(\"Could not parse numerical inputs for portfolio analysis. Please provide valid numbers.\")"

    elif "budget planning" in query_lower or "monthly expenses" in query_lower:
        try:
            income = float(input("Enter your monthly income: "))
            expenses_input = input("Enter your monthly expenses (comma-separated, e.g., 500,200,150): ")
            expenses = [float(e.strip()) for e in expenses_input.split(',') if e.strip()]
            total_expenses = sum(expenses)
            savings = income - total_expenses

            generated_code = f"""
result = {{'income': {income}, 'total_expenses': {total_expenses}, 'savings': {savings}}}
final_result = f"Your monthly income is ${{result['income']:.2f}}, total expenses are ${{result['total_expenses']:.2f}}, resulting in monthly savings of ${{result['savings']:.2f}}."
"""
        except ValueError:
            generated_code = "print(\"Could not parse numerical inputs for budget planning. Please provide valid numbers.\")"
    
    else:
        generated_code = "final_result = \"I cannot generate a specific program for that financial query yet. Please try a query related to future value, portfolio analysis, or budget planning.\""

    return generated_code

def code_execution_environment(code_string):
    local_vars = {'math': math}
    try:
        exec(code_string, {}, local_vars)
        return local_vars.get('final_result', 'No specific result variable set by the executed code.')
    except Exception as e:
        return f"Error during code execution: {e}"

def response_generation_module(original_query, execution_result):
    if "No specific result variable set" in execution_result or "Error during code execution" in execution_result:
        return f"I encountered an issue processing your request: {execution_result}"
    else:
        return f"Based on your query: \"{original_query}\"\nMy analysis indicates: {execution_result}"

def llm_orchestrator(user_query):
    print(f"\nProcessing your query: '{user_query}'...")
    generated_code = code_generation_module(user_query)
    print("\nGenerated Code (Simulated):\n---\n" + generated_code + "\n---")
    execution_output = code_execution_environment(generated_code)
    response = response_generation_module(user_query, execution_output)
    return response

if __name__ == "__main__":
    print("Welcome to the Smart Financial Advisor (Simulated PAL LLM)!")
    print("You can ask about 'future value of investment', 'risk and return of portfolio', or 'budget planning'.\n")

    while True:
        user_input = input("Your financial query (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        final_answer = llm_orchestrator(user_input)
        print("\nAdvisor's Response:\n" + final_answer + "\n")

    print("Thank you for using the Smart Financial Advisor. Goodbye!")
