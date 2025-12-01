import math

class UserInterface:
    def get_query(self):
        return input("\nEnter your financial query: ")

    def display_response(self, response):
        print(f"\nFinancial Advisor: {response}")

class CodeExecutor:
    def execute_code(self, code_str, context=None):
        if context is None:
            context = {}
        try:
            exec(code_str, globals(), context)
            return context.get('result')
        except Exception as e:
            return f"Error during code execution: {e}"

class LLMSimulator:
    def __init__(self, code_executor):
        self.code_executor = code_executor

    def generate_and_execute_code(self, query):
        generated_code = self._generate_code_from_query(query)
        if generated_code:
            print(f"\n(Simulated LLM generating code:\n{generated_code}\n)")
            execution_result = self.code_executor.execute_code(generated_code)
            return self._formulate_response(query, execution_result)
        else:
            return "I'm sorry, I cannot generate specific code for that request at the moment. Please ask about compound interest or simple portfolio allocation."

    def _generate_code_from_query(self, query):
        query_lower = query.lower()
        if "compound interest" in query_lower or "future value" in query_lower:
            # Example: "Calculate compound interest for $1000 at 5% for 10 years." 
            # This simulation extracts numbers from a hypothetical query.
            # In a real PAL system, the LLM would intelligently parse and generate variables.
            principal = 1000  # Default values for simulation
            rate = 0.05
            time = 10
            
            # Simple parsing for demonstration
            import re
            principal_match = re.search(r'\$(\d+)', query)
            if principal_match: principal = float(principal_match.group(1))
            rate_match = re.search(r'(\d+\.?\d*)%', query)
            if rate_match: rate = float(rate_match.group(1)) / 100
            time_match = re.search(r'(\d+)\s+year', query)
            if time_match: time = int(time_match.group(1))

            return f"""P = {principal}
r = {rate}
t = {time}
A = P * (1 + r)**t
interest = A - P
result = {{'amount': A, 'interest': interest}}"""
        elif "portfolio allocation" in query_lower or "optimize portfolio" in query_lower:
            # Simple two-asset portfolio simulation (e.g., Stock A, Bond B)
            # In a real scenario, this would involve fetching real data, optimization libraries.
            return """import numpy as np\n\n# Simulated returns and standard deviations for two assets\nexpected_return_asset1 = 0.12\nstd_dev_asset1 = 0.15\nexpected_return_asset2 = 0.06\nstd_dev_asset2 = 0.08\ncorrelation = 0.3\n\n# Assuming equal weights for simplicity in this simulation (0.5, 0.5)\nw1 = 0.5\nw2 = 0.5\n\nportfolio_return = w1 * expected_return_asset1 + w2 * expected_return_asset2\nportfolio_variance = (w1**2 * std_dev_asset1**2) + (w2**2 * std_dev_asset2**2) + (2 * w1 * w2 * std_dev_asset1 * std_dev_asset2 * correlation)\nportfolio_std_dev = np.sqrt(portfolio_variance)\n\nresult = {{'portfolio_return': portfolio_return, 'portfolio_risk': portfolio_std_dev}}"""
        return None

    def _formulate_response(self, query, execution_result):
        if isinstance(execution_result, str) and execution_result.startswith("Error"):
            return f"I encountered an error while processing your request: {execution_result}"
        
        query_lower = query.lower()
        if "compound interest" in query_lower and isinstance(execution_result, dict):
            amount = execution_result.get('amount')
            interest = execution_result.get('interest')
            if amount is not None and interest is not None:
                return f"Based on your query, the future value of your investment will be ${amount:.2f}, earning ${interest:.2f} in compound interest."
        elif "portfolio allocation" in query_lower and isinstance(execution_result, dict):
            portfolio_return = execution_result.get('portfolio_return')
            portfolio_risk = execution_result.get('portfolio_risk')
            if portfolio_return is not None and portfolio_risk is not None:
                return f"For a simulated balanced portfolio, the expected annual return is {portfolio_return:.2%} with an annual risk (standard deviation) of {portfolio_risk:.2%}."
        
        return f"I processed your request, and the result was: {execution_result}. How else can I assist you?"

def main():
    ui = UserInterface()
    executor = CodeExecutor()
    llm_advisor = LLMSimulator(executor)

    print("Welcome to the PAL Financial Advisor Simulation!")
    print("You can ask questions like: ")
    print("- Calculate compound interest for $1000 at 5% for 10 years.")
    print("- What is the expected return and risk for a simple portfolio allocation?")
    print("Type 'exit' to quit.")

    while True:
        user_query = ui.get_query()
        if user_query.lower() == 'exit':
            print("Thank you for using the PAL Financial Advisor. Goodbye!")
            break
        
        response = llm_advisor.generate_and_execute_code(user_query)
        ui.display_response(response)

if __name__ == "__main__":
    main()
