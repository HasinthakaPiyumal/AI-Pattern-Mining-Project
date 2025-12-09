import math

def calculate_compound_interest(principal, annual_rate, years, compounds_per_year):
    rate = annual_rate / 100
    amount = principal * (1 + rate / compounds_per_year)**(compounds_per_year * years)
    return amount

def calculate_future_value(present_value, annual_interest_rate, periods):
    rate = annual_interest_rate / 100
    future_value = present_value * (1 + rate)**periods
    return future_value

def calculate_simple_risk_score(volatility_index, market_sentiment_score):
    # This is a very simplified risk score calculation for demonstration
    risk_score = (volatility_index * 0.7) + (market_sentiment_score * 0.3)
    return round(risk_score, 2)

class PythonInterpreter:
    def execute(self, code_string, context=None):
        if context is None:
            context = {}
        # Add financial functions to the execution context
        context['calculate_compound_interest'] = calculate_compound_interest
        context['calculate_future_value'] = calculate_future_value
        context['calculate_simple_risk_score'] = calculate_simple_risk_score

        output = {}
        try:
            exec(code_string, globals(), context)
            output['result'] = context.get('result', 'No explicit result defined in code.')
            output['error'] = None
        except Exception as e:
            output['result'] = None
            output['error'] = str(e)
        return output

class FinancialLLMSimulator:
    def generate_code_and_response(self, user_query, executed_results):
        # In a real PAL system, an LLM would dynamically generate this code
        # and the natural language response.
        # This is a highly simplified simulation.

        if "compound interest" in user_query.lower():
            # Example: "Calculate compound interest for $1000 at 5% for 10 years, compounded annually."
            code = "result = calculate_compound_interest(principal=1000, annual_rate=5, years=10, compounds_per_year=1)"
            nl_response_template = "Based on your request, with a principal of $1000, an annual rate of 5%, compounded annually for 10 years, the future value would be: {result}"
        elif "future value" in user_query.lower():
            # Example: "What is the future value of $5000 invested at 7% for 5 years?"
            code = "result = calculate_future_value(present_value=5000, annual_interest_rate=7, periods=5)"
            nl_response_template = "For an initial investment of $5000 at 7% over 5 years, the future value is: {result}"
        elif "risk score" in user_query.lower():
            # Example: "What is the risk score for a stock with volatility 0.8 and market sentiment 0.6?"
            code = "result = calculate_simple_risk_score(volatility_index=0.8, market_sentiment_score=0.6)"
            nl_response_template = "Considering a volatility index of 0.8 and market sentiment of 0.6, the calculated risk score is: {result}. Please note this is a simplified score."
        else:
            code = "result = 'Please rephrase your financial query. I can help with compound interest, future value, or simplified risk scores.'"
            nl_response_template = "I couldn't process your request. {result}"

        if executed_results and executed_results.get('result') is not None:
            final_nl_response = nl_response_template.format(result=round(executed_results['result'], 2))
        elif executed_results and executed_results.get('error'):
            final_nl_response = f"An error occurred during calculation: {executed_results['error']}"
        else:
             final_nl_response = nl_response_template.format(result="(calculation not performed or explicit result not found)")

        return code, final_nl_response

def main():
    print("\n--- Personalized Financial Planning Assistant (PAL Simulation) ---")
    print("I can help with basic financial calculations like compound interest, future value, and simplified risk scores.")
    print("Type 'exit' to quit.\n")

    interpreter = PythonInterpreter()
    llm_simulator = FinancialLLMSimulator()

    while True:
        user_input = input("Your financial query: ")
        if user_input.lower() == 'exit':
            break

        # 1. LLM generates code (simulated)
        generated_code, initial_nl_response = llm_simulator.generate_code_and_response(user_input, None)
        print(f"\n[Simulated LLM Generated Code]:\n{generated_code}")

        # 2. Execute the generated code
        execution_output = interpreter.execute(generated_code)

        if execution_output['error']:
            print(f"[Execution Error]: {execution_output['error']}")
            final_response = f"I encountered an error while processing your request: {execution_output['error']}"
        else:
            print(f"[Execution Result]: {execution_output['result']}")
            # 3. LLM uses execution output to formulate final response (simulated)
            _, final_response = llm_simulator.generate_code_and_response(user_input, execution_output)

        print(f"\n[Assistant's Response]: {final_response}\n")

if __name__ == "__main__":
    main()