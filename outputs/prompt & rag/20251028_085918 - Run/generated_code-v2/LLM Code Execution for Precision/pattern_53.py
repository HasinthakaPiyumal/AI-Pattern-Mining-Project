import io
import contextlib

def simulate_llm_code_generation(question):
    if "compound interest" in question.lower():
        return """\nprincipal = 1000  # Example: $1000\nrate = 0.05       # Example: 5% annual interest rate\ntime = 10         # Example: 10 years\nn_compounds = 1   # Example: compounded annually\n\n# Compound Interest Formula: A = P * (1 + r/n)^(nt)\namount = principal * (1 + rate / n_compounds)**(n_compounds * time)\ninterest_earned = amount - principal\nprint(f"Amount after {time} years: {amount:.2f}")\nprint(f"Total interest earned: {interest_earned:.2f}")\n"""
    elif "loan amortization" in question.lower():
        return """\nimport math\n\nprincipal = 100000  # Example: $100,000 loan\nannual_interest_rate = 0.04 # Example: 4% annual interest rate\nloan_term_years = 30 # Example: 30 years\n\nmonthly_interest_rate = annual_interest_rate / 12\nnumber_of_payments = loan_term_years * 12\n\n# Monthly Payment Formula: M = P [ i(1 + i)^n ] / [ (1 + i)^n – 1]\nif monthly_interest_rate > 0:\n    monthly_payment = principal * (monthly_interest_rate * (1 + monthly_interest_rate)**number_of_payments) / (((1 + monthly_interest_rate)**number_of_payments) - 1)\nelse:\n    monthly_payment = principal / number_of_payments # Simple interest for 0% rate\n\ntotal_payment = monthly_payment * number_of_payments\ntotal_interest = total_payment - principal\n\nprint(f"Monthly Payment: {monthly_payment:.2f}")\nprint(f"Total Payment: {total_payment:.2f}")\nprint(f"Total Interest Paid: {total_interest:.2f}")\n"""
    elif "investment returns" in question.lower() or "roi" in question.lower():
        return """\ninitial_investment = 5000 # Example: $5000\nfinal_value = 7500      # Example: $7500\n\nroi = ((final_value - initial_investment) / initial_investment) * 100\nprint(f"Return on Investment (ROI): {roi:.2f}%")\n"""
    else:
        return """print("I need more specific details for a financial calculation. Please ask about compound interest, loan amortization, or investment returns.")"""

def execute_python_code(code):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            exec(code)
        except Exception as e:
            return f"Error during code execution: {e}"
    return f.getvalue().strip()

def simulate_llm_answer_formulation(question, numerical_result):
    if numerical_result.startswith("Error"): # Check if the result indicates an error
        return f"I encountered an issue calculating the answer for '{question}'. The error was: {numerical_result}"
    elif "compound interest" in question.lower():
        return f"Based on your query about compound interest, here are the calculated details:\n{numerical_result}"
    elif "loan amortization" in question.lower():
        return f"For your loan amortization question, the calculations are as follows:\n{numerical_result}"
    elif "investment returns" in question.lower() or "roi" in question.lower():
        return f"Regarding your investment returns query, here is the calculated Return on Investment:\n{numerical_result}"
    else:
        return f"Here is the result of my calculation based on your request:\n{numerical_result}"

def main():
    print("Welcome to the Smart Financial Advisor!\n")
    print("Ask me about compound interest, loan amortization, or investment returns.\n")
    while True:
        user_question = input("Your financial question (type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        if not user_question.strip():
            print("Please enter a question.")
            continue

        print("\n--- AI Processing ---")
        generated_code = simulate_llm_code_generation(user_question)
        print("Generated Code:\n", generated_code)
        
        calculation_result = execute_python_code(generated_code)
        print("Code Execution Output:\n", calculation_result)
        
        final_answer = simulate_llm_answer_formulation(user_question, calculation_result)
        print("\n--- Smart Financial Advisor ---")
        print(final_answer)
        print("\n")

if __name__ == "__main__":
    main()