import math

def calculate_compound_interest(principal, rate, time, compounds_per_year):
    amount = principal * (1 + rate / compounds_per_year)**(compounds_per_year * time)
    return amount

def calculate_loan_amortization(principal, annual_rate, years):
    monthly_rate = annual_rate / 12
    num_payments = years * 12
    if monthly_rate == 0:
        monthly_payment = principal / num_payments
    else:
        monthly_payment = (principal * monthly_rate) / (1 - (1 + monthly_rate)**-num_payments)
    total_payment = monthly_payment * num_payments
    total_interest = total_payment - principal
    return monthly_payment, total_payment, total_interest

def calculate_investment_growth(initial_investment, annual_return_rate, years):
    future_value = initial_investment * (1 + annual_return_rate)**years
    return future_value

def estimate_tax(income, tax_rate):
    tax_amount = income * tax_rate
    return tax_amount

def calculate_retirement_savings(current_age, retirement_age, annual_contribution, annual_return_rate):
    years_to_retirement = retirement_age - current_age
    if years_to_retirement <= 0:
        return 0.0
    future_value = 0
    for _ in range(years_to_retirement):
        future_value = (future_value + annual_contribution) * (1 + annual_return_rate)
    return future_value

def generate_code_from_query(query):
    query_lower = query.lower()
    code_to_execute = ""

    if "compound interest" in query_lower:
        try:
            principal = float([s for s in query.split() if s.startswith("$")][0][1:])
            rate = float([s for s in query.split() if "%" in s][0].replace("%", "")) / 100
            time = int([s for s in query.split() if "years" in s or "year" in s][0].split()[0])
            compounds_per_year = 1
            if "monthly" in query_lower:
                compounds_per_year = 12
            elif "quarterly" in query_lower:
                compounds_per_year = 4
            elif "semi-annually" in query_lower:
                compounds_per_year = 2
            code_to_execute = f"calculate_compound_interest({principal}, {rate}, {time}, {compounds_per_year})"
        except (ValueError, IndexError):
            return "", "Could not parse parameters for compound interest."

    elif "loan amortization" in query_lower or "monthly payment" in query_lower:
        try:
            principal_str = [s for s in query.split() if s.startswith("$")][0][1:]
            principal = float(principal_str.replace(",", ""))
            annual_rate_str = [s for s in query.split() if "%" in s][0].replace("%", "")
            annual_rate = float(annual_rate_str) / 100
            years = int([s for s in query.split() if "years" in s or "year" in s][-1])
            code_to_execute = f"calculate_loan_amortization({principal}, {annual_rate}, {years})"
        except (ValueError, IndexError):
            return "", "Could not parse parameters for loan amortization."

    elif "investment growth" in query_lower or "future value" in query_lower:
        try:
            initial_investment = float([s for s in query.split() if s.startswith("$")][0][1:])
            annual_return_rate = float([s for s in query.split() if "%" in s][0].replace("%", "")) / 100
            years = int([s for s in query.split() if "years" in s or "year" in s][-1])
            code_to_execute = f"calculate_investment_growth({initial_investment}, {annual_return_rate}, {years})"
        except (ValueError, IndexError):
            return "", "Could not parse parameters for investment growth."

    elif "estimate tax" in query_lower or "calculate tax" in query_lower:
        try:
            income = float([s for s in query.split() if s.startswith("$")][0][1:])
            tax_rate = float([s for s in query.split() if "%" in s][0].replace("%", "")) / 100
            code_to_execute = f"estimate_tax({income}, {tax_rate})"
        except (ValueError, IndexError):
            return "", "Could not parse parameters for tax estimation."

    elif "retirement savings" in query_lower:
        try:
            current_age_token = [s for s in query.split() if "current_age" in s or "I'm" in s]
            current_age = int(current_age_token[0].split()[-1]) if current_age_token else 0
            retirement_age_token = [s for s in query.split() if "retire at" in s]
            retirement_age = int(retirement_age_token[0].split()[-1]) if retirement_age_token else 0
            annual_contribution_token = [s for s in query.split() if "contribute" in s]
            annual_contribution = float(annual_contribution_token[0].split("$")[-1]) if annual_contribution_token else 0.0
            annual_return_rate = float([s for s in query.split() if "%" in s][-1].replace("%", "")) / 100
            code_to_execute = f"calculate_retirement_savings({current_age}, {retirement_age}, {annual_contribution}, {annual_return_rate})"
        except (ValueError, IndexError):
            return "", "Could not parse parameters for retirement savings."

    else:
        return "", "Sorry, I can only help with compound interest, loan amortization, investment growth, tax estimation, or retirement savings."

    return code_to_execute, ""

def execute_code_and_get_result(code_string, calculation_functions):
    result = None
    error_message = None
    try:
        exec_globals = calculation_functions.copy()
        exec_locals = {}
        exec(f"__result = {code_string}", exec_globals, exec_locals)
        result = exec_locals.get("__result")
    except Exception as e:
        error_message = f"Error during code execution: {e}"
    return result, error_message

def formulate_response(original_query, code_result):
    if code_result is None:
        return "I encountered an error during computation. Please try rephrasing your query."
    
    if "compound interest" in original_query.lower():
        return f"Based on your query, the future value with compound interest will be approximately ${code_result:,.2f}."
    elif "loan amortization" in original_query.lower() or "monthly payment" in original_query.lower():
        monthly_payment, total_payment, total_interest = code_result
        return (f"For your loan, the estimated monthly payment is ${monthly_payment:,.2f}. "
                f"The total amount paid will be ${total_payment:,.2f}, "
                f"with total interest amounting to ${total_interest:,.2f}.")
    elif "investment growth" in original_query.lower() or "future value" in original_query.lower():
        return f"Your investment is estimated to grow to approximately ${code_result:,.2f}."
    elif "estimate tax" in original_query.lower() or "calculate tax" in original_query.lower():
        return f"Your estimated tax liability is ${code_result:,.2f}."
    elif "retirement savings" in original_query.lower():
        return f"Your projected retirement savings will be approximately ${code_result:,.2f}."
    else:
        return f"I calculated the result: {code_result}. How can I help further?"

def main():
    print("Welcome to the PAL Financial Advisor!")
    print("I can help with compound interest, loan amortization, investment growth, tax estimation, and retirement savings.")
    print("Type 'exit' to quit.")

    calculation_functions = {
        "calculate_compound_interest": calculate_compound_interest,
        "calculate_loan_amortization": calculate_loan_amortization,
        "calculate_investment_growth": calculate_investment_growth,
        "estimate_tax": estimate_tax,
        "calculate_retirement_savings": calculate_retirement_savings,
        "math": math
    }

    while True:
        user_query = input("\nHow can I assist you with your finances? ")
        if user_query.lower() == 'exit':
            print("Thank you for using the PAL Financial Advisor. Goodbye!")
            break

        print(f"Thinking about your query: \"{user_query}\"")
        
        code_to_run, generation_error = generate_code_from_query(user_query)
        
        if generation_error:
            print(f"PAL Advisor: {generation_error}")
            continue

        if not code_to_run:
            print("PAL Advisor: I couldn't generate specific code for that request.")
            continue

        print(f"PAL Advisor (simulated LLM): Generated code: `{code_to_run}`")
        
        computation_result, execution_error = execute_code_and_get_result(code_to_run, calculation_functions)
        
        if execution_error:
            print(f"PAL Advisor: {execution_error}")
            continue

        final_answer = formulate_response(user_query, computation_result)
        print(f"PAL Advisor: {final_answer}")

if __name__ == "__main__":
    main()