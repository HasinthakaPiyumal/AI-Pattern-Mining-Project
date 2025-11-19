import io
import contextlib

class LLMService:
    def generate_code_for_financial_query(self, query: str, financial_data: dict) -> str:
        """
        Simulates an LLM generating Python code for a financial query.
        In a real scenario, this would involve an actual LLM API call with prompt engineering.
        """
        if "compound interest" in query.lower() and "principal" in financial_data and "rate" in financial_data and "years" in financial_data:
            # Example: Calculate compound interest
            code = f"""
def calculate_compound_interest(principal, annual_rate, years, compound_per_year=12):
    rate_per_period = annual_rate / compound_per_year
    num_periods = years * compound_per_year
    future_value = principal * (1 + rate_per_period)**num_periods
    return future_value

principal = {financial_data['principal']}
annual_rate = {financial_data['rate']}
years = {financial_data['years']}
result = calculate_compound_interest(principal, annual_rate, years)
print(f"The future value with compound interest is: {{result:.2f}}")
            """
        elif "loan payment" in query.lower() and "principal" in financial_data and "annual_rate" in financial_data and "months" in financial_data:
            # Example: Calculate monthly loan payment
            code = f"""
def calculate_monthly_loan_payment(principal, annual_rate, months):
    # Convert annual_rate percentage to decimal and then to monthly rate
    monthly_rate = (annual_rate / 100) / 12
    if monthly_rate == 0:
        return principal / months
    payment = (principal * monthly_rate) / (1 - (1 + monthly_rate)**-months)
    return payment

principal = {financial_data['principal']}
annual_rate = {financial_data['annual_rate']}
months = {financial_data['months']}
result = calculate_monthly_loan_payment(principal, annual_rate, months)
print(f"The monthly loan payment is: {{result:.2f}}")
            """
        elif "portfolio risk" in query.lower():
            code = f"""
# This is a placeholder for a more complex portfolio risk calculation
# In a real scenario, the LLM would generate code using libraries like numpy for statistical analysis.
print("Portfolio risk analysis would be performed here using advanced algorithms.")
            """
        else:
            code = "print(\'Could not generate specific code for the given query. Please refine your request.\')"
        return code

    def generate_advice_from_results(self, query: str, financial_data: dict, calculation_result: str) -> str:
        """
        Simulates an LLM generating financial advice based on the query and calculation results.
        """
        advice = f"Based on your query: '{query}' and the calculation result: '{calculation_result}', here is some personalized financial advice:\n"

        if "compound interest" in query.lower():
            advice += "Consider the power of compounding for long-term investments. Even small regular contributions can grow significantly over time. Reinvesting your earnings can accelerate this growth."
        elif "loan payment" in query.lower():
            advice += "Understanding your monthly loan payments is crucial for budgeting. Aim to pay more than the minimum if possible to reduce the total interest paid and shorten the loan term."
        elif "portfolio risk" in query.lower():
            advice += "Assessing portfolio risk involves looking at various factors like diversification, market volatility, and your personal risk tolerance. Diversifying across different asset classes can help mitigate risk."
        else:
            advice += "It's important to regularly review your financial situation and seek professional advice for complex decisions."
        return advice
