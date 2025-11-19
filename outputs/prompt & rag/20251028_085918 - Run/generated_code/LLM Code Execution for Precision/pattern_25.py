import math

def calculate_compound_interest(principal, monthly_investment, annual_rate, years):
    """
    Calculates the future value of an investment with compound interest and regular contributions.
    """
    monthly_rate = annual_rate / 12 / 100
    months = years * 12

    future_value_principal = principal * (1 + monthly_rate)**months
    future_value_contributions = monthly_investment * (((1 + monthly_rate)**months - 1) / monthly_rate) * (1 + monthly_rate)

    return round(future_value_principal + future_value_contributions, 2)

def calculate_loan_amortization(principal, annual_rate, years, extra_payment=0):
    """
    Calculates the monthly payment and generates an amortization schedule for a loan.
    Returns (monthly_payment, total_interest, schedule).
    """
    monthly_rate = annual_rate / 12 / 100
    months = years * 12

    if monthly_rate == 0:
        monthly_payment_raw = principal / months
    else:
        monthly_payment_raw = principal * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
    
    monthly_payment = math.ceil(monthly_payment_raw * 100) / 100 # Round up to nearest cent

    # Recalculate monthly payment if extra_payment is added to ensure minimum payment covers interest
    if extra_payment > 0:
        # The actual payment will be monthly_payment + extra_payment
        # We need to make sure the original monthly_payment is at least covering the interest
        pass # The loop below will handle the actual payment logic

    schedule = []
    current_principal = principal
    total_interest_paid = 0
    
    month_count = 0
    while current_principal > 0 and month_count < months * 2: # Add a safety break for very long loans
        month_count += 1
        interest_payment = current_principal * monthly_rate
        
        effective_payment = monthly_payment + extra_payment
        
        # Ensure payment covers at least interest
        principal_payment = max(0, effective_payment - interest_payment)
        
        # If the remaining principal is less than the principal payment, adjust
        if current_principal - principal_payment < 0:
            principal_payment = current_principal
            effective_payment = interest_payment + principal_payment

        current_principal -= principal_payment
        total_interest_paid += interest_payment
        
        schedule.append({
            "month": month_count,
            "beginning_balance": round(principal + (interest_payment + principal_payment) - effective_payment, 2) if month_count > 1 else round(principal, 2),
            "payment": round(effective_payment, 2),
            "interest_paid": round(interest_payment, 2),
            "principal_paid": round(principal_payment, 2),
            "ending_balance": round(max(0, current_principal), 2)
        })
        
        # If principal becomes very small, consider it paid off to avoid tiny residual payments
        if current_principal < 0.01:
            current_principal = 0
            break

    # Adjust the last payment to zero out the loan exactly
    if schedule and schedule[-1]["ending_balance"] > 0: # If there's still a balance due to rounding
        last_payment_entry = schedule[-1]
        remaining_balance = last_payment_entry["ending_balance"]
        # Find the next payment to cover this remaining balance
        # For simplicity, let's assume one more payment covers it entirely
        # In a real scenario, this would be more complex to calculate the exact final payment
        final_interest = remaining_balance * monthly_rate # Interest on the remaining small balance
        final_principal_payment = remaining_balance
        final_payment = final_interest + final_principal_payment

        schedule.append({
            "month": month_count + 1,
            "beginning_balance": round(remaining_balance, 2),
            "payment": round(final_payment, 2),
            "interest_paid": round(final_interest, 2),
            "principal_paid": round(final_principal_payment, 2),
            "ending_balance": 0.00
        })
        total_interest_paid += final_interest

    return round(monthly_payment, 2), round(total_interest_paid, 2), schedule
