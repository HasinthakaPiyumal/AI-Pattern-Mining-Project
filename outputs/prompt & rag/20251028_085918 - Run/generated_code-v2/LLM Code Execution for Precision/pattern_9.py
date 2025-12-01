from fastapi import FastAPI
import uvicorn
import os
import io
import contextlib
import numpy_financial as npf

# Simulate environment variable loading (in a real app, use .env file)
# from dotenv import load_dotenv
# load_dotenv()

app = FastAPI()

class FinancialQuery:
    query: str

def simulate_llm_code_generation(query: str) -> str:
    if "investment" in query.lower() and "worth" in query.lower() and "return" in query.lower():
        # Example: "How much will my $10,000 investment be worth in 5 years with a 7% annual return?"
        # Extract parameters (simplified for simulation)
        initial_investment = 10000
        years = 5
        annual_return_rate = 0.07
        return (
            f"import numpy_financial as npf\n"
            f"pv = -{initial_investment}\n"
            f"rate = {annual_return_rate}\n"
            f"nper = {years}\n"
            f"fv = npf.fv(rate, nper, 0, pv)\n"
            f"print(f\"Your investment will be worth: ${fv:,.2f}\")"
        )
    elif "loan" in query.lower() and "monthly payments" in query.lower() and "interest" in query.lower():
        # Example: "What are the monthly payments for a $200,000 loan over 30 years at 4% interest?"
        # Extract parameters (simplified for simulation)
        principal = 200000
        years = 30
        annual_interest_rate = 0.04
        
        monthly_rate = annual_interest_rate / 12
        nper_months = years * 12
        
        return (
            f"import numpy_financial as npf\n"
            f"principal = {principal}\n"
            f"monthly_rate = {monthly_rate}\n"
            f"nper_months = {nper_months}\n"
            f"pmt = npf.pmt(monthly_rate, nper_months, -principal)\n"
            f"print(f\"Your estimated monthly payment will be: ${pmt:,.2f}\")"
        )
    else:
        return 