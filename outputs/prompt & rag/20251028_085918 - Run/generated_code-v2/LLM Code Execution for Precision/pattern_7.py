import streamlit as st
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import re
import io
import sys


app = FastAPI()

class Query(BaseModel):
    user_query: str

def mock_llm_process(user_query: str, code_output: str = None, extracted_params: dict = None):
    
    principal = 0
    annual_interest_rate = 0
    years = 0
    compound_per_year = 4

    principal_match = re.search(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', user_query)
    if principal_match:
        try:
            principal = float(principal_match.group(1).replace(',', ''))
        except ValueError:
            pass
    
    interest_rate_match = re.search(r'(\d+(?:\.\d+)?)\s*%', user_query)
    if interest_rate_match:
        try:
            annual_interest_rate = float(interest_rate_match.group(1))
        except ValueError:
            pass
    
    years_match = re.search(r'(\d+)\s*(?:year|yr)s?', user_query)
    if years_match:
        try:
            years = int(years_match.group(1))
        except ValueError:
            pass

    if extracted_params is None:
        extracted_params = {
            "principal": principal,
            "annual_interest_rate": annual_interest_rate,
            "years": years,
            "compound_per_year": compound_per_year
        }

    if code_output is None:
        if "loan amortization" in user_query.lower() or "monthly payment" in user_query.lower():
            return {
                "type": "code_generation",
                "code": f"""
def calculate_loan_amortization(principal, annual_interest_rate, years):\n    monthly_interest_rate = annual_interest_rate / 12 / 100\n    number_of_payments = years * 12\n    if monthly_interest_rate == 0:\n        monthly_payment = principal / number_of_payments\n    else:\n        monthly_payment = (principal * monthly_interest_rate) / (1 - (1 + monthly_interest_rate)**-number_of_payments)\n    \n    total_payment = monthly_payment * number_of_payments\n    total_interest = total_payment - principal\n    \n    return {{\n        "monthly_payment": round(monthly_payment, 2),\n        "total_payment": round(total_payment, 2),\n        "total_interest": round(total_interest, 2)\n    }}\n\nresult = calculate_loan_amortization({principal}, {annual_interest_rate}, {years})\nprint(json.dumps(result))\n                """,
                "extracted_params": extracted_params
            }
        elif "compound interest" in user_query.lower() or "investment growth" in user_query.lower():
            return {
                "type": "code_generation",
                "code": f"""
def calculate_compound_interest(principal, annual_interest_rate, years, compound_per_year):\n    amount = principal * (1 + (annual_interest_rate / 100) / compound_per_year)**(compound_per_year * years)\n    interest_earned = amount - principal\n    return {{\n        "final_amount": round(amount, 2),\n        "interest_earned": round(interest_earned, 2)\n    }}\n\nresult = calculate_compound_interest({principal}, {annual_interest_rate}, {years}, {compound_per_year})\nprint(json.dumps(result))\n                """,
                "extracted_params": extracted_params
            }
        else:
            return {
                "type": "text_response",
                "response": "I can help with financial calculations like loan amortization or compound interest. Please specify your query more precisely.",
                "extracted_params": extracted_params
            }
    else:
        try:
            output_dict = json.loads(code_output)
            
            principal_val = extracted_params.get("principal", 0)
            annual_interest_rate_val = extracted_params.get("annual_interest_rate", 0)
            years_val = extracted_params.get("years", 0)
            compound_per_year_val = extracted_params.get("compound_per_year", 0)

            if "monthly_payment" in output_dict:
                return {
                    "type": "text_response",
                    "response": f"For a loan of ${principal_val:,.2f} at {annual_interest_rate_val}% interest over {years_val} years, your estimated monthly payment would be ${output_dict['monthly_payment']:,.2f}, with a total repayment of ${output_dict['total_payment']:,.2f} and total interest paid of ${output_dict['total_interest']:,.2f}."
                }
            elif "final_amount" in output_dict:
                return {
                    "type": "text_response",
                    "response": f"With an initial principal of ${principal_val:,.2f} at {annual_interest_rate_val}% interest compounded {compound_per_year_val} times a year for {years_val} years, your investment would grow to an estimated final amount of ${output_dict['final_amount']:,.2f}, having earned ${output_dict['interest_earned']:,.2f} in interest."
                }
            else:
                return {
                    "type": "text_response",
                    "response": f"I processed the calculation, but the output format was unexpected. Raw output: {code_output}"
                }
        except json.JSONDecodeError:
            return {
                "type": "text_response",
                "response": f"An error occurred during calculation or the output was not in a readable format. Raw output: {code_output}"
            }


@app.post("/advise")
async def get_financial_advice(query: Query):
    user_query = query.user_query
    
    llm_code_response = mock_llm_process(user_query)
    
    if llm_code_response["type"] == "text_response":
        return {"advice": llm_code_response["response"]}
        
    generated_code = llm_code_response["code"]
    extracted_params = llm_code_response["extracted_params"]
    
    code_output = ""
    old_stdout = sys.stdout
    redirected_output = io.StringIO()

    try:
        safe_globals = {
            "__builtins__": {
                "print": print,
                "json": json,
                "round": round,
                "float": float,
                "int": int,
                "str": str,
                "len": len,
                "abs": abs,
                "max": max,
                "min": min,
                "sum": sum,
                "pow": pow
            },
            "math": __import__("math")
        }
        safe_locals = {}
        
        sys.stdout = redirected_output
        
        exec(generated_code, safe_globals, safe_locals)
        code_output = redirected_output.getvalue().strip()
        
    except Exception as e:
        code_output = f"Error during code execution: {str(e)}"
    finally:
        sys.stdout = old_stdout
    
    llm_final_response = mock_llm_process(user_query, code_output=code_output, extracted_params=extracted_params)
    
    return {"advice": llm_final_response["response"]}


st.title("Personalized Financial Advisor")
st.write("Ask me complex financial questions (e.g., loan amortization, compound interest) and I'll provide a precise answer.")
st.markdown("---")

user_input = st.text_input(
    "Your financial query (e.g., 'What is the monthly payment for a $100,000 loan at 5% interest over 30 years?' or 'How much will a $15,000 investment grow to in 10 years at 7% interest compounded quarterly?'):",
    "What is the monthly payment for a $100,000 loan at 5% interest over 30 years?"
)

if st.button("Get Advice"):
    if user_input:
        with st.spinner("Calculating your financial advice..."):
            try:
                response = requests.post("http://localhost:8000/advise", json={"user_query": user_input})
                response.raise_for_status()
                
                advice = response.json().get("advice", "Could not get advice.")
                st.subheader("Your Financial Advice:")
                st.info(advice)
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Please ensure it's running at http://localhost:8000. Instructions are in the code.")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}. Check the FastAPI server logs for more details.")
    else:
        st.warning("Please enter a financial query to get advice.")

st.markdown("---")
st.caption("Disclaimer: This is a demonstration of Program-Aided Language Models (PAL) prompting and should not be used for actual financial decisions. The LLM is mocked for illustrative purposes, and the code execution environment is simplified. Real-world applications require robust LLM integration and secure, isolated execution environments.")

