import numpy_financial as nf

def llm_proxy(query: str) -> str:
    generated_code = ""
    result_value = None
    response_template = "I couldn't process that financial query. Please try rephrasing."

    if "future value" in query.lower():
        # Simulate parsing parameters from a more complex query for demonstration
        # For simplicity, hardcoding values for now.
        rate = 0.05
        nper = 10
        pv = -1000
        pmt = 0
        generated_code = f"""import numpy_financial as nf\nrate = {rate}\nnper = {nper}\npv = {pv}\npmt = {pmt}\n_computed_result = nf.fv(rate, nper, pmt, pv)\n"""
        response_template = f"The future value of an investment of ${abs(pv)} at {rate*100}% interest over {nper} years is: {{result_placeholder:,.2f}}."

    elif "present value" in query.lower():
        rate = 0.03
        nper = 5
        pmt = 100
        fv = 0
        generated_code = f"""import numpy_financial as nf\nrate = {rate}\nnper = {nper}\npmt = {pmt}\nfv = {fv}\n_computed_result = nf.pv(rate, nper, pmt, fv)\n"""
        response_template = f"The present value of an annuity paying ${pmt} annually for {nper} years at {rate*100}% discount rate is: {{result_placeholder:,.2f}}."

    elif "mortgage payment" in query.lower() or "loan payment" in query.lower():
        rate = 0.045 / 12  # 4.5% annual interest, monthly
        nper = 30 * 12    # 30 years, monthly payments
        pv = 200000       # Loan amount
        fv = 0
        generated_code = f"""import numpy_financial as nf\nrate = {rate}\nnper = {nper}\npv = {pv}\nfv = {fv}\n_computed_result = nf.pmt(rate, nper, pv, fv)\n"""
        response_template = f"For a loan of ${pv:,.2f} at an annual rate of {0.045*100}% over {30} years, the monthly payment is: {{result_placeholder:,.2f}}."

    if generated_code:
        local_vars = {}
        try:
            exec(generated_code, {"nf": nf}, local_vars)
            result_value = local_vars.get("_computed_result")
            if result_value is not None:
                return response_template.format(result_placeholder=result_value)
            else:
                return "An error occurred during code execution or result extraction."
        except Exception as e:
            return f"An error occurred during code execution: {e}"
    else:
        return response_template

def main():
    print("Financial Advisory and Portfolio Optimization Assistant (PAL Simulation)")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nEnter your financial query: ")
        if user_query.lower() == 'exit':
            break

        response = llm_proxy(user_query)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()