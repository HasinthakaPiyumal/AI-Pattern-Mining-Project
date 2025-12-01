import streamlit as st
import pandas as pd
import subprocess
import os
import json

def get_user_input():
    st.title("Financial Advisory Bot with PAL Prompting")
    st.header("Enter Your Financial Details")

    risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
    investment_horizon = st.selectbox("Investment Horizon", ["Short-term (1-3 years)", "Medium-term (3-10 years)", "Long-term (10+ years)"])
    desired_returns = st.slider("Desired Annual Returns (%)", 1.0, 20.0, 7.0)

    st.subheader("Current Portfolio (Asset Name: Value, Risk_Factor)")
    st.markdown("Example: `Stocks:10000:7, Bonds:5000:3` (Risk Factor 1-10)")
    portfolio_input = st.text_area("Enter your portfolio assets", "Stocks:10000:7, Bonds:5000:3, Real Estate:20000:5")

    portfolio_data = []
    try:
        items = [item.strip().split(":") for item in portfolio_input.split(",")]
        for item in items:
            if len(item) == 3:
                portfolio_data.append({
                    "asset": item[0].strip(),
                    "value": float(item[1].strip()),
                    "risk_factor": float(item[2].strip())
                })
            else:
                st.warning(f"Skipping malformed entry: {':'.join(item)}. Expected 'Asset:Value:Risk_Factor'.")
    except Exception as e:
        st.error(f"Error parsing portfolio: {e}. Please use format 'Asset:Value:Risk_Factor'.")
        portfolio_data = []

    user_data = {
        "risk_tolerance": risk_tolerance,
        "investment_horizon": investment_horizon,
        "desired_returns": desired_returns,
        "portfolio": portfolio_data
    }
    return user_data

def mock_llm_generate_code(user_data):
    portfolio_json_str = json.dumps(user_data.get("portfolio", []))

    generated_code = f"""
import pandas as pd
import json

portfolio_data_list = json.loads('{portfolio_json_str}')
df_portfolio = pd.DataFrame(portfolio_data_list)

if not df_portfolio.empty:
    total_value = df_portfolio['value'].sum()
    df_portfolio['weighted_risk'] = df_portfolio['value'] * df_portfolio['risk_factor']
    average_risk_score = df_portfolio['weighted_risk'].sum() / total_value if total_value > 0 else 0
    
    print(f"Total Portfolio Value: {{total_value:.2f}}")
    print(f"Average Portfolio Risk Score (Weighted): {{average_risk_score:.2f}}")
elif not portfolio_data_list:
    print("No portfolio data provided for calculation.")
    print("Total Portfolio Value: 0.00")
    print("Average Portfolio Risk Score (Weighted): 0.00")
else:
    print("Error processing portfolio data for calculation.")
    print("Total Portfolio Value: 0.00")
    print("Average Portfolio Risk Score (Weighted): 0.00")
"""
    return generated_code

def execute_generated_code(code_string):
    script_filename = "temp_financial_script.py"
    try:
        with open(script_filename, "w") as f:
            f.write(code_string)

        result = subprocess.run(
            ["python", script_filename],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error executing code: {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    finally:
        if os.path.exists(script_filename):
            os.remove(script_filename)

def mock_llm_generate_response(user_data, code_output):
    response = "Based on your input and our analysis:\n\n"
    response += f"Your stated risk tolerance: {user_data['risk_tolerance']}\n"
    response += f"Your investment horizon: {user_data['investment_horizon']}\n"
    response += f"Your desired annual returns: {user_data['desired_returns']}%\n\n"

    total_value = "N/A"
    calculated_risk_score = "N/A"
    for line in code_output.split('\n'):
        if "Total Portfolio Value" in line:
            total_value = line.split(":")[1].strip()
        elif "Average Portfolio Risk Score (Weighted)" in line:
            calculated_risk_score = line.split(":")[1].strip()

    response += f"From our calculations:\n"
    response += f"- Total Portfolio Value: {total_value}\n"
    response += f"- Average Portfolio Risk Score (Weighted 1-10): {calculated_risk_score}\n\n"

    if calculated_risk_score != "N/A" and calculated_risk_score != "0.00":
        try:
            score = float(calculated_risk_score)
            stated_risk_map = {"Low": 3, "Medium": 6, "High": 9}
            stated_risk_proxy = stated_risk_map.get(user_data['risk_tolerance'], 5)

            if score > stated_risk_proxy + 1:
                response += f"Recommendation: Your calculated portfolio risk score ({score:.2f}) is higher than what typically aligns with your stated '{user_data['risk_tolerance']}' risk tolerance. Consider re-evaluating your asset allocation to reduce exposure to higher-risk assets.\n"
            elif score < stated_risk_proxy - 1:
                response += f"Recommendation: Your calculated portfolio risk score ({score:.2f}) is lower than what typically aligns with your stated '{user_data['risk_tolerance']}' risk tolerance. If you're comfortable with more risk, you might explore opportunities in growth-oriented assets to potentially achieve higher returns.\n"
            else:
                response += "Recommendation: Your portfolio's calculated average risk score appears to be generally aligned with your stated risk tolerance. A balanced approach with periodic review is recommended.\n"
        except ValueError:
            response += "Recommendation: Unable to provide a specific risk-based recommendation due to calculation issues.\n"
    else:
        response += "Recommendation: Please provide valid portfolio data to receive specific recommendations.\n"

    response += "\nDisclaimer: This is a simulated financial advisory bot for demonstration purposes. It does not provide real financial advice. Consult a qualified financial advisor for personalized guidance."
    return response

def main():
    user_data = get_user_input()

    if st.button("Get Financial Advice"):
        if not user_data["portfolio"]:
            st.error("Please enter your portfolio details to get advice.")
            return

        st.subheader("Generating Advice...")

        generated_code = mock_llm_generate_code(user_data)
        st.subheader("Generated Python Code by LLM (for execution):")
        st.code(generated_code, language="python", height=200)

        st.subheader("Code Execution Output:")
        code_execution_output = execute_generated_code(generated_code)
        st.text_area("Output from the executed Python script:", code_execution_output, height=150)

        final_advice = mock_llm_generate_response(user_data, code_execution_output)

        st.subheader("Financial Advisory Report")
        st.write(final_advice)

if __name__ == "__main__":
    main()