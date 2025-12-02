import streamlit as st
import re
import io
import sys

class PALAgent:
    def __init__(self):
        pass

    def _mock_llm_call(self, prompt):
        if "generate Python code" in prompt:
            if "future value" in prompt:
                return """
```python
def calculate_future_value(monthly_investment, years, annual_return_rate):
    monthly_return_rate = annual_return_rate / 12 / 100
    months = years * 12
    if monthly_return_rate == 0:
        fv = monthly_investment * months
    else:
        fv = monthly_investment * (((1 + monthly_return_rate)**months - 1) / monthly_return_rate)
    return round(fv, 2)

result = calculate_future_value(500, 15, 8)
print(result)
```
"""
            return "```python\nprint('No specific code mock for this query.')\n```"
        elif "computational result" in prompt:
            match = re.search(r"computational result: ([\d.]+)", prompt)
            result = match.group(1) if match else "an unspecified amount"
            return f"Based on your query and the calculation, the future value of your investment is approximately ${result}. This suggests a strong potential for growth over time, assuming the specified return rates. Consider diversifying your portfolio and regularly reviewing your investment strategy to meet your financial goals."
        return "I am sorry, I cannot process this request at the moment."

    def _extract_code(self, llm_response):
        match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
        if match:
            return match.group(1)
        return None

    def _safe_exec(self, code_block):
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        execution_result = None
        local_scope = {}
        try:
            exec(code_block, {}, local_scope)
            if 'result' in local_scope:
                execution_result = str(local_scope['result'])
            else:
                execution_result = redirected_output.getvalue().strip()
        except Exception as e:
            execution_result = f"Error: {e}"
        finally:
            sys.stdout = old_stdout
        return execution_result

    def get_financial_advice(self, user_query):
        code_gen_prompt = f"The user wants to know: '{user_query}'. Please generate Python code to perform the necessary financial calculation. Enclose the code in a '```python\\n...\\n```' block. Make sure the code prints the final numerical result."

        llm_code_response = self._mock_llm_call(code_gen_prompt)

        python_code = self._extract_code(llm_code_response)
        if python_code:
            computational_result = self._safe_exec(python_code)

            advice_gen_prompt = f"The user asked: '{user_query}'. The computational result is: {computational_result}. Please provide personalized financial advice based on this information in natural language."

            llm_advice_response = self._mock_llm_call(advice_gen_prompt)
            return llm_advice_response
        else:
            return "I could not generate or execute the required financial calculation code. Please try rephrasing your query."

st.set_page_config(page_title="Financial Advisor Bot", layout="centered")

st.title("💰 Financial Advisor Bot (PAL Prompting Demo)")
st.markdown("""
This bot uses a simulated Program-Aided Language Model (PAL) prompting approach
to provide financial advice. It attempts to generate and execute Python code
for calculations before formulating a natural language response.
""")

user_query = st.text_input(
    "Ask me a financial question (e.g., 'Calculate the future value of investing $500 monthly for 15 years at an 8% annual return.')",
    "Calculate the future value of investing $500 monthly for 15 years at an 8% annual return."
)

if st.button("Get Financial Advice"):
    if user_query:
        st.info("Thinking...")
        agent = PALAgent()
        advice = agent.get_financial_advice(user_query)
        st.success("Here's your personalized financial advice:")
        st.write(advice)
    else:
        st.warning("Please enter a financial query.")

st.markdown("---")
st.caption("Disclaimer: This is a demonstration for an AI design pattern and should not be used for actual financial advice. The LLM interaction is simulated.")