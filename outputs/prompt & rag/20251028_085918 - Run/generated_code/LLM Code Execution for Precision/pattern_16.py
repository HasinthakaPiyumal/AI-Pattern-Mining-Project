import openai

class LLMService:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

    def generate_code_prompt(self, query):
        # This prompt instructs the LLM to generate Python code for financial calculations.
        return f"""You are a financial Python code generator. Based on the user's request, generate executable Python code to perform the necessary financial calculations. The code should print the final result clearly. Do not include any explanations or extra text, only the Python code. You can use standard libraries like numpy, pandas if necessary, or the functions provided in 'financial_tools.py'.

User request: {query}

Example for NPV calculation:
```python
def calculate_npv(rate, cash_flows):
    npv = 0
    for i, cf in enumerate(cash_flows):
        npv += cf / (1 + rate)**i
    return npv

project_cash_flows = [-100000, 20000, 30000, 40000, 50000]
discount_rate = 0.10
result = calculate_npv(discount_rate, project_cash_flows)
print(f"NPV: {{result}}")
```

Your code:
"""

    def generate_advice_prompt(self, original_query, code_output):
        # This prompt instructs the LLM to formulate natural language advice based on the code output.
        return f"""Based on the following original user query and the output from the financial calculation code, provide a comprehensive and data-driven financial advisory in natural language. Explain the results and offer personalized recommendations.

Original Query: {original_query}

Financial Calculation Output:
{code_output}

Financial Advice:
"""

    def get_completion(self, prompt, model="gpt-3.5-turbo", temperature=0.0):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful financial assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error during LLM completion: {e}"
