import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set your OpenAI API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

class SmartFinancialAdvisor:
    def __init__(self):
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.llm = OpenAI(temperature=0.0)

    def _generate_code(self, query: str) -> str:
        code_prompt_template = PromptTemplate(
            input_variables=["query"],
            template=(
                "You are a financial expert. Based on the following financial question, "
                "generate a Python code snippet that performs the necessary calculations. "
                "The code should be executable and print the final numerical result. "
                "Only output the Python code, no explanations or extra text.\n\n" 
                "Question: {query}\n" 
                "Python Code:"
            )
        )
        code_chain = LLMChain(llm=self.llm, prompt=code_prompt_template)
        generated_code = code_chain.run(query=query)
        return generated_code.strip()

    def _execute_code(self, code: str) -> str:
        try:
            local_vars = {}
            exec(code, {"__builtins__": None}, local_vars)
            return str(local_vars.get("result", "No explicit 'result' variable set in code."))
        except Exception as e:
            return f"Error during code execution: {e}"

    def _generate_explanation(self, query: str, calculation_result: str) -> str:
        explanation_prompt_template = PromptTemplate(
            input_variables=["query", "calculation_result"],
            template=(
                "Based on the following financial question and the calculation result, "
                "provide a comprehensive and easy-to-understand financial explanation and advice. "
                "Explain what the numbers mean and give actionable insights.\n\n" 
                "Question: {query}\n" 
                "Calculation Result: {calculation_result}\n\n" 
                "Financial Advice:"
            )
        )
        explanation_chain = LLMChain(llm=self.llm, prompt=explanation_prompt_template)
        explanation = explanation_chain.run(query=query, calculation_result=calculation_result)
        return explanation.strip()

    def get_financial_advice(self, query: str) -> str:
        print("Generating Python code for calculation...")
        generated_code = self._generate_code(query)
        print("Generated Code:\n" + generated_code + "\n")

        print("Executing code...")
        calculation_result = self._execute_code(generated_code)
        print("Calculation Result: " + calculation_result + "\n")

        print("Generating financial advice explanation...")
        final_advice = self._generate_explanation(query, calculation_result)
        return final_advice

if __name__ == "__main__":
    advisor = SmartFinancialAdvisor()

    print("\n--- Smart Financial Advisor ---")
    print("Ask me a financial question (e.g., 'What is the future value of an investment of $1000 at an annual interest rate of 5% compounded monthly for 10 years?'):")
    user_query = input("Your question: ")

    if user_query:
        advice = advisor.get_financial_advice(user_query)
        print("\n--- Your Financial Advice --- ")
        print(advice)
    else:
        print("Please enter a financial question.")
