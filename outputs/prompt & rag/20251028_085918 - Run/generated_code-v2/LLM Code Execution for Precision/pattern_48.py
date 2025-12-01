from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

import code_executor
import financial_utils

class FinancialAdvisoryBot:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        self.code_executor = code_executor.CodeExecutor()
        self.agent_executor = self._initialize_agent()

    def _initialize_agent(self):
        tools = [
            Tool(
                name="Python_Interpreter",
                func=self._execute_financial_code,
                description="""
                A Python interpreter for executing financial calculations and operations.
                Input: Python code as a string. 
                The code can use functions from the 'financial_utils' module, such as:
                - calculate_simple_interest(principal: float, rate: float, time: float) -> float
                - calculate_compound_interest(principal: float, rate: float, time: float, n: int) -> float
                - calculate_roi(initial_investment: float, final_value: float) -> float
                - present_value(future_value: float, rate: float, periods: float) -> float
                Example usage:
                print(financial_utils.calculate_simple_interest(1000, 0.05, 2))
                result = financial_utils.calculate_compound_interest(1000, 0.05, 10, 4)
                """
            ),
        ]

        # Define the prompt template for the ReAct agent
        # This prompt encourages the LLM to think, then act (potentially by writing code)
        # and then observe the results to formulate a final answer.
        prompt_template = PromptTemplate.from_template(
            """You are a financial advisory bot designed to provide precise financial advice and analysis.
            You have access to a Python interpreter tool to perform complex calculations.
            When a user asks a question requiring numerical computation or financial analysis, 
            you should generate Python code to solve it using the provided 'financial_utils' functions.
            After executing the code, use the results to formulate a clear and accurate financial recommendation or explanation.

            TOOLS:
            ------
            {tools}

            FORMAT INSTRUCTIONS:
            --------------------
            {format_instructions}

            USER'S INPUT:
            -------------
            {input}

            YOUR RESPONSE:
            --------------
            """
        )

        agent = create_react_agent(self.llm, tools, prompt_template)
        return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    def _execute_financial_code(self, code_string: str) -> str:
        # Make financial_utils available in the execution context
        # WARNING: This is still not fully secure for arbitrary code.
        # In a real application, you would need a more robust sandboxing solution.
        exec_globals = {"financial_utils": financial_utils}
        exec_locals = {}

        # Modify the code string to run within the CodeExecutor, 
        # ensuring financial_utils is in scope and result is captured.
        # The CodeExecutor already tries to capture the last expression's result
        # but for clarity, we can ensure 'result' is explicitly set or printed.
        modified_code = (
            f"import financial_utils\n"  # Ensure financial_utils is available if not in exec_globals
            f"_result_holder = None\n"
            f"try:\n"
            f"{code_string.replace('\n', '\n    ')}\n"
            f"except Exception as e:\n"
            f"    _result_holder = str(e)\n"
            f"result = _result_holder if _result_holder is not None else None"
        )

        execution_output = self.code_executor.execute_python_code(code_string)
        
        if execution_output["error"]:
            return f"Execution Error: {execution_output['error']}"
        elif execution_output["stdout"]:
            return f"Output: {execution_output['stdout'].strip()}"
        else:
            return f"Result: {execution_output['result']}"


    def run(self, query: str) -> str:
        try:
            response = self.agent_executor.invoke({"input": query})
            return response["output"]
        except Exception as e:
            return f"An error occurred during agent execution: {e}"

if __name__ == "__main__":
    bot = FinancialAdvisoryBot()
    print("Financial Advisory Bot ready! Type 'exit' to quit.")
    while True:
        user_query = input("\nYour financial question: ")
        if user_query.lower() == 'exit':
            break
        
        # Example of how to structure the prompt to guide the LLM
        # The prompt template in _initialize_agent already does this.
        # This part is more for the user interaction loop.
        
        print("Bot thinking...")
        bot_response = bot.run(user_query)
        print(f"\nBot: {bot_response}")
