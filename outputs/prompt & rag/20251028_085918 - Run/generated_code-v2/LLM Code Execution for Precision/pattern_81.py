import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import io
import contextlib

# Load environment variables from .env file
load_dotenv()

# Initialize LLM
# Ensure OPENAI_API_KEY is set in your .env file or environment variables
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

def execute_python_code(code: str) -> str:
    """
    Executes Python code in a restricted environment and captures its output.
    """
    old_stdout = io.StringIO()
    redirect = contextlib.redirect_stdout(old_stdout)
    
    # Restrict the globals to built-in functions for security
    # In a production environment, a more robust sandboxing solution
    # (e.g., separate process, Docker container) would be highly recommended.
    restricted_globals = {"__builtins__": {},
                          "print": print, # Allow print for output
                          "abs": abs, "max": max, "min": min, "round": round,
                          "sum": sum, "len": len, "range": range,
                          "float": float, "int": int, "str": str, "bool": bool,
                          "list": list, "dict": dict, "tuple": tuple, "set": set
                          }
    
    try:
        with redirect:
            # Execute the code. Using exec is inherently risky; this is a basic example.
            exec(code, restricted_globals)
        output = old_stdout.getvalue()
        return output.strip()
    except Exception as e:
        return f"Error during code execution: {e}"

def smart_financial_advisor(user_query: str) -> str:
    """
    Processes a user's financial query using an LLM to generate and execute
    Python code for precise calculations, then formulates a natural language response.
    """
    # Step 1: Prompt the LLM to generate Python code
    code_generation_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial assistant. Generate Python code to solve the following financial problem. The code should print the final numerical result. Do not include any explanations, just the pure Python code block. Use standard Python libraries. For financial calculations, you might need math or numpy, but try to keep it simple if possible. If the query involves advanced finance functions, assume basic formula applications."),
        ("human", "{query}")
    ])
    
    code_chain = code_generation_template | llm | StrOutputParser()
    
    # Example of a slightly more complex prompt that guides the LLM
    # to use specific variable names or function patterns.
    # For this example, we'll keep it simple as the LLM is expected to be smart enough.
    
    generated_code = code_chain.invoke({"query": user_query})
    
    # Extract code block if LLM adds markdown formatting (common for code generation)
    if "```python" in generated_code:
        generated_code = generated_code.split("```python")[1].split("```")[0].strip()
    elif "```" in generated_code:
        generated_code = generated_code.split("```")[1].split("```")[0].strip()
        
    print(f"Generated Code:\n{generated_code}\n") # For debugging/visibility

    # Step 2: Execute the generated code
    execution_result = execute_python_code(generated_code)
    print(f"Execution Result: {execution_result}\n") # For debugging/visibility

    # Step 3: Prompt the LLM to formulate a natural language answer
    answer_generation_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful financial advisor. Based on the user's original query and the calculated result, provide a clear, concise, and friendly natural language answer."),
        ("human", "Original Query: {query}\nCalculation Result: {result}\n\nBased on this, what is the final answer?")
    ])
    
    answer_chain = answer_generation_template | llm | StrOutputParser()
    
    final_answer = answer_chain.invoke({"query": user_query, "result": execution_result})
    
    return final_answer

if __name__ == "__main__":
    print("\n--- Smart Financial Advisor Demo ---\n")
    
    # Example 1: Compound interest calculation
    query1 = "Calculate the future value of an investment with an initial principal of $10,000, compounded annually at 6% for 5 years."
    response1 = smart_financial_advisor(query1)
    print(f"User Query: {query1}")
    print(f"Advisor: {response1}\n")

    # Example 2: Monthly loan payment
    query2 = "What is the monthly payment for a $200,000 loan at an annual interest rate of 4.5% over 30 years?"
    response2 = smart_financial_advisor(query2)
    print(f"User Query: {query2}")
    print(f"Advisor: {response2}\n")
    
    # Example 3: Invalid code (LLM might generate, or simply an execution error)
    query3 = "What is 5 divided by zero?"
    response3 = smart_financial_advisor(query3)
    print(f"User Query: {query3}")
    print(f"Advisor: {response3}\n")
    
    # Example 4: Complex investment scenario
    query4 = "Calculate the future value of an investment with an initial principal of $5,000, monthly contributions of $100, an annual interest rate of 7% compounded monthly, over 10 years."
    response4 = smart_financial_advisor(query4)
    print(f"User Query: {query4}")
    print(f"Advisor: {response4}\n")
