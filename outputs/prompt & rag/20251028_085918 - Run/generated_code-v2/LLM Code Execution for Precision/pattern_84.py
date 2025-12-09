import gradio as gr
import os
import subprocess
import re
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from decimal import Decimal, getcontext

# Set environment variable for API key (replace with your actual API key or secure loading)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- Configuration ---
# Ensure you have OPENAI_API_KEY set in your environment variables
# For local development, you can uncomment the line above and set it directly,
# but for production, use secure environment variable management.

# Initialize LLM - Using a placeholder for demonstration. 
# Replace with your actual LLM integration (e.g., Google's Gemini, Anthropic's Claude)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Financial Calculation Utilities (Prompt for LLM to generate these) ---
# The LLM will be prompted to generate Python code that includes these types of calculations.
# For example, it might generate a function like this:
# def calculate_compound_interest(principal, rate, years, compounds_per_year):
#     amount = principal * (1 + rate / compounds_per_year)**(compounds_per_year * years)
#     return amount - principal

# --- PAL Engine: Code Generation Module ---
def generate_code_from_query(query: str) -> str:
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an AI financial analyst. Your task is to generate precise Python code "
         "to solve financial queries. The code should be self-contained, executable, "
         "and print the final result to standard output in a clear, human-readable format, "
         "preferably as a JSON object with a 'result' key, or a descriptive string. "
         "Use the `decimal` module for high precision calculations when dealing with money. "
         "You can import `math`, `decimal`, `numpy` (if array operations are needed), "
         "or `pandas` (if data manipulation is needed). "
         "Only output the Python code block, no other text or explanation. "
         "Example: If asked for compound interest, generate code to calculate and print it. "
         "If a user asks for a monthly investment to reach a target, the code should calculate that."
        ),
        ("human", "{query}")
    ])
    
    chain = prompt_template | llm
    response = chain.invoke({"query": query}).content
    
    # Extract Python code block from the LLM's response
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    else:
        # If no code block is found, try to return the whole response if it looks like code
        if "import" in response or "def " in response or "print(" in response:
             return response.strip()
        return f"# Could not generate executable code for query: {query}\n# LLM response: {response}"

# --- PAL Engine: Code Execution Environment (Sandbox) ---
def execute_python_code(code: str) -> str:
    temp_file = "_generated_financial_code.py"
    try:
        with open(temp_file, "w") as f:
            f.write(code)
        
        # Execute the generated Python script in an isolated environment
        process = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            text=True, # Capture output as text
            check=False, # Don't raise an exception for non-zero exit codes
            timeout=30 # Add a timeout to prevent infinite loops
        )
        
        if process.returncode != 0:
            error_output = process.stderr.strip()
            return f"Error during code execution:\n{error_output}\n\nGenerated Code:\n{code}"
        else:
            return process.stdout.strip()
            
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after 30 seconds.\n\nGenerated Code:\n{code}"
    except Exception as e:
        return f"An unexpected error occurred during execution: {e}\n\nGenerated Code:\n{code}"
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file) # Clean up the temporary file

# --- Main Application Logic ---
def financial_analyst_assistant(query: str) -> str:
    if not query.strip():
        return "Please enter a financial query."
        
    generated_code = generate_code_from_query(query)
    
    if generated_code.startswith("# Could not generate executable code"):
        return generated_code # Return the error message directly from code generation
        
    execution_result = execute_python_code(generated_code)
    
    full_response = (
        f"### Your Query:\n{query}\n\n"
        f"### Generated Python Code:\n```python\n{generated_code}\n```\n\n"
        f"### Execution Result:\n```\n{execution_result}\n```\n\n"
        f"---\n"
        f"Based on the execution, here is the final analysis:\n"
    )
    
    # Try to parse JSON output from the execution result for a more refined answer
    try:
        result_json = json.loads(execution_result)
        if "result" in result_json:
            full_response += result_json["result"]
        else:
            full_response += "The code executed successfully, but the output format was not a recognized JSON with a 'result' key.\n" + execution_result
    except json.JSONDecodeError:
        # If not JSON, just append the raw execution output
        full_response += execution_result
    
    return full_response

# --- Gradio User Interface ---
if __name__ == "__main__":
    interface = gr.Interface(
        fn=financial_analyst_assistant,
        inputs=gr.Textbox(lines=5, label="Enter your financial query"),
        outputs=gr.Markdown(label="Financial Analysis"),
        title="AI-powered Financial Analyst Assistant (PAL Prompting)",
        description=(
            "Ask complex financial questions, and the AI will generate Python code to calculate "
            "the answer precisely. The code will be executed, and the result presented to you." 
            "\n\nExamples: "
            "\n- Calculate the compound interest for $5000 over 10 years at an annual rate of 7%, compounded monthly." 
            "\n- What is the future value of an annuity with monthly payments of $200 for 20 years at an annual interest rate of 6%?"
            "\n- Calculate the monthly payment for a $300,000 mortgage over 30 years at a fixed annual interest rate of 4.5%." 
        )
    )
    interface.launch()
