import os
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from io import StringIO
import sys

# Ensure OPENAI_API_KEY is set in your environment variables
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def execute_python_code(code_snippet: str) -> str:
    """
    Safely executes a Python code snippet and captures its output.
    """
    # Create a dictionary to serve as the local and global scope for execution
    local_vars = {}
    global_vars = {}

    # Redirect stdout to capture print statements
    old_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output

    try:
        exec(code_snippet, global_vars, local_vars)
        # If the LLM assigns a result to 'calculation_result', we want to return that.
        # Otherwise, we return captured stdout.
        if 'calculation_result' in local_vars:
            return str(local_vars['calculation_result'])
        elif 'calculation_result' in global_vars:
            return str(global_vars['calculation_result'])
        else:
            # Return any printed output if no specific variable was set
            return redirected_output.getvalue().strip()
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        sys.stdout = old_stdout # Restore stdout

def financial_assistant(user_query: str) -> str:
    """
    Processes a financial query using an LLM to generate and execute code for calculations,
    then uses the LLM again to explain the results.
    """

    # --- Step 1: LLM generates Python code for calculation ---
    code_generation_system_message = """
    You are an expert financial analyst. Your task is to generate a Python code snippet to accurately calculate the financial metric requested by the user. 
    The code should be executable and assign the final calculated numerical result to a variable named 'calculation_result'. 
    Do not include any explanations, comments, or extra text, only the raw Python code. 
    Ensure all necessary imports are included (e.g., 'math').
    Example: If asked 'Calculate future value of $1000 at 5% annual interest compounded annually for 10 years.', 
    you should output: 'principal = 1000\nrate = 0.05\ntime = 10\ncalculation_result = principal * (1 + rate)**time'
    """
    code_generation_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(code_generation_system_message),
        HumanMessagePromptTemplate.from_template("{query}")
    ])
    
    chain_code_generation = code_generation_prompt | llm
    llm_response_code = chain_code_generation.invoke({"query": user_query})
    generated_code = llm_response_code.content.strip()

    if not generated_code:
        return "I could not generate a valid Python code snippet for your request."
    
    # --- Step 2: Execute the generated Python code ---
    calculated_result = execute_python_code(generated_code)

    if "Error executing code:" in calculated_result:
        return f"There was an error during calculation: {calculated_result}. Please rephrase your query or provide more details."
    
    # --- Step 3: LLM generates natural language explanation ---
    explanation_system_message = """
    You are a helpful financial assistant. Based on the user's original query and the calculated numerical result, 
    provide a clear, concise, and insightful explanation. 
    Include the calculated value and any relevant financial context or recommendations.
    """
    explanation_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(explanation_system_message),
        HumanMessagePromptTemplate.from_template("Original Query: {original_query}\nCalculated Result: {result}\n\nExplain this result and provide any relevant insights.")
    ])

    chain_explanation = explanation_prompt | llm
    llm_response_explanation = chain_explanation.invoke({
        "original_query": user_query,
        "result": calculated_result
    })
    
    final_explanation = llm_response_explanation.content.strip()

    return final_explanation

# --- Gradio Interface ---
if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY environment variable is not set. Please set it before running the application.")
        print("You can set it like: export OPENAI_API_KEY='your_key_here'")
        # Optionally, you can sys.exit() or provide a placeholder for the key if you want to run locally with a hardcoded key for testing (not recommended for production)

    interface = gr.Interface(
        fn=financial_assistant,
        inputs=gr.Textbox(lines=2, placeholder="e.g., Calculate the future value of $5000 with an 8% annual interest compounded quarterly for 7 years."),
        outputs="text",
        title="PAL Financial Assistant",
        description="Ask financial questions, and I will use an LLM to generate and execute Python code for accurate calculations, then explain the results."
    )

    interface.launch()
