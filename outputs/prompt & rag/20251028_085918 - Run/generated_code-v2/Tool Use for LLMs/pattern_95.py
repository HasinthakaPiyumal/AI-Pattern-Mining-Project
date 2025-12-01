import streamlit as st
import os
import io
import sys
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Configure Streamlit page
st.set_page_config(page_title="Intelligent Math & Coding Tutor", layout="wide")

st.title("🧠 Intelligent Math & Coding Tutor")
st.markdown("""
Welcome! I can help you solve complex mathematical problems and debug programming exercises
by generating and executing Python code as reasoning steps.
""")

# --- Secure Code Execution Function ---
def execute_python_code(code: str):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    redirected_error = io.StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_error

    execution_result = {}
    try:
        # Basic sandbox: restrict builtins and prevent direct file/system access.
        # WARNING: This is NOT truly secure for untrusted arbitrary code execution.
        # For a production system, consider dedicated secure sandboxes (e.g., Docker, separate process with strict ACLs).
        restricted_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "range": range,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
            },
            "math": __import__("math") # Allow basic math module
        }
        exec(code, restricted_globals)
        execution_result["output"] = redirected_output.getvalue()
        execution_result["error"] = redirected_error.getvalue()
    except Exception as e:
        execution_result["error"] = f"Execution error: {e}\n{redirected_error.getvalue()}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return execution_result

# --- LLM Setup ---
# Ensure OPENAI_API_KEY is set in your environment variables
# Example (for local testing): os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
if "OPENAI_API_KEY" in os.environ:
    llm = ChatOpenAI(model="gpt-4", temperature=0.2) # Using gpt-4 for better code generation
else:
    st.error("Please set your OPENAI_API_KEY environment variable. You can get one from platform.openai.com.")
    llm = None

# Prompt Template for the LLM to guide its response
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert tutor in mathematics and programming. "
     "When given a problem, your first step is to think step-by-step. "
     "Then, generate a concise Python code snippet that solves the problem or provides a crucial step towards the solution. "
     "The code should be enclosed in triple backticks (```python ... ```). "
     "After generating the code, provide a brief explanation of your thought process and what the code does. "
     "Finally, offer a natural language explanation of the solution based on the code's output. "
     "Do not execute the code yourself. Focus on generating the correct and executable Python code."),
    ("human", "{problem_description}")
])

# --- Streamlit Frontend ---
problem_input = st.text_area(
    "Enter your mathematical problem or programming challenge:",
    "Calculate the 10th Fibonacci number.",
    height=150
)

if st.button("Solve/Debug"):
    if not llm:
        st.stop()

    if problem_input:
        with st.spinner("Thinking and generating code..."):
            # Create an LLM chain
            chain = prompt_template | llm

            # Invoke the LLM with the user's problem
            llm_response = chain.invoke({"problem_description": problem_input})
            llm_response_content = llm_response.content

            st.subheader("LLM's Reasoning and Generated Code:")

            # --- Parse LLM Response for Code and Explanations ---
            code_start_tag = "```python"
            code_end_tag = "```"

            code_start_index = llm_response_content.find(code_start_tag)
            code_end_index = llm_response_content.find(code_end_tag, code_start_index + len(code_start_tag))

            generated_code = ""
            if code_start_index != -1 and code_end_index != -1:
                generated_code = llm_response_content[code_start_index + len(code_start_tag):code_end_index].strip()
                st.code(generated_code, language="python")

                # Display LLM's explanations before and after the code block
                st.markdown(llm_response_content[:code_start_index].strip())
                st.markdown(llm_response_content[code_end_index + len(code_end_tag):].strip())

            else:
                st.warning("LLM did not generate a Python code block in the expected format.")
                st.markdown(llm_response_content) # Display raw LLM output if code not found
                generated_code = "" # Ensure empty if no code found

        if generated_code:
            st.subheader("Code Execution Result:")
            with st.spinner("Executing generated code..."):
                execution_output = execute_python_code(generated_code)

                if execution_output["output"]:
                    st.success("Output:")
                    st.code(execution_output["output"], language="text")
                if execution_output["error"]:
                    st.error("Error during execution:")
                    st.code(execution_output["error"], language="text")
                if not execution_output["output"] and not execution_output["error"]:
                    st.info("Code executed, but produced no explicit output to stdout.")

    else:
        st.warning("Please enter a problem to get started!")