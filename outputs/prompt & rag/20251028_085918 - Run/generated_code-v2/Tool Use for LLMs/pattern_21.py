
import streamlit as st
import io
import contextlib

# --- 1. Simulate LLM Code Generation ---
# In a real application, this would involve an actual LLM API call
# (e.g., using OpenAI's API or a local LLM via Hugging Face transformers/langchain).
# For demonstration, we'll have a simplified function.

def generate_math_solution_code(problem: str) -> str:
    """
    Simulates an LLM generating Python code to solve a mathematical problem.
    In a real scenario, this would involve prompting an LLM.
    """
    problem_lower = problem.lower()

    if "addition" in problem_lower and ("what is" in problem_lower or "sum of" in problem_lower):
        parts = problem.replace("?", "").replace("what is", "").replace("the sum of", "").replace("plus", "+").split("+")
        try:
            nums = [float(p.strip()) for p in parts if p.strip()]
            if len(nums) == 2:
                return f"num1 = {nums[0]}\nnum2 = {nums[1]}\nresult = num1 + num2\nprint(f\"The sum is: {{result}}\")"
        except (ValueError, IndexError):
            pass # Fallback to generic
    elif "multiplication" in problem_lower and ("what is" in problem_lower or "product of" in problem_lower):
        parts = problem.replace("?", "").replace("what is", "").replace("the product of", "").replace("times", "*").split("*")
        try:
            nums = [float(p.strip()) for p in parts if p.strip()]
            if len(nums) == 2:
                return f"num1 = {nums[0]}\nnum2 = {nums[1]}\nresult = num1 * num2\nprint(f\"The product is: {{result}}\")"
        except (ValueError, IndexError):
            pass # Fallback to generic
    elif "area of a rectangle" in problem_lower:
        import re
        length_match = re.search(r"length of (\\d+\\.?\\d*)", problem_lower)
        width_match = re.search(r"width of (\\d+\\.?\\d*)", problem_lower)
        if length_match and width_match:
            length = float(length_match.group(1))
            width = float(width_match.group(1))
            return f"length = {length}\nwidth = {width}\narea = length * width\nprint(f\"The area of the rectangle is: {{area}}\")"

    # Default/fallback for more complex or unrecognized problems
    # In a real LLM, this would be much more sophisticated
    return f"""
# Math problem: {problem}
# A more advanced LLM would generate specific code here.
# For now, let's provide a placeholder or a simple calculation if possible.
try:
    # Attempt to use eval for simple numerical expressions if present
    import re
    math_expression_pattern = r"(\\d+(\\\\.\\d+)?)\\s*[\\+\\-\\*/]\\s*(\\d+(\\\\.\\d+)?)"
    math_expression_match = re.search(math_expression_pattern, "{problem}")
    if math_expression_match:
        # Ensure we only evaluate safe expressions
        expression_str = math_expression_match.group(0)
        # Basic safety check: ensure only digits and basic operators are present
        if all(c.isdigit() or c in "+-*/. " for c in expression_str):
            result = eval(expression_str)
            print(f"Based on the extracted expression '{{expression_str}}', the result is: {{result}}")
        else:
            print("Extracted expression contains unsafe characters. An LLM would generate specific Python code.")
    else:
        print("This problem requires advanced reasoning. An LLM would generate specific Python code.")
        print("Example: If the problem was 'What is 5 + 3?', the code might be:")
        print("x = 5")
        print("y = 3")
        print("print(x + y)")
except Exception as e:
    print(f"Could not automatically solve simple expression: {{e}}")
    print("This problem requires advanced reasoning. An LLM would generate specific Python code.")
"""

# --- 2. Code Interpreter ---
@contextlib.contextmanager
def stdout_capture():
    """Context manager to capture stdout."""
    old_stdout = io.StringIO()
    new_stdout = io.StringIO()
    with contextlib.redirect_stdout(new_stdout):
        yield new_stdout
    old_stdout.close()

def execute_code(code: str) -> (str, str):
    """
    Executes Python code in a sandboxed environment and captures output and errors.
    Returns (captured_output, error_message).
    """
    captured_output = ""
    error_message = ""
    local_scope = {}

    try:
        # Restrict builtins to prevent arbitrary code execution for enhanced safety
        # In a real production system, a more robust sandboxing solution (e.g., separate process, container) is recommended.
        exec(code, {"__builtins__": {}}, local_scope)
        captured_output = local_scope.get("stdout_buffer", io.StringIO()).getvalue() # Capture if code redirected output
    except Exception as e:
        error_message = str(e)

    # If the code directly used print, it would be captured by redirect_stdout
    # The above exec() doesn't automatically capture print in the try block's direct scope.
    # We need to wrap the exec with stdout_capture.
    with stdout_capture() as s:
        try:
            exec(code, {"__builtins__": {}}, local_scope)
            captured_output = s.getvalue()
        except Exception as e:
            error_message = str(e)

    return captured_output, error_message

# --- 3. Streamlit Application ---
def main():
    st.set_page_config(page_title="Intelligent Math Tutor (Program of Thoughts)", layout="wide")
    st.title("🧠 Intelligent Math Tutor powered by Program of Thoughts")
    st.subheader("Leveraging LLMs and code execution for mathematical problem-solving")

    st.markdown("""
    This application demonstrates the "Program of Thoughts" pattern:
    1.  You input a mathematical problem.
    2.  An AI (simulated LLM) generates Python code as reasoning steps.
    3.  A code interpreter executes this Python code.
    4.  You get the result and the code that led to it.
    This approach enhances transparency and accuracy for computational tasks.
    """)

    problem_input = st.text_area("Enter your mathematical problem here:",
                                 "What is the sum of 123 and 456?",
                                 height=100)

    if st.button("Solve Problem"):
        if not problem_input:
            st.warning("Please enter a mathematical problem.")
            return

        st.info("Generating code as reasoning steps...")
        generated_code = generate_math_solution_code(problem_input)

        st.subheader("Generated Python Code (Reasoning Steps):")
        st.code(generated_code, language="python")

        st.info("Executing the generated code...")
        output, error = execute_code(generated_code)

        st.subheader("Execution Result:")
        if error:
            st.error(f"An error occurred during code execution:\n`{error}`")
            st.markdown("Please review the generated code or problem statement. The sandbox environment limits direct access to certain Python features.")
        else:
            if output.strip():
                st.success("Code executed successfully!")
                st.code(output, language="text")
            else:
                st.warning("Code executed, but produced no explicit output. This might mean the problem was not fully resolved by the generated code, or the code didn't print anything.")
                st.markdown("Check the `Generated Python Code` section to ensure it includes `print()` statements for the expected output.")

        st.subheader("Explanation of Program of Thoughts:")
        st.markdown("""
        The core idea of "Program of Thoughts" is to explicitly integrate an external tool (a code interpreter) 
        with an LLM's reasoning capabilities. Instead of directly outputting an answer, the LLM generates 
        **executable code** as intermediate reasoning steps. This code is then run by an interpreter, 
        providing several benefits:

        *   **Accuracy:** For mathematical and algorithmic tasks, code execution ensures precise, error-free computation.
        *   **Verifiability:** The code provides a transparent trace of how the answer was derived, making the reasoning auditable.
        *   **Tool Use:** It effectively leverages the strengths of both LLMs (understanding and code generation) 
            and deterministic tools (computation via code interpreter).

        In this application, when you provide a math problem, the simulated AI constructs a Python program. 
        This program represents the AI's "thought process" for solving the problem. The interpreter then 
        executes these thoughts to arrive at the final, verifiable answer.
        """)

if __name__ == "__main__":
    main()
