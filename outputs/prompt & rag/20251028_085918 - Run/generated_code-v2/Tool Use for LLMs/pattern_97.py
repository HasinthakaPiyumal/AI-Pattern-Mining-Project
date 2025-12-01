import streamlit as st
import io
import contextlib
import sys

def simulate_code_generation(problem: str) -> tuple[str, str]:
    """Simulates an LLM generating Python code and an explanation for a math problem.
    In a real application, this would involve an actual LLM call.
    """
    problem_lower = problem.lower()
    generated_code = ""
    explanation = ""

    if "solve x" in problem_lower and "=" in problem_lower:
        # Simple equation solving simulation using sympy
        try:
            # Extract the equation part
            eq_str = problem_lower.split("solve ", 1)[1]
            if "for x" in eq_str:
                eq_str = eq_str.replace("for x", "").strip()
            
            # Clean up potential extra words
            for keyword in ["the equation", "equation", "find the value of x in"]: # added for robustness
                if eq_str.startswith(keyword):
                    eq_str = eq_str[len(keyword):].strip()

            # Basic parsing assumption for 'x + 2 = 5' like structures
            if '=' in eq_str:
                left, right = eq_str.split('=', 1)
                generated_code = (
                    "from sympy import symbols, Eq, solve\n"
                    f"x = symbols('x')\n"
                    f"equation = Eq({left.strip()}, {right.strip()})\n"
                    "solution = solve(equation, x)\n"
                    "if solution:\n"
                    "    print(f\"The solution for x is: {solution[0]}\")\n"
                    "else:\n"
                    "    print(\"No simple solution found for x.\")"
                )
                explanation = f"The problem was interpreted as an algebraic equation, and SymPy was used to define 'x', create the equation '{eq_str}', and solve for 'x'."
            else:
                generated_code = f"print('Could not parse equation: {eq_str}')"
                explanation = "The problem could not be parsed as a simple equation."

        except Exception as e:
            generated_code = f"print('Error simulating code generation for equation: {e}')"
            explanation = "An error occurred during the simulation of code generation for an equation."

    elif "derivative of" in problem_lower:
        # Simple derivative simulation
        try:
            func_str = problem_lower.split("derivative of", 1)[1].strip()
            if "with respect to" in func_str:
                func_part, var_part = func_str.split("with respect to", 1)
                var = var_part.strip()
                func = func_part.strip()
            else:
                # Default to 'x' if not specified
                func = func_str.strip()
                var = 'x'

            generated_code = (
                "from sympy import symbols, diff\n"
                f"x = symbols('{var}')\n"
                f"function = {func}\n"
                f"derivative = diff(function, x)\n"
                "print(f\"The derivative of {function} with respect to {x} is: {derivative}\")"
            )
            explanation = f"The problem was interpreted as finding the derivative of '{func}' with respect to '{var}'. SymPy's 'diff' function was used."
        except Exception as e:
            generated_code = f"print('Error simulating code generation for derivative: {e}')"
            explanation = "An error occurred during the simulation of code generation for a derivative problem."

    elif "integrate" in problem_lower:
        # Simple integral simulation
        try:
            func_str = problem_lower.split("integrate", 1)[1].strip()
            if "from" in func_str and "to" in func_str:
                # Definite integral (simplified for demonstration)
                parts = func_str.split("from")
                func = parts[0].strip()
                limits_part = parts[1].split("to")
                lower_limit = limits_part[0].strip()
                upper_limit = limits_part[1].strip()
                var = 'x' # Assume x for simplicity

                generated_code = (
                    "from sympy import symbols, integrate\n"
                    f"x = symbols('{var}')\n"
                    f"function = {func}\n"
                    f"integral = integrate(function, (x, {lower_limit}, {upper_limit}))\n"
                    "print(f\"The definite integral of {function} from {x}={lower_limit} to {x}={upper_limit} is: {integral}\")"
                )
                explanation = f"The problem was interpreted as a definite integral of '{func}' from {lower_limit} to {upper_limit}. SymPy's 'integrate' function was used."
            else:
                # Indefinite integral
                if "with respect to" in func_str:
                    func_part, var_part = func_str.split("with respect to", 1)
                    var = var_part.strip()
                    func = func_part.strip()
                else:
                    func = func_str.strip()
                    var = 'x'

                generated_code = (
                    "from sympy import symbols, integrate\n"
                    f"x = symbols('{var}')\n"
                    f"function = {func}\n"
                    f"integral = integrate(function, x)\n"
                    "print(f\"The indefinite integral of {function} with respect to {x} is: {integral} + C\")"
                )
                explanation = f"The problem was interpreted as an indefinite integral of '{func}' with respect to '{var}'. SymPy's 'integrate' function was used."
        except Exception as e:
            generated_code = f"print('Error simulating code generation for integral: {e}')"
            explanation = "An error occurred during the simulation of code generation for an integral problem."

    else:
        # Default for unrecognized problems
        generated_code = f"print('Problem not recognized for code generation: {problem}')"
        explanation = "The system could not generate specific code for this type of problem. Try a simple algebraic equation, derivative, or integral."
    
    return generated_code, explanation

def execute_python_code(code: str) -> tuple[str, str]:
    """Executes Python code in a sandboxed environment and captures output and errors.
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        with contextlib.redirect_stdout(stdout_capture):
            with contextlib.redirect_stderr(stderr_capture):
                # Create a limited global and local scope for execution
                exec(code, {'__builtins__': {}}, {})
        output = stdout_capture.getvalue()
        error = stderr_capture.getvalue()
    except Exception as e:
        output = ""
        error = f"Execution Error: {e}\n{stderr_capture.getvalue()}"
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    return output, error

# Streamlit App Layout
st.set_page_config(layout="wide", page_title="Intelligent Math Tutor (PAL)")
st.title("🧠 Intelligent Math Tutor (Program-aided Language Model - PAL)")
st.markdown("This tutor uses a simulated AI to generate Python code for mathematical problems and executes it to find solutions.")

problem_input = st.text_area("Enter your mathematical problem here:", 
                             "Solve the equation x + 5 = 10 for x", 
                             height=100)

if st.button("Solve Problem"):  
    if problem_input:
        st.subheader("Thinking Process (Simulated LLM)")
        st.info("The AI is analyzing the problem and generating Python code...")

        # Simulate LLM code generation
        generated_code, generation_explanation = simulate_code_generation(problem_input)

        st.subheader("Generated Python Code")
        st.code(generated_code, language="python")
        st.caption(f"_Explanation for code generation: {generation_explanation}_\n\n_Note: In a real PAL system, this code would be generated by an actual Large Language Model based on the problem input._")

        st.subheader("Code Execution Output")
        st.info("Executing the generated Python code...")

        # Execute the generated code
        execution_output, execution_error = execute_python_code(generated_code)

        if execution_output:
            st.success("Code executed successfully!")
            st.code(execution_output, language="text")
        if execution_error:
            st.error("Errors occurred during code execution:")
            st.code(execution_error, language="text")
        
        st.subheader("Final Explanation and Answer")
        if execution_output:
            st.success("Here is the final answer and explanation based on the code execution:")
            # Simulate a final LLM call to synthesize the explanation based on the output
            final_explanation = f"Based on the execution of the generated Python code, the result is:\n\n```\n{execution_output}\n```\n\nThis was obtained by applying the mathematical operations translated into Python by the AI. Specifically, {generation_explanation.lower()}"
            st.write(final_explanation)
        else:
            st.warning("Could not provide a final answer due to execution errors or no output.")

    else:
        st.warning("Please enter a mathematical problem to solve.")

st.markdown("--- Source: Program-aided Language Model (PAL) AI Design Pattern ---")
