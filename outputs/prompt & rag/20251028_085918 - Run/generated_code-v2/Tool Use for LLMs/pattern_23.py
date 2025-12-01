def solve_math_problem(problem_description: str) -> dict:
    """
    Simulates a Program-aided Language Model (PAL) to solve math problems.
    Translates a natural language math problem into executable Python code,
    executes it, and provides a solution.
    """
    python_code = ""
    result = None
    explanation = "Could not process the problem."

    problem_description_lower = problem_description.lower()

    # Simple keyword-based translation to a Python expression
    # This is a highly simplified NLP for demonstration purposes.
    # In a real application, a more robust NLP module or an LLM would generate the code.
    operator = ""
    expression_parts = []

    if "plus" in problem_description_lower:
        expression_parts = problem_description_lower.replace("what is ", "").replace("calculate ", "").replace("\?", "").split("plus")
        operator = "+"
    elif "minus" in problem_description_lower:
        expression_parts = problem_description_lower.replace("what is ", "").replace("calculate ", "").replace("\?", "").split("minus")
        operator = "-"
    elif "multiplied by" in problem_description_lower:
        expression_parts = problem_description_lower.replace("what is ", "").replace("calculate ", "").replace("\?", "").split("multiplied by")
        operator = "*"
    elif "times" in problem_description_lower:
        expression_parts = problem_description_lower.replace("what is ", "").replace("calculate ", "").replace("\?", "").split("times")
        operator = "*"
    elif "divided by" in problem_description_lower:
        expression_parts = problem_description_lower.replace("what is ", "").replace("calculate ", "").replace("\?", "").split("divided by")
        operator = "/"
    else:
        explanation = "Problem not recognized. Please use simple arithmetic like 'X plus Y', 'X minus Y', 'X multiplied by Y', or 'X divided by Y'."
        return {
            "problem": problem_description,
            "generated_code": python_code,
            "result": result,
            "explanation": explanation
        }

    try:
        # Attempt to extract numerical values from the parts.
        # This is a very crude method and would need a proper tokenizer/parser in a real application.
        num_strs = []
        for part in expression_parts:
            # Simple attempt to find numbers (digits and decimals) in the string part
            clean_part = ''.join(c for c in part if c.isdigit() or c == '.')
            if clean_part:
                num_strs.append(clean_part)

        if len(num_strs) == 2:
            num1 = float(num_strs[0])
            num2 = float(num_strs[1])
            if operator == "/" and num2 == 0:
                raise ValueError("Division by zero is not allowed.")
            python_code = f"{num1} {operator} {num2}"
            result = eval(python_code)
            explanation = (
                f"The problem '{problem_description}' was translated into Python code "
                f"'{python_code}'. Executing this code gives the result: {result}."
            )
        else:
            explanation = "Could not extract two numbers for the operation. Please ensure the problem is clearly stated with two numbers."

    except ValueError as e:
        explanation = f"Error processing numbers or operation: {e}. Please ensure numbers are valid."
    except Exception as e:
        explanation = f"An unexpected error occurred during code execution: {e}"

    return {
        "problem": problem_description,
        "generated_code": python_code,
        "result": result,
        "explanation": explanation
    }

# Example Usage (uncomment to test):
# print(solve_math_problem("What is 5 plus 3?"))
# print(solve_math_problem("Calculate 10 minus 4."))
# print(solve_math_problem("What is 7 multiplied by 6?"))
# print(solve_math_problem("Divide 20 by 5."))
# print(solve_math_problem("What is 10 divided by 0?"))
# print(solve_math_problem("Find the sum of 1.5 and 2.5"))
# print(solve_math_problem("Solve 2 times 8"))
# print(solve_math_problem("What is 5 + 3?")) # This would fail with current parser as it looks for 'plus'
