
import gradio as gr
import io
import sys

class ProgrammingTask:
    def __init__(self, name, description, problem_statement, test_cases):
        self.name = name
        self.description = description
        self.problem_statement = problem_statement
        self.test_cases = test_cases

    def evaluate(self, user_code):
        feedback = ""
        try:
            old_stdout = sys.stdout
            redirected_output = io.StringIO()
            sys.stdout = redirected_output

            exec_globals = {}
            exec(user_code, exec_globals)

            sys.stdout = old_stdout

            all_passed = True
            for inputs, expected_output in self.test_cases:
                if isinstance(inputs, tuple):
                    func_args = inputs
                else:
                    func_args = (inputs,)
                
                # Assuming the user's code defines a function that can be called
                # We need to find the function name from the problem statement or define a convention
                # For simplicity, let's assume the problem statement implies the function name
                # e.g., 'Define a function called 'add'...' -> assume 'add' is the function
                
                # This part is a simplification. A robust solution would parse the problem statement
                # or have a specific field for the expected function name.
                # For now, let's try to find a function name if present in problem_statement
                func_name = None
                if "function called '" in self.problem_statement:
                    start = self.problem_statement.find("function called '") + len("function called '")
                    end = self.problem_statement.find("'", start)
                    if start != -1 and end != -1:
                        func_name = self.problem_statement[start:end]
                elif "function `" in self.problem_statement:
                    start = self.problem_statement.find("function `") + len("function `")
                    end = self.problem_statement.find("`", start)
                    if start != -1 and end != -1:
                        func_name = self.problem_statement[start:end]
                
                if func_name and func_name in exec_globals and callable(exec_globals[func_name]):
                    actual_output = exec_globals[func_name](*func_args)
                else:
                    # Fallback for simple print statements or if function name not found
                    actual_output = redirected_output.getvalue().strip()
                    if actual_output == "" and expected_output == "": # Special case for no explicit output
                        pass # allow tests without explicit output for simple exec
                    else:
                        feedback += f"Could not find function '{func_name}' or no explicit output for input {inputs}.\n"
                        all_passed = False
                        break

                if actual_output == expected_output:
                    feedback += f"Test passed for input {inputs}.\n"
                else:
                    feedback += f"Test failed for input {inputs}: Expected '{expected_output}', Got '{actual_output}'.\n"
                    all_passed = False
            
            if all_passed:
                feedback += "All tests passed!\n"
            
            return all_passed, feedback

        except SyntaxError as e:
            feedback = f"Syntax Error: {e}\n"
            return False, feedback
        except Exception as e:
            feedback = f"Runtime Error: {e}\n{redirected_output.getvalue()}"
            return False, feedback
        finally:
            sys.stdout = old_stdout


class CurriculumManager:
    def __init__(self, tasks):
        self.tasks = tasks
        self.current_task_index = 0

    def get_current_task(self):
        if 0 <= self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None

    def next_task(self):
        if self.current_task_index < len(self.tasks) - 1:
            self.current_task_index += 1
            return self.get_current_task()
        return None

    def get_progress(self):
        return f"Task {self.current_task_index + 1} of {len(self.tasks)}"


# Define the curriculum tasks
tasks = [
    ProgrammingTask(
        name="Hello World",
        description="Your first step into programming!",
        problem_statement="Write a Python program that prints the string 'Hello, World!' to the console.",
        test_cases=[(None, "Hello, World!")] # None as input for simple print, expected stdout
    ),
    ProgrammingTask(
        name="Add Two Numbers",
        description="Learn to define and call a function.",
        problem_statement="Define a function called `add` that takes two numbers as arguments and returns their sum.",
        test_cases=[
            ((2, 3), 5),
            ((-1, 5), 4),
            ((0, 0), 0)
        ]
    ),
    ProgrammingTask(
        name="Is Even",
        description="Practice with conditional statements.",
        problem_statement="Define a function called `is_even` that takes an integer `n` as an argument and returns `True` if `n` is even, `False` otherwise.",
        test_cases=[
            ((4,), True),
            ((7,), False),
            ((0,), True)
        ]
    ),
    ProgrammingTask(
        name="Sum of List",
        description="Work with loops and lists.",
        problem_statement="Define a function called `sum_list` that takes a list of numbers as an argument and returns the sum of all numbers in the list.",
        test_cases=[
            (([1, 2, 3],), 6),
            (([10, -5, 0],), 5),
            (([],), 0)
        ]
    )
]

curriculum = CurriculumManager(tasks)

def get_task_info():
    current_task = curriculum.get_current_task()
    if current_task:
        return current_task.name, current_task.description, current_task.problem_statement, curriculum.get_progress()
    return "No more tasks!", "", "", "Finished"

def run_code(user_code):
    current_task = curriculum.get_current_task()
    if current_task:
        passed, feedback = current_task.evaluate(user_code)
        return feedback
    return "No task available."

def next_task(user_code, current_feedback):
    current_task = curriculum.get_current_task()
    if current_task:
        passed, _ = current_task.evaluate(user_code)
        if passed:
            next_t = curriculum.next_task()
            if next_t:
                name, desc, prob, prog = get_task_info()
                return name, desc, prob, prog, "", "" # Clear code input and previous feedback
            else:
                return "Curriculum Completed!", "", "You have finished all tasks.", curriculum.get_progress(), "", "" # Clear code input and previous feedback
        else:
            return current_task.name, current_task.description, current_task.problem_statement, curriculum.get_progress(), current_code_input, "Please correct your code before moving to the next task." # Keep current code and update feedback
    return "", "", "", "", "", "No task available to advance."

# Initialize UI elements with the first task's information
initial_name, initial_desc, initial_prob, initial_prog = get_task_info()

with gr.Blocks() as demo:
    gr.Markdown("# Intelligent Programming Tutor")

    with gr.Row():
        task_name_output = gr.Textbox(label="Current Task", value=initial_name, interactive=False)
        task_progress_output = gr.Textbox(label="Progress", value=initial_prog, interactive=False)

    task_description_output = gr.Textbox(label="Description", value=initial_desc, interactive=False, lines=2)
    problem_statement_output = gr.Markdown(value=f"### Problem:\n{initial_prob}")

    code_input = gr.Code(label="Your Code", language="python", lines=10, value="")
    feedback_output = gr.Textbox(label="Feedback", lines=5, interactive=False)

    with gr.Row():
        run_button = gr.Button("Run Code")
        next_button = gr.Button("Next Task (Only if current task passed)")

    run_button.click(
        fn=run_code,
        inputs=[code_input],
        outputs=[feedback_output]
    )

    next_button.click(
        fn=next_task,
        inputs=[code_input, feedback_output], # Pass current code and feedback to re-evaluate on next_task click
        outputs=[task_name_output, task_description_output, problem_statement_output, task_progress_output, code_input, feedback_output]
    )

demo.launch()