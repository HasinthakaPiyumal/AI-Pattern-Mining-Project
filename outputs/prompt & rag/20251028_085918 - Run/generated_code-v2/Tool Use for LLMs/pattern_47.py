import streamlit as st
import sympy
import random

class CurriculumManager:
    def __init__(self):
        self.stages = [
            "Arithmetic",
            "Algebra",
            "Pre-Calculus",
            "Calculus"
        ]
        self.current_stage_idx = 0
        self.progress_in_stage = 0  # Number of problems correctly solved in current stage
        self.problems_per_stage = 3 # How many problems to solve before advancing stage

    def get_current_stage(self):
        return self.stages[self.current_stage_idx]

    def advance_progress(self):
        self.progress_in_stage += 1
        if self.progress_in_stage >= self.problems_per_stage:
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                self.progress_in_stage = 0
                return True # Stage advanced
        return False # Stage not advanced

    def reset_progress(self):
        self.current_stage_idx = 0
        self.progress_in_stage = 0

class ProblemGenerator:
    def generate_problem(self, stage):
        if stage == "Arithmetic":
            return self._generate_arithmetic_problem()
        elif stage == "Algebra":
            return self._generate_algebra_problem()
        elif stage == "Pre-Calculus":
            return self._generate_precalculus_problem()
        elif stage == "Calculus":
            return self._generate_calculus_problem()
        return "Error: Unknown stage"

    def _generate_arithmetic_problem(self):
        ops = ['+', '-', '*', '/']
        op = random.choice(ops)
        num1 = random.randint(1, 20)
        num2 = random.randint(1, 20)
        if op == '/':
            # Ensure division results in an integer for simplicity in early stages
            result = num1 * num2
            problem_str = f"{result} {op} {num1}"
            correct_answer = sympy.sympify(f"({result}) / ({num1})")
        else:
            problem_str = f"{num1} {op} {num2}"
            correct_answer = sympy.sympify(f"({num1}) {op} ({num2})")
        return problem_str, correct_answer

    def _generate_algebra_problem(self):
        x = sympy.symbols('x')
        a = random.randint(1, 5)
        b = random.randint(1, 10)
        c = random.randint(1, 20)
        # Simple linear equation: ax + b = c
        problem_expr = a*x + b - c
        problem_str = f"Solve for x: {a}x + {b} = {c}"
        solution = sympy.solve(problem_expr, x)
        if solution: # sympy.solve returns a list
            correct_answer = solution[0]
        else:
            correct_answer = None # Should not happen for simple linear eq
        return problem_str, correct_answer

    def _generate_precalculus_problem(self):
        x = sympy.symbols('x')
        choice = random.choice([1, 2])
        if choice == 1:
            # Evaluate a simple function: f(x) = ax^2 + b at a given x
            a = random.randint(1, 3)
            b = random.randint(1, 5)
            val = random.randint(1, 5)
            f_x = a*x**2 + b
            problem_str = f"If f(x) = {f_x}, what is f({val})?"
            correct_answer = f_x.subs(x, val)
        else:
            # Simple limit problem: lim x->a (x+b)
            a = random.randint(1, 3)
            b = random.randint(1, 5)
            problem_str = f"Find the limit as x approaches {a} for the function (x + {b})."
            correct_answer = sympy.limit(x + b, x, a)
        return problem_str, correct_answer

    def _generate_calculus_problem(self):
        x = sympy.symbols('x')
        choice = random.choice([1, 2])
        if choice == 1:
            # Simple derivative: d/dx (ax^n + b)
            a = random.randint(1, 5)
            n = random.randint(2, 4)
            b = random.randint(1, 10)
            func = a*x**n + b
            problem_str = f"Find the derivative of f(x) = {func}."
            correct_answer = sympy.diff(func, x)
        else:
            # Simple indefinite integral: integral(ax^n) dx
            a = random.randint(1, 5)
            n = random.randint(1, 3)
            func = a*x**n
            problem_str = f"Find the indefinite integral of f(x) = {func}."
            # SymPy's integrate adds a constant C, we'll ignore it for evaluation simplicity
            correct_answer = sympy.integrate(func, x)
        return problem_str, correct_answer

class SolutionEvaluator:
    def evaluate(self, student_answer_str, correct_answer_sym):
        try:
            student_answer_sym = sympy.sympify(student_answer_str)
            # Use .equals() for robust symbolic comparison
            return student_answer_sym.equals(correct_answer_sym)
        except (sympy.SympifyError, TypeError):
            return False

class FeedbackProvider:
    def get_feedback(self, is_correct, current_stage, problem_str, correct_answer_sym):
        if is_correct:
            return "Correct! Great job."
        else:
            # Simplified feedback; in a real app, an LLM would generate richer explanations.
            if current_stage == "Arithmetic":
                return f"Incorrect. Please recheck your arithmetic. The correct answer was {correct_answer_sym}."
            elif current_stage == "Algebra":
                return f"Incorrect. Remember the rules of algebra. The correct answer was x = {correct_answer_sym}."
            elif current_stage == "Pre-Calculus":
                return f"Incorrect. Review pre-calculus concepts. The correct answer was {correct_answer_sym}."
            elif current_stage == "Calculus":
                return f"Incorrect. Focus on your calculus rules. The correct answer was {correct_answer_sym}."
        

# Streamlit App
st.title("🧠 AI-Powered Personalized Math Tutor")

if 'curriculum' not in st.session_state:
    st.session_state.curriculum = CurriculumManager()
    st.session_state.problem_generator = ProblemGenerator()
    st.session_state.solution_evaluator = SolutionEvaluator()
    st.session_state.feedback_provider = FeedbackProvider()
    st.session_state.current_problem_text = ""
    st.session_state.current_correct_answer = None
    st.session_state.feedback = ""
    st.session_state.show_problem = False

curriculum = st.session_state.curriculum
problem_generator = st.session_state.problem_generator
solution_evaluator = st.session_state.solution_evaluator
feedback_provider = st.session_state.feedback_provider

st.sidebar.header("Progress")
st.sidebar.write(f"Current Stage: **{curriculum.get_current_stage()}**")
st.sidebar.write(f"Problems solved in stage: {curriculum.progress_in_stage} / {curriculum.problems_per_stage}")

if not st.session_state.show_problem:
    st.subheader("Welcome! Let's start learning math.")
    if st.button("Start New Session" if curriculum.current_stage_idx > 0 or curriculum.progress_in_stage > 0 else "Start Learning"):
        curriculum.reset_progress()
        st.session_state.current_problem_text, st.session_state.current_correct_answer = problem_generator.generate_problem(curriculum.get_current_stage())
        st.session_state.feedback = ""
        st.session_state.show_problem = True
        st.rerun()

if st.session_state.show_problem:
    st.subheader(f"Stage: {curriculum.get_current_stage()}")
    st.write(f"Problem: {st.session_state.current_problem_text}")

    student_answer = st.text_input("Your answer:", key="answer_input")

    if st.button("Submit Answer"):
        if student_answer:
            is_correct = solution_evaluator.evaluate(student_answer, st.session_state.current_correct_answer)
            st.session_state.feedback = feedback_provider.get_feedback(is_correct, curriculum.get_current_stage(), st.session_state.current_problem_text, st.session_state.current_correct_answer)
            
            if is_correct:
                stage_advanced = curriculum.advance_progress()
                st.success(st.session_state.feedback)
                if stage_advanced:
                    st.balloons()
                    st.subheader(f"🎉 Congratulations! You've advanced to the {curriculum.get_current_stage()} stage!")
                st.session_state.current_problem_text, st.session_state.current_correct_answer = problem_generator.generate_problem(curriculum.get_current_stage())
                st.experimental_rerun() # Rerun to clear input and show new problem
            else:
                st.error(st.session_state.feedback)
        else:
            st.warning("Please enter an answer.")

    if st.session_state.feedback and not st.session_state.feedback.startswith("Correct!"):
        st.info(st.session_state.feedback)


    # Display new problem if it's set
    # This part is handled by the reruns on correct answer submission
    # If the user clicks 'Start Learning' or 'Submit' after correcting, a new problem will be generated.
