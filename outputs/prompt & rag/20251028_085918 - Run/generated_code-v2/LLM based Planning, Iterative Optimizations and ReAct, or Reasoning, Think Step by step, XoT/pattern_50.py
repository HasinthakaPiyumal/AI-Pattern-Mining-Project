import random

class IntelligentTutor:
    def __init__(self):
        self.problems = {
            "calculus_derivative": {
                "problem": "Find the derivative of f(x) = 3x^2 + 2x - 5.",
                "concepts": ["power rule", "constant rule", "sum rule"],
                "solution_steps": [
                    "Recall the power rule: d/dx(x^n) = nx^(n-1)",
                    "Recall the constant multiple rule: d/dx(cf(x)) = c * d/dx(f(x))",
                    "Recall the sum rule: d/dx(f(x) + g(x)) = d/dx(f(x)) + d/dx(g(x))",
                    "Apply power rule to 3x^2: 3 * 2x^(2-1) = 6x",
                    "Apply power rule to 2x: 2 * 1x^(1-1) = 2",
                    "Apply constant rule to -5: 0",
                    "Combine the results: 6x + 2 + 0 = 6x + 2"
                ],
                "answer": "6x + 2"
            },
            "physics_force": {
                "problem": "A 10 kg object is pushed with a force of 50 N. What is its acceleration?",
                "concepts": ["Newton's Second Law", "force", "mass", "acceleration"],
                "solution_steps": [
                    "Recall Newton's Second Law: F = ma",
                    "Identify given values: F = 50 N, m = 10 kg",
                    "Rearrange the formula to solve for acceleration: a = F/m",
                    "Substitute values: a = 50 N / 10 kg",
                    "Calculate the acceleration: a = 5 m/s^2"
                ],
                "answer": "5 m/s^2"
            }
        }

    def _simulate_llm_response(self, prompt):
        prompt_lower = prompt.lower()
        if "power rule" in prompt_lower:
            return "The power rule states that the derivative of x^n is nx^(n-1)."
        elif "constant rule" in prompt_lower:
            return "The derivative of a constant is always zero."
        elif "sum rule" in prompt_lower:
            return "The derivative of a sum of functions is the sum of their derivatives."
        elif "newton's second law" in prompt_lower or "f=ma" in prompt_lower:
            return "Newton's Second Law of Motion states that Force equals Mass times Acceleration (F=ma). This law describes how forces cause objects to accelerate."
        elif "derivative" in prompt_lower and "definition" in prompt_lower:
            return "The derivative measures the instantaneous rate of change of a function. Geometrically, it's the slope of the tangent line to the function's graph at a given point."
        elif "force definition" in prompt_lower:
            return "Force is an influence that can cause an object to change its velocity, i.e., to accelerate. It has both magnitude and direction."
        else:
            return "I can provide information on various STEM concepts. Could you please specify what concept you'd like to understand?"

    def _step_back_prompt(self, problem_statement, relevant_concepts):
        if not relevant_concepts:
            return None, "I'm not sure what foundational concepts are relevant here. Let's try to break down the problem directly."

        concept_to_ask = random.choice(relevant_concepts)
        generic_question = f"Can you explain the concept of {concept_to_ask} in simple terms, or remind me of its key formula/principle?"
        
        print(f"Tutor (Step-Back Question): {generic_question}")
        llm_response = self._simulate_llm_response(generic_question)
        print(f"Tutor (Foundational Knowledge): {llm_response}")
        return concept_to_ask, llm_response

    def guide_student(self, problem_key, student_attempt=None, stuck_flag=False):
        problem_info = self.problems.get(problem_key)
        if not problem_info:
            return "I don't have that problem in my database."

        problem_statement = problem_info["problem"]
        relevant_concepts = problem_info["concepts"]

        if stuck_flag or student_attempt and student_attempt.lower() != problem_info["answer"].lower():
            print(f"\nStudent seems stuck on: {problem_statement}")
            print(f"Student's attempt: {student_attempt}" if student_attempt else "")
            
            concept, foundational_knowledge = self._step_back_prompt(problem_statement, relevant_concepts)
            
            if foundational_knowledge:
                print(f"\nNow that we've refreshed our understanding of {concept}, let's apply this to the problem: {problem_statement}")
                if concept == "power rule":
                    return f"Remember how the power rule works for terms like x^n. How would you apply that to '3x^2' or '2x' in the problem?"
                elif concept == "constant rule":
                    return f"Think about the constant term '-5'. What happens to a constant when you take its derivative?"
                elif concept == "Newton's Second Law":
                    return f"Given F={problem_info['problem'].split('force of ')[1].split(' ')[0]} N and m={problem_info['problem'].split('A ')[1].split(' ')[0]} kg, how can you use F=ma to find acceleration?"
                else:
                    return f"Consider how the foundational knowledge about '{concept}' helps you approach the problem. What's the first step you would take?"
            else:
                return f"Let's break down the problem directly. What are the key components of {problem_statement}?"
        else:
            if student_attempt and student_attempt.lower() == problem_info["answer"].lower():
                return "That's correct! Great job."
            else:
                return "Please provide your attempt or let me know if you're stuck."

    def run_tutoring_session(self):
        print("Welcome to the Intelligent Tutoring System!\n")
        print("Available problems:")
        for i, key in enumerate(self.problems.keys()):
            print(f"{i+1}. {self.problems[key]['problem']}")
        print("\nType 'quit' to exit.")

        while True:
            problem_choice = input("\nWhich problem would you like to work on (e.g., 'calculus_derivative' or 'physics_force')?: ")
            if problem_choice.lower() == 'quit':
                break
            
            if problem_choice not in self.problems:
                print("Invalid problem choice. Please try again.")
                continue

            current_problem_key = problem_choice
            print(f"\nProblem: {self.problems[current_problem_key]['problem']}")
            
            stuck_count = 0
            while True:
                student_input = input("Your attempt or 'help' if you're stuck: ")
                if student_input.lower() == 'quit':
                    return
                elif student_input.lower() == 'help':
                    stuck_count += 1
                    guidance = self.guide_student(current_problem_key, stuck_flag=True)
                    print(f"Tutor: {guidance}")
                else:
                    result = self.guide_student(current_problem_key, student_attempt=student_input)
                    print(f"Tutor: {result}")
                    if "correct" in result.lower():
                        break
                    elif "stuck" not in result.lower() and "try again" not in result.lower():
                        if student_input.lower() != self.problems[current_problem_key]['answer'].lower():
                            stuck_count += 1
                            if stuck_count >= 2:
                                guidance = self.guide_student(current_problem_key, stuck_flag=True)
                                print(f"Tutor: {guidance}")
                                stuck_count = 0 # Reset after providing step-back
            print("\n--- Problem Solved! ---")

if __name__ == "__main__":
    tutor = IntelligentTutor()
    tutor.run_tutoring_session()