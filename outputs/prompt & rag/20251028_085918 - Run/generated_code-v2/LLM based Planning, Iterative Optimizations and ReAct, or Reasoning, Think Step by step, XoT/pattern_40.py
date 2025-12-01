import streamlit as st

# Predefined problems and their simulated CoT and final answers
PROBLEMS = [
    {
        "id": 1,
        "question": "If a car travels at 60 miles per hour, how long will it take to travel 180 miles?",
        "model_cot": "Let's think step by step.\nStep 1: Identify the given values: speed = 60 mph, distance = 180 miles.\nStep 2: Recall the formula relating distance, speed, and time: Time = Distance / Speed.\nStep 3: Substitute the given values into the formula: Time = 180 miles / 60 mph.\nStep 4: Calculate the time: Time = 3 hours.\nTherefore, it will take 3 hours to travel 180 miles.",
        "model_answer": "3 hours"
    },
    {
        "id": 2,
        "question": "A rectangle has a length of 8 cm and a width of 5 cm. What is its area?",
        "model_cot": "Let's think step by step.\nStep 1: Identify the given values: length = 8 cm, width = 5 cm.\nStep 2: Recall the formula for the area of a rectangle: Area = Length * Width.\nStep 3: Substitute the given values into the formula: Area = 8 cm * 5 cm.\nStep 4: Calculate the area: Area = 40 square cm.\nTherefore, the area of the rectangle is 40 square cm.",
        "model_answer": "40 square cm"
    },
    {
        "id": 3,
        "question": "What is 15% of 200?",
        "model_cot": "Let's think step by step.\nStep 1: Understand what '15% of 200' means. It means (15/100) * 200.\nStep 2: Convert the percentage to a decimal or fraction: 15% = 0.15 or 15/100.\nStep 3: Multiply the decimal/fraction by 200: 0.15 * 200.\nStep 4: Perform the multiplication: 0.15 * 200 = 30.\nTherefore, 15% of 200 is 30.",
        "model_answer": "30"
    }
]

def simulate_llm_cot(problem_id):
    """Simulates an LLM generating a Chain of Thought for a given problem."""
    for problem in PROBLEMS:
        if problem["id"] == problem_id:
            return problem["model_cot"], problem["model_answer"]
    return "", ""

def generate_feedback(student_cot, student_answer, model_cot, model_answer):
    """Generates feedback by comparing student's CoT and answer with the model's."""
    feedback = "### Feedback\n\n"
    if student_cot.strip() == model_cot.strip():
        feedback += "Your chain of thought is very similar to the model's! Great job.\n"
    else:
        feedback += "Consider reviewing your step-by-step reasoning. Here's a possible model chain of thought:\n"
        feedback += f"\n**Your Reasoning:**\n```\n{student_cot}\n```\n\n**Model Reasoning:**\n```\n{model_cot}\n```\n"
    
    if student_answer.strip().lower() == model_answer.strip().lower():
        feedback += "\nYour final answer is correct!\n"
    else:
        feedback += f"\nYour final answer is incorrect. The correct answer was: {model_answer}\n"
    
    return feedback

def main():
    st.set_page_config(page_title="Intelligent Tutoring System - CoT Prompting")
    st.title("🧠 Intelligent Tutoring System")
    st.subheader("Improve your problem-solving skills with Chain-of-Thought Prompting!")

    if "current_problem_index" not in st.session_state:
        st.session_state.current_problem_index = 0
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False
    if "student_cot_input" not in st.session_state:
        st.session_state.student_cot_input = ""
    if "student_answer_input" not in st.session_state:
        st.session_state.student_answer_input = ""

    current_problem = PROBLEMS[st.session_state.current_problem_index]

    st.markdown(f"### Problem {current_problem['id']}")
    st.write(current_problem["question"])

    st.markdown("--- Say your thoughts step-by-step ---")
    student_cot = st.text_area("Enter your step-by-step reasoning here ('Let's think step by step.'):", 
                               height=200, 
                               key="cot_input",
                               value=st.session_state.student_cot_input)
    st.session_state.student_cot_input = student_cot

    student_answer = st.text_input("Enter your final answer:", 
                                   key="answer_input",
                                   value=st.session_state.student_answer_input)
    st.session_state.student_answer_input = student_answer

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Submit Solution", type="primary"):
            if student_cot and student_answer:
                model_cot, model_answer = simulate_llm_cot(current_problem["id"])
                st.session_state.feedback = generate_feedback(student_cot, student_answer, model_cot, model_answer)
                st.session_state.show_feedback = True
            else:
                st.warning("Please provide both your step-by-step reasoning and your final answer.")

    with col2:
        if st.button("Next Problem"):
            st.session_state.current_problem_index = (st.session_state.current_problem_index + 1) % len(PROBLEMS)
            st.session_state.show_feedback = False
            st.session_state.student_cot_input = ""
            st.session_state.student_answer_input = ""
            st.experimental_rerun()

    if st.session_state.show_feedback:
        st.markdown(st.session_state.feedback)

if __name__ == "__main__":
    main()