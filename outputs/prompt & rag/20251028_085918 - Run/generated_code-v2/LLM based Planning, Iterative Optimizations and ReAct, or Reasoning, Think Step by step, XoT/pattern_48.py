import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import LengthBasedExampleSelector

# Set your OpenAI API key from environment variables or direct assignment
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Ensure the API key is set
if "OPENAI_API_KEY" not in os.environ:
    st.error("OPENAI_API_KEY environment variable not set. Please set it to run this application.")
    st.stop()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# --- Example Problems for Few-Shot CoT --- 
examples = [
    {
        "question": "What is 10 divided by 2?",
        "thought": "This is a simple division problem. I need to find out how many times 2 fits into 10. 2 * 1 = 2, 2 * 2 = 4, 2 * 3 = 6, 2 * 4 = 8, 2 * 5 = 10.",
        "answer": "5"
    },
    {
        "question": "If a baker bakes 5 cakes per hour, how many cakes can they bake in 3 hours?",
        "thought": "This is a multiplication problem. To find the total number of cakes, I need to multiply the number of cakes per hour by the number of hours. Cakes per hour = 5. Number of hours = 3. Total cakes = 5 * 3 = 15.",
        "answer": "15"
    },
    {
        "question": "John has 12 apples. He gives 4 apples to Mary and eats 2. How many apples does John have left?",
        "thought": "This is a multi-step subtraction problem. \nStep 1: John starts with 12 apples.\nStep 2: He gives 4 to Mary, so he has 12 - 4 = 8 apples left.\nStep 3: He eats 2 apples, so he has 8 - 2 = 6 apples left.",
        "answer": "6"
    }
]

# --- Prompt Engineering for CoT --- 
example_prompt = PromptTemplate(
    input_variables=["question", "thought", "answer"],
    template="Question: {question}\nThought: {thought}\nAnswer: {answer}"
)

# Create a few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    example_selector=LengthBasedExampleSelector(
        examples=examples,
        get_text_func=lambda x: x["question"] + x["thought"] + x["answer"],
        max_length=250
    ),
    example_prompt=example_prompt,
    prefix="You are a helpful math tutor. Below are some examples of math problems and their step-by-step reasoning.\n",
    suffix="\nNow, solve the following problem, showing your thought process step-by-step.\nQuestion: {input_problem}\nThought:",
    input_variables=["input_problem"],
)

# Combined chat prompt template including system instruction and few-shot examples
full_cot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an excellent and patient math tutor for K-12 students. Always break down problems into logical, easy-to-understand steps."),
        ("user", few_shot_prompt)
    ]
)

# --- Streamlit Application --- 
st.title("🧠 AI-Powered Adaptive Math Tutor")
st.write("Enter a math problem, and I'll help you solve it by showing you the thought process!")

# Input for the math problem
student_problem = st.text_input("Your Math Problem:", placeholder="e.g., If a train travels 60 miles in 2 hours, how far does it travel in 5 hours?")

# Session state to store the generated CoT for feedback
if 'generated_cot' not in st.session_state:
    st.session_state['generated_cot'] = ""
if 'generated_answer' not in st.session_state:
    st.session_state['generated_answer'] = ""

if student_problem:
    if st.button("Get Step-by-Step Hint/Solution"):
        with st.spinner("Thinking step-by-step..."):
            # Generate CoT reasoning for the student's problem
            chain = full_cot_prompt | llm
            response_content = chain.invoke({"input_problem": student_problem}).content
            
            # Extract thought and answer from the LLM's response
            thought_start = response_content.find("Thought:")
            answer_start = response_content.find("Answer:")
            
            if answer_start != -1 and thought_start != -1:
                generated_thought = response_content[thought_start + len("Thought:"):answer_start].strip()
                generated_answer = response_content[answer_start + len("Answer:"):].strip()
            elif thought_start != -1:
                generated_thought = response_content[thought_start + len("Thought:"):].strip()
                generated_answer = "N/A (could not parse answer)"
            else:
                generated_thought = response_content
                generated_answer = "N/A (could not parse answer)"

            st.session_state['generated_cot'] = generated_thought
            st.session_state['generated_answer'] = generated_answer
            
            st.subheader("Tutor's Thought Process:")
            st.info(generated_thought)
            st.write(f"**Final Answer:** {generated_answer}")

    if st.session_state['generated_cot']:
        st.subheader("Your Turn!")
        st.write("Try to solve the problem or explain your steps based on the tutor's example.")
        student_attempt = st.text_area("Your solution or reasoning:", height=150)

        if st.button("Check My Answer"):
            if not student_attempt:
                st.warning("Please enter your solution or reasoning before checking.")
            else:
                with st.spinner("Evaluating your attempt..."):
                    # Use LLM to evaluate student's attempt against the generated CoT and provide feedback
                    feedback_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful and encouraging math tutor. Evaluate the student's attempt based on the correct reasoning provided. Point out where they went wrong or what they did well, and guide them towards the correct path without directly giving the answer again unless necessary for clarification. Focus on the reasoning process."),
                        ("user", f"Here's the problem: {student_problem}\nHere's the correct reasoning: {st.session_state['generated_cot']}\nHere's the student's attempt: {student_attempt}\n\nProvide constructive feedback on the student's attempt. Don't just give the final answer, focus on guiding their thought process.")
                    ])
                    feedback_chain = feedback_prompt | llm
                    feedback_response = feedback_chain.invoke({}).content
                    st.subheader("Tutor's Feedback:")
                    st.success(feedback_response)

# Instructions to run the app
st.sidebar.markdown("### How to Run")
st.sidebar.markdown("1. Make sure you have `streamlit` and `langchain-openai` installed: `pip install streamlit langchain-openai`")
st.sidebar.markdown("2. Set your OpenAI API key as an environment variable: `export OPENAI_API_KEY='your_key_here'`")
st.sidebar.markdown("3. Save the code as `math_tutor_app.py`")
st.sidebar.markdown("4. Run from your terminal: `streamlit run math_tutor_app.py`")