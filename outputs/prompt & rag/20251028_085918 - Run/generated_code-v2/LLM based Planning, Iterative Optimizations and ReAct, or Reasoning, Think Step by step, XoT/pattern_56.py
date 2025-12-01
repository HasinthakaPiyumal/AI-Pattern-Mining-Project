import streamlit as st
import openai
import os
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from collections import Counter
import random

COMPLEX_EXAMPLES = {
    "problem_1": {
        "question": "A spherical balloon is being inflated. Its radius is increasing at a rate of 2 cm/s. At what rate is the volume of the balloon increasing when its radius is 10 cm? (Volume of a sphere V = (4/3)πr^3)",
        "steps": [
            "1. Identify the given information: dr/dt = 2 cm/s, r = 10 cm.",
            "2. Identify what needs to be found: dV/dt.",
            "3. Write the formula for the volume of a sphere: V = (4/3)πr^3.",
            "4. Differentiate the volume formula with respect to time (t) using the chain rule: dV/dt = (4/3)π * 3r^2 * dr/dt = 4πr^2 * dr/dt.",
            "5. Substitute the given values into the differentiated formula: dV/dt = 4π(10 cm)^2 * (2 cm/s).",
            "6. Calculate the result: dV/dt = 4π(100 cm^2) * (2 cm/s) = 800π cm^3/s.",
            "Answer: The volume is increasing at a rate of 800π cm^3/s."
        ],
        "answer": "800π cm^3/s"
    },
    "problem_2": {
        "question": "Find the general solution to the differential equation dy/dx + 2xy = x.",
        "steps": [
            "1. Identify the form of the differential equation: It is a first-order linear differential equation, dy/dx + P(x)y = Q(x), where P(x) = 2x and Q(x) = x.",
            "2. Calculate the integrating factor, μ(x) = e^(∫P(x)dx).",
            "3. ∫P(x)dx = ∫2x dx = x^2.",
            "4. So, μ(x) = e^(x^2).",
            "5. Multiply the entire differential equation by the integrating factor: e^(x^2) * (dy/dx + 2xy) = x * e^(x^2).",
            "6. The left side is the derivative of (y * μ(x)): d/dx (y * e^(x^2)) = x * e^(x^2).",
            "7. Integrate both sides with respect to x: ∫d/dx (y * e^(x^2)) dx = ∫x * e^(x^2) dx.",
            "8. For the right integral, use a substitution: let u = x^2, then du = 2x dx, so (1/2)du = x dx.",
            "9. ∫x * e^(x^2) dx = ∫e^u * (1/2)du = (1/2)e^u + C = (1/2)e^(x^2) + C.",
            "10. So, y * e^(x^2) = (1/2)e^(x^2) + C.",
            "11. Solve for y: y = (1/2) + C * e^(-x^2).",
            "Answer: The general solution is y = 1/2 + C*e^(-x^2)."
        ],
        "answer": "y = 1/2 + C*e^(-x^2)"
    }
}

llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

def assess_complexity(query: str) -> bool:
    return len(query.split()) > 15 or len(query.split('.')) > 2

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Problem: {question}"),
        ("ai", "Steps: {steps}\nAnswer: {answer}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=[]
)

main_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert STEM tutor. Provide detailed, step-by-step reasoning to solve complex problems. When multiple steps are involved, break them down clearly."),
        few_shot_prompt,
        ("human", "Problem: {question}"),
    ]
)

def generate_reasoning_chains(question: str, num_chains: int = 5, use_complex_examples: bool = False) -> list[str]:
    examples_for_prompt = []
    if use_complex_examples:
        for key in COMPLEX_EXAMPLES:
            examples_for_prompt.append({
                "question": COMPLEX_EXAMPLES[key]["question"],
                "steps": "\n".join(COMPLEX_EXAMPLES[key]["steps"]),
                "answer": COMPLEX_EXAMPLES[key]["answer"]
            })
    
    few_shot_prompt.examples = examples_for_prompt
    
    chain = main_prompt_template | llm | StrOutputParser()
    
    generated_chains = []
    for _ in range(num_chains):
        response = chain.invoke({"question": question})
        generated_chains.append(response)
    return generated_chains

def aggregate_solutions(solutions: list[str], length_threshold: int = 150) -> str:
    filtered_solutions = [s for s in solutions if len(s.split()) >= length_threshold / 2]
    
    if not filtered_solutions:
        if solutions:
            return max(solutions, key=len)
        return "Could not generate a coherent solution. Please try rephrasing the problem."

    answers = []
    for sol in filtered_solutions:
        answer_line = next((line for line in sol.split('\n') if line.strip().startswith("Answer:")), None)
        if answer_line:
            answers.append(answer_line.replace("Answer:", "").strip())
    
    if answers:
        most_common_answer = Counter(answers).most_common(1)
        if most_common_answer:
            for sol in filtered_solutions:
                if most_common_answer[0][0] in sol:
                    return sol
    
    return max(filtered_solutions, key=len)

st.set_page_config(page_title="AI-Powered Personalized STEM Tutor")
st.title("📚 AI-Powered Personalized STEM Tutor")
st.markdown("Enter a complex STEM problem, and I'll help you solve it step-by-step using advanced reasoning!")

problem_input = st.text_area("Enter your STEM problem here:", height=150)

if st.button("Get Solution"):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set your OPENAI_API_KEY environment variable to use this application.")
    elif problem_input:
        with st.spinner("Analyzing and generating solution... This may take a moment."):
            is_complex = assess_complexity(problem_input)
            
            generated_chains = generate_reasoning_chains(
                question=problem_input,
                num_chains=5,
                use_complex_examples=is_complex
            )
            
            final_solution = aggregate_solutions(generated_chains)
            
            st.subheader("💡 Our Step-by-Step Solution:")
            st.write(final_solution)
            
            st.subheader("🔍 Behind the Scenes:")
            st.write(f"Complexity detected: {'Yes' if is_complex else 'No'}")
            st.write(f"Number of reasoning chains generated: {len(generated_chains)}")
    else:
        st.warning("Please enter a STEM problem to get a solution.")
