import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

class EssayRequest(BaseModel):
    essay_text: str

class EvaluationDetail(BaseModel):
    role: str
    feedback: str

class EvaluationResponse(BaseModel):
    evaluations: list[EvaluationDetail]

def get_llm_response_mock(prompt_template_str: str, essay: str) -> str:
    if "critical academic" in prompt_template_str.lower():
        return "As a critical academic, I find the argument structure of this essay somewhat weak. The introduction lacks a clear thesis statement, and some claims are not adequately supported by evidence. Consider strengthening your topic sentences and providing more in-depth analysis of your sources. The conclusion is abrupt and doesn't fully synthesize the points discussed."
    elif "supportive mentor" in prompt_template_str.lower():
        return "You've made a great start on this essay! I particularly like the initial ideas presented. To take it to the next level, perhaps expand on your examples a bit more to really drive your points home. Don't be afraid to show more of your unique voice. Keep up the good work!"
    elif "grammar specialist" in prompt_template_str.lower():
        return "From a grammar perspective, the essay is generally clear. I noticed a few instances of subject-verb agreement errors and some awkward phrasing. Pay close attention to comma usage, especially with introductory clauses. Running a spell check and a grammar checker would be beneficial."
    elif "creative writing coach" in prompt_template_str.lower():
        return "As a creative writing coach, I see potential for more vivid imagery and engaging language. While the content is informative, try to incorporate more storytelling elements or unique metaphors to captivate your reader. Experiment with sentence structure variety to improve flow and rhythm."
    else:
        return f"General feedback for the essay: {essay[:100]}..."


@app.post("/evaluate-essay", response_model=EvaluationResponse)
async def evaluate_essay(request: EssayRequest):
    essay_text = request.essay_text
    evaluations = []

    roles = [
        ("critical academic", "You are a critical academic professor. Provide a rigorous and constructive evaluation of the essay's arguments, structure, evidence, and overall academic rigor. Point out weaknesses and suggest improvements from an academic standpoint."),
        ("supportive mentor", "You are a supportive and encouraging mentor. Provide feedback that highlights the essay's strengths and offers gentle suggestions for improvement, focusing on fostering the student's growth and confidence."),
        ("grammar specialist", "You are a meticulous grammar and style specialist. Focus solely on grammatical errors, punctuation, spelling, sentence structure, word choice, and adherence to formal writing conventions. Provide clear corrections or suggestions."),
        ("creative writing coach", "You are a creative writing coach. Evaluate the essay's originality, voice, storytelling, descriptive language, and overall engagement. Suggest ways to make the writing more captivating, imaginative, and impactful.")
    ]

    for role_name, role_persona in roles:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", role_persona + " Evaluate the following essay."),
                ("human", "{essay}")
            ]
        )
        formatted_prompt_str = prompt_template.format(essay=essay_text)
        
        feedback = get_llm_response_mock(role_persona, essay_text)

        evaluations.append(EvaluationDetail(role=role_name, feedback=feedback))

    return EvaluationResponse(evaluations=evaluations)

def run_streamlit_app():
    st.set_page_config(layout="wide")
    st.title("AI-Powered Essay Evaluator")
    st.markdown("Upload your essay or paste the text below to receive diverse feedback from AI evaluators playing different roles.")

    essay_input = st.text_area("Paste your essay here:", height=300)

    if st.button("Get Evaluation"):
        if essay_input:
            st.info("Generating diverse evaluations... This may take a moment.")
            try:
                response = requests.post(
                    "http://localhost:8000/evaluate-essay",
                    json={"essay_text": essay_input},
                    timeout=60
                )
                response.raise_for_status()
                evaluation_results = response.json()

                st.subheader("Evaluation Results:")
                for eval_detail in evaluation_results["evaluations"]:
                    st.markdown(f"**Role: {eval_detail['role'].replace('_', ' ').title()}**")
                    st.write(eval_detail['feedback'])
                    st.markdown("--- ")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the evaluation backend. Please ensure the FastAPI server is running (check terminal).")
            except requests.exceptions.Timeout:
                st.error("The request to the evaluation backend timed out. The essay might be too long or the server is slow.")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred during evaluation: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Please enter some essay text to get an evaluation.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        run_streamlit_app()
    elif len(sys.argv) > 1 and sys.argv[1] == "fastapi":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("To run the FastAPI backend: python essay_evaluator.py fastapi")
        print("To run the Streamlit frontend: python essay_evaluator.py streamlit")
        print("Ensure the FastAPI backend is running before starting the Streamlit frontend.")