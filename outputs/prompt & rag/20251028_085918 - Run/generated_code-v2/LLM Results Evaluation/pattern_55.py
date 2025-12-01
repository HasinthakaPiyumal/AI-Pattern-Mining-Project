from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import gradio as gr

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

role_definitions = {
    "High School History Teacher": "As a high school history teacher, evaluate the content for historical accuracy, clarity for a 10th-grade audience, relevance to curriculum, and potential for engaging classroom discussion. Provide constructive feedback on how to improve its pedagogical effectiveness.",
    "10th Grade Student": "As a 10th-grade student, read the content and evaluate its understandability, interestingness, and whether it holds your attention. Is it too difficult, too easy, or just right? What parts are confusing or boring?",
    "Parent": "As a parent, assess the content for appropriateness, educational value, and potential biases. Is the language clear and concise? Would this content be beneficial for my child's learning? Is it safe and reliable?",
    "Subject Matter Expert": "As a subject matter expert in the relevant field, critically evaluate the content for factual accuracy, depth of information, currency, and sophisticated understanding of the topic. Identify any inaccuracies, oversimplifications, or missing crucial details."
}

def evaluate_content_with_roles(content: str, selected_roles: list) -> str:
    if not selected_roles:
        return "Please select at least one role for evaluation."

    evaluations = {}
    for role_name in selected_roles:
        if role_name in role_definitions:
            role_prompt_instruction = role_definitions[role_name]
            prompt_template = PromptTemplate(
                input_variables=["role_instruction", "content"],
                template="Act as {role_instruction}\n\nEvaluate the following educational content:\n\nContent: {content}\n\nProvide a detailed evaluation from your perspective, focusing on accuracy, clarity, engagement, and pedagogical effectiveness. Suggest specific improvements."
            )
            chain = LLMChain(llm=llm, prompt=prompt_template)
            response = chain.invoke({"role_instruction": role_prompt_instruction, "content": content})
            evaluations[role_name] = response["text"]
        else:
            evaluations[role_name] = "Role definition not found."

    formatted_output = ""
    for role, evaluation_text in evaluations.items():
        formatted_output += f"## Evaluation by {role}:\n\n{evaluation_text}\n\n---\n\n"
    return formatted_output

# Gradio Interface
iface = gr.Interface(
    fn=evaluate_content_with_roles,
    inputs=[
        gr.Textbox(lines=10, label="Educational Content to Evaluate", placeholder="Paste your educational content here..."),
        gr.CheckboxGroup(list(role_definitions.keys()), label="Select Evaluator Roles")
    ],
    outputs=gr.Markdown(label="Evaluation Results"),
    title="Educational Content Quality Assessor",
    description="Evaluate educational content from multiple perspectives using AI-powered role-based evaluators."
)

if __name__ == "__main__":
    iface.launch()