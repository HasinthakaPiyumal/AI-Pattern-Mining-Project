import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessage, HumanMessage

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in a .env file.")

llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

EVALUATOR_ROLES = {
    "English Literature Professor": "As an English Literature Professor, evaluate the essay's thematic depth, literary devices, argumentation, and overall critical analysis. Provide detailed feedback on areas for improvement in these aspects.",
    "Grammar and Syntax Specialist": "As a Grammar and Syntax Specialist, meticulously review the essay for grammatical errors, spelling mistakes, punctuation issues, sentence structure, and clarity. Highlight specific examples and suggest corrections.",
    "Creative Writing Coach": "As a Creative Writing Coach, assess the essay's originality, voice, narrative flow, imagery, and engagement. Offer advice on how to make the writing more impactful and imaginative.",
    "Peer Reviewer": "As a Peer Reviewer, provide constructive feedback from a student's perspective. Focus on clarity, understanding, and general effectiveness. What parts were confusing? What was compelling?"
}

def process_essay(essay_text: str) -> str:
    if not essay_text.strip():
        return "Please enter an essay to receive feedback."

    all_feedback = []

    for role, instructions in EVALUATOR_ROLES.items():
        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=instructions),
                HumanMessage(content=essay_text)
            ]
        )
        
        try:
            response = llm.invoke(chat_template.messages)
            feedback = response.content
            all_feedback.append(f"### Feedback from {role}:\n{feedback}\n")
        except Exception as e:
            all_feedback.append(f"### Error for {role}:\nCould not generate feedback due to an error: {e}\n")

    return "\n---\n".join(all_feedback)


iface = gr.Interface(
    fn=process_essay,
    inputs=gr.Textbox(lines=20, label="Enter your essay here", placeholder="Type or paste your essay..."),
    outputs=gr.Markdown(label="Role-based Feedback"),
    title="Personalized Essay Feedback System",
    description="Get diverse feedback on your essay from multiple AI-powered evaluators, each adopting a different role. Input your essay and click 'Submit' to receive insights from an English Literature Professor, Grammar Specialist, Creative Writing Coach, and Peer Reviewer."
)

iface.launch()