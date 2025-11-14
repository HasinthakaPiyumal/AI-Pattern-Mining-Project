import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

# Set your OpenAI API key from environment variables
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- LLM Initialization ---
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# --- Persona Definitions and Prompts ---
define_persona = lambda role, objective, output_format: f"You are a {role}. Your task is to {objective}. {output_format}"

grammar_specialist_prompt_template = ChatPromptTemplate.from_messages([
    ("system", define_persona(
        "Grammar Specialist",
        "meticulously review the provided essay for grammatical errors, spelling mistakes, punctuation errors, and syntax issues. Provide a list of identified errors and suggested corrections. Focus solely on correctness.",
        "Format your feedback as a bulleted list of issues and suggested fixes."
    )),
    ("human", "Evaluate the following essay: {essay_text}")
])

creative_writing_critic_prompt_template = ChatPromptTemplate.from_messages([
    ("system", define_persona(
        "Creative Writing Critic",
        "evaluate the provided essay for its creativity, originality, style, narrative flow, and engagement. Assess the effectiveness of literary devices, imagery, and overall artistic merit. Provide constructive feedback on how the essay could be more compelling or unique.",
        "Format your feedback as a detailed paragraph discussing strengths and areas for improvement in creative aspects."
    )),
    ("human", "Evaluate the following essay: {essay_text}")
])

logical_flow_analyst_prompt_template = ChatPromptTemplate.from_messages([
    ("system", define_persona(
        "Logical Flow Analyst",
        "assess the coherence, logical progression of arguments, transitions between paragraphs, and overall structural integrity of the essay. Identify any breaks in logic, unsupported claims, or areas where the essay's organization could be improved.",
        "Format your feedback as a paragraph detailing the logical structure and any recommended improvements."
    )),
    ("human", "Evaluate the following essay: {essay_text}")
])

subject_matter_expert_prompt_template = ChatPromptTemplate.from_messages([
    ("system", define_persona(
        "Subject Matter Expert in academic content",
        "evaluate the provided essay for factual accuracy, depth of understanding, relevance to the prompt, and appropriate use of evidence or examples. Assume the essay is on a general academic topic. Provide feedback on the content's validity and intellectual rigor.",
        "Format your feedback as a paragraph summarizing the content's strengths and weaknesses regarding accuracy and depth."
    )),
    ("human", "Evaluate the following essay: {essay_text}")
])

feedback_aggregator_prompt_template = ChatPromptTemplate.from_messages([
    ("system", define_persona(
        "Feedback Aggregator",
        "synthesize all this feedback into a single, comprehensive, and well-structured report for a student. Start with an overall summary, then present findings under clear headings for each specialist area, and conclude with actionable recommendations. Ensure the tone is constructive and encouraging.",
        ""
    )),
    ("human", "Here is the individual feedback:\n\nGrammar Specialist:\n{grammar_feedback}\n\nCreative Writing Critic:\n{creative_feedback}\n\nLogical Flow Analyst:\n{logical_feedback}\n\nSubject Matter Expert:\n{subject_feedback}\n\nPlease aggregate this into a single report.")
])

# --- Evaluator Agent Functions ---
def evaluate_with_persona(essay_text: str, prompt_template: ChatPromptTemplate) -> str:
    """Sends the essay to an LLM with a specific persona prompt and returns the feedback."""
    chain = prompt_template | llm
    response = chain.invoke({"essay_text": essay_text})
    return response.content

# --- Main Essay Evaluation Function ---
def evaluate_essay(essay_text: str) -> str:
    """Orchestrates the multi-perspective essay evaluation."""
    if not os.getenv("OPENAI_API_KEY"):
        return "Error: OPENAI_API_KEY environment variable not set. Please set your OpenAI API key."
    
    # Evaluate with each persona
    grammar_feedback = evaluate_with_persona(essay_text, grammar_specialist_prompt_template)
    creative_feedback = evaluate_with_persona(essay_text, creative_writing_critic_prompt_template)
    logical_feedback = evaluate_with_persona(essay_text, logical_flow_analyst_prompt_template)
    subject_feedback = evaluate_with_persona(essay_text, subject_matter_expert_prompt_template)

    # Aggregate feedback
    aggregation_chain = feedback_aggregator_prompt_template | llm
    aggregated_report = aggregation_chain.invoke({
        "grammar_feedback": grammar_feedback,
        "creative_feedback": creative_feedback,
        "logical_feedback": logical_feedback,
        "subject_feedback": subject_feedback,
    }).content

    return aggregated_report

# --- Gradio User Interface ---
if __name__ == "__main__":
    # Example usage for testing (replace with your actual essay)
    example_essay = """
The rapid advancement of artificial intelligence (AI) has sparked considerable debate regarding its potential impact on human employment. While some envision a future where AI automates routine tasks, freeing humans for more creative endeavors, others fear widespread job displacement and economic disruption. Historically, technological revolutions have often led to the creation of new industries and job roles, even as older ones become obsolete. However, the unprecedented cognitive capabilities of modern AI suggest a more profound shift. Therefore, understanding the nuances of AI integration is crucial for navigating this evolving landscape.
"""

    gr.Interface(
        fn=evaluate_essay,
        inputs=gr.Textbox(lines=15, label="Paste your essay here for evaluation", placeholder="Enter your essay..."),
        outputs=gr.Textbox(label="Multi-Perspective Evaluation Report", interactive=False),
        title="AI-Powered Multi-Perspective Essay Evaluator",
        description="Get comprehensive feedback on your essay from a Grammar Specialist, Creative Writing Critic, Logical Flow Analyst, and Subject Matter Expert, aggregated into a single report.",
        examples=[[example_essay]]
    ).launch()
