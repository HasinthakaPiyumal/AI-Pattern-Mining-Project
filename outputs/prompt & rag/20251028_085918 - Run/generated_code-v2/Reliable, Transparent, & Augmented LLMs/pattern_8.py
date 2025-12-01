import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import gradio as gr

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

initial_query_template = """You are a highly knowledgeable medical AI assistant. Provide a comprehensive and accurate answer to the following medical query:

Medical Query: {query}

Answer:"""

initial_query_prompt = PromptTemplate(template=initial_query_template, input_variables=["query"])
initial_query_chain = LLMChain(llm=llm, prompt=initial_query_prompt)

self_calibration_template = """You previously answered a medical query. Now, critically evaluate your own answer for correctness and provide a confidence score.

Original Medical Query: {question}
Your Initial Answer: {answer}

Based on the above, please provide your self-assessment in the following format:
Confidence: [0-100]% Recommendation: [Accept | Verify with Human Expert | Revise]
Reasoning: [Your brief reasoning for the confidence and recommendation]

Self-Assessment:"""

self_calibration_prompt = PromptTemplate(template=self_calibration_template, input_variables=["question", "answer"])
self_calibration_chain = LLMChain(llm=llm, prompt=self_calibration_prompt)

def assess_medical_query(query: str):
    initial_answer = initial_query_chain.run(query)
    self_assessment_raw = self_calibration_chain.run(question=query, answer=initial_answer)

    # Basic parsing of the self-assessment for display
    confidence_recommendation = "N/A"
    reasoning = "N/A"

    lines = self_assessment_raw.split('\n')
    for line in lines:
        if line.startswith("Confidence:"):
            confidence_recommendation = line.strip()
        elif line.startswith("Reasoning:"):
            reasoning = line.strip()
            
    full_self_assessment = f"{confidence_recommendation}\n{reasoning}"

    return initial_answer, full_self_assessment


iface = gr.Interface(
    fn=assess_medical_query,
    inputs=gr.Textbox(lines=5, label="Enter Medical Query"),
    outputs=[
        gr.Textbox(label="Initial LLM Answer"),
        gr.Textbox(label="LLM Self-Assessment (Confidence & Recommendation)")
    ],
    title="Medical Query Confidence Assessor",
    description="Enter a medical query and get an initial LLM answer, followed by the LLM's self-assessment of its own confidence and a recommendation."
)

iface.launch()