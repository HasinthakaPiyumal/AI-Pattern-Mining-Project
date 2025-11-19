import os
import gradio as gr
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_llm_response(query):
    prompt = f"""You are an intelligent customer support agent. Your task is to answer user queries accurately, provide the reasoning behind your answer, and rate your confidence in the accuracy of your answer.

User Query: {query}

Instructions:
1. Provide a direct and concise answer to the user's query.
2. Explain the steps or information you considered to arrive at this answer. If you would typically use an external tool (e.g., a product database or order system), describe what information you would seek from it.
3. Rate your confidence in the accuracy of your answer on a scale of 0-100%.

Format your response exactly as follows:
Answer: <Your Answer>
Reasoning: <Your Reasoning>
Confidence: <Your Confidence Score>%"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful and honest assistant."}, 
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500,
    )

    full_response_content = response.choices[0].message.content
    
    answer = "N/A"
    reasoning = "N/A"
    confidence = "N/A"

    try:
        lines = full_response_content.split('\n')
        for line in lines:
            if line.startswith("Answer:"):
                answer = line.replace("Answer:", "").strip()
            elif line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()
            elif line.startswith("Confidence:"):
                confidence = line.replace("Confidence:", "").strip()
    except Exception as e:
        print(f"Error parsing response: {e}")
        answer = full_response_content 
        reasoning = "Could not parse detailed reasoning."
        confidence = "Low (parsing error)"

    return answer, reasoning, confidence

def chatbot_interface(user_query):
    answer, reasoning, confidence = get_llm_response(user_query)
    return answer, reasoning, confidence

iface = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your customer support query here..."),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Reasoning"),
        gr.Textbox(label="Confidence Score"),
    ],
    title="Intelligent Customer Support Agent",
    description="Ask a question, and the AI agent will provide an answer, its reasoning, and a confidence score."
)

if __name__ == "__main__":
    iface.launch()