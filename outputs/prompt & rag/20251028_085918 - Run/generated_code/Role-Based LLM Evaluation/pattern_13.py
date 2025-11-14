import os
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

class EssayGraderAgent:
    """Represents a single LLM grading agent with a specific persona."""
    def __init__(self, role: str, instructions: str, model: str = "gpt-4o"):
        self.role = role
        self.instructions = instructions
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def evaluate_essay(self, essay_text: str) -> str:
        """Sends the essay to the LLM with the agent's persona instructions."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a {self.role}. {self.instructions}"},
                    {"role": "user", "content": f"Please evaluate the following essay:\n\n{essay_text}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error during {self.role} evaluation: {e}"


class MultiPerspectiveEssayGrader:
    """Orchestrates multiple LLM agents to provide comprehensive essay feedback."""
    def __init__(self, model: str = "gpt-4o"):
        self.agents = [
            EssayGraderAgent(
                role="Grammar and Mechanics Expert",
                instructions="Your primary goal is to meticulously review the provided essay for all aspects of grammar, spelling, punctuation, and syntax. Identify all errors and suggest precise corrections. Provide feedback in a clear, bulleted list format, highlighting the error and the proposed fix.",
                model=model
            ),
            EssayGraderAgent(
                role="Content and Argument Analyst",
                instructions="Focus on the substance of the essay. Evaluate the clarity and strength of the thesis statement, the depth and relevance of supporting arguments, the use of evidence, and the logical consistency of the points made. Provide constructive feedback on how to strengthen the essay's core message and argumentation.",
                model=model
            ),
            EssayGraderAgent(
                role="Creativity and Style Judge",
                instructions="Assess the essay for originality, engaging language, unique perspectives, and overall writing style. Comment on the vocabulary, sentence variety, and the essay's ability to captivate the reader. Suggest ways to enhance creativity and stylistic flair.",
                model=model
            ),
            EssayGraderAgent(
                role="Structure and Organization Reviewer",
                instructions="Examine the overall structure of the essay, including the introduction, body paragraphs, transitions, and conclusion. Evaluate the logical flow between ideas and paragraphs, and how effectively the essay guides the reader. Provide recommendations for improving the essay's organization.",
                model=model
            )
        ]

    def grade_essay(self, essay_text: str) -> str:
        """Gathers and aggregates feedback from all grading agents."""
        all_feedback = []
        for agent in self.agents:
            feedback = agent.evaluate_essay(essay_text)
            all_feedback.append(f"### {agent.role} Feedback\n{feedback}\n")
        
        return "\n---\n".join(all_feedback)


def create_gradio_interface():
    """Creates and launches the Gradio web interface for the essay grader."""
    grader = MultiPerspectiveEssayGrader()

    def process_essay(essay_input: str) -> str:
        if not essay_input.strip():
            return "Please enter an essay to be graded."
        return grader.grade_essay(essay_input)

    iface = gr.Interface(
        fn=process_essay,
        inputs=gr.Textbox(lines=15, label="Submit Your Essay Here", placeholder="Paste your essay text..."),
        outputs=gr.Markdown(label="Comprehensive Essay Feedback"),
        title="📝 Multi-Perspective LLM Essay Grader",
        description="Get comprehensive feedback on your essay from multiple AI expert perspectives: Grammar, Content, Creativity, and Structure. (Powered by OpenAI GPT-4o)"
    )
    return iface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()
