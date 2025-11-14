import os
from dotenv import load_dotenv
from typing import List, Dict, TypedDict, Annotated, Sequence
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
import operator


# 1. Load environment variables
load_dotenv()

# 2. Pydantic Models for Structured Output
class Feedback(BaseModel):
    persona: str = Field(description="The persona of the LLM providing this feedback, e.g., 'Grammar and Syntax Expert'.")
    feedback_points: List[str] = Field(description="List of specific feedback points related to the essay.")
    score: int = Field(description="Score given by this persona (e.g., out of 25).")

class EssayEvaluation(BaseModel):
    essay_text: str = Field(description="The original essay text that was evaluated.")
    individual_feedback: List[Feedback] = Field(description="List of feedback and scores from each individual evaluator persona.")
    final_score: int = Field(description="The aggregated final score for the essay, out of 100.")
    overall_report: str = Field(description="A comprehensive report synthesizing all individual feedback and providing an overall assessment.")

# 3. LLM Initialization
def initialize_llm(model_name: str = "gpt-4o", temperature: float = 0.5):
    """Initializes and returns a ChatOpenAI instance."""
    # Ensure OPENAI_API_KEY is set in your .env file or environment variables
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return ChatOpenAI(model_name=model_name, temperature=temperature)

# Initialize the primary LLM
llm = initialize_llm()

# 4. Evaluator Agent Definitions
def create_evaluator_agent(persona_name: str, scoring_criteria: str) -> Runnable:
    """
    Creates a LangChain Runnable for an individual evaluator agent.
    Each agent uses a PydanticOutputParser to ensure structured feedback.
    """
    parser = PydanticOutputParser(pydantic_object=Feedback)

    # The system prompt explicitly tells the LLM to include the 'persona' field
    system_template = f"""You are an AI acting as a {persona_name}. Your task is to evaluate an essay based on your specialized criteria.
    {scoring_criteria}
    You must provide concrete feedback points and assign a score out of 25.
    You MUST include the 'persona' field in your JSON output, set exactly to '{persona_name}'.
    {{format_instructions}}
    """
    human_template = "Here is the essay to evaluate:\n\n{{essay}}"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]).partial(format_instructions=parser.get_format_instructions())

    return (
        prompt
        | llm
        | parser
    )

# Define specific evaluator agents
grammar_agent = create_evaluator_agent(
    "Grammar and Syntax Expert",
    "Focus on grammar, spelling, punctuation, sentence structure, and clarity of expression. Penalize errors heavily. A perfect essay in your domain gets 25."
)

content_agent = create_evaluator_agent(
    "Content and Cohesion Analyst",
    "Focus on the essay's arguments, logical flow, relevance to the prompt, depth of analysis, and overall coherence. Assess how well ideas are connected and developed. A perfect essay in your domain gets 25."
)

critical_thinking_agent = create_evaluator_agent(
    "Critical Thinking Evaluator",
    "Focus on the essay's critical analysis, originality of thought, ability to evaluate different perspectives, and the strength of its reasoning. Does it demonstrate independent thought? A perfect essay in your domain gets 25."
)

style_agent = create_evaluator_agent(
    "Clarity and Style Reviewer",
    "Focus on the essay's writing style, word choice, tone, engagement, and overall readability. Is the language precise and impactful? Is it enjoyable to read? A perfect essay in your domain gets 25."
)

# 5. Synthesis Agent Definition
def create_synthesis_agent() -> Runnable:
    """
    Creates a LangChain Runnable for the synthesis agent (Chief Academic Grader).
    This agent aggregates individual feedback into a comprehensive report and final score.
    """
    parser = PydanticOutputParser(pydantic_object=EssayEvaluation)

    system_template = """You are an AI acting as the Chief Academic Grader. Your role is to synthesize feedback from multiple expert evaluators and provide a comprehensive overall assessment and final score for an essay.
    You will receive individual feedback and scores from a Grammar and Syntax Expert, a Content and Cohesion Analyst, a Critical Thinking Evaluator, and a Clarity and Style Reviewer.
    Your task is to:
    1. Review all individual feedback and scores provided in JSON format.
    2. Identify common themes, strengths, and weaknesses across all perspectives.
    3. Calculate a final overall score (out of 100). This should be a direct sum of the individual scores, as each contributes equally to the total. (e.g., if four agents score out of 25, total is out of 100).
    4. Write a comprehensive 'overall_report' that summarizes the essay's performance across all dimensions, highlights key areas for improvement, and justifies the final score. The 'overall_report' should be a well-structured paragraph or two.
    {{format_instructions}}
    """
    human_template = "Original Essay:\n\n{essay_text}\n\nIndividual Feedback (JSON format, separated by '---'):\n\n{individual_feedback_json}"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]).partial(format_instructions=parser.get_format_instructions())

    return (
        prompt
        | llm
        | parser
    )

synthesis_agent = create_synthesis_agent()


# 6. LangGraph State Definition
class AgentState(TypedDict):
    """
    Represents the state of our graph.

    - `essay`: The raw essay text.
    - `individual_feedback`: A list of Feedback objects from individual evaluators.
      `operator.add` is used to accumulate feedback from parallel nodes.
    """
    essay: str
    individual_feedback: Annotated[List[Feedback], operator.add] 

# 7. Graph Nodes
def essay_input_node(state: AgentState) -> AgentState:
    """Initializes the state with the essay and an empty list for feedback."""
    # Ensure individual_feedback starts empty when a new essay is processed
    return {"essay": state["essay"], "individual_feedback": []}

def run_grammar_evaluator(state: AgentState) -> AgentState:
    """Runs the Grammar and Syntax Expert agent and adds its feedback to the state."""
    feedback = grammar_agent.invoke({"essay": state["essay"]})
    return {"individual_feedback": [feedback]}

def run_content_evaluator(state: AgentState) -> AgentState:
    """Runs the Content and Cohesion Analyst agent and adds its feedback to the state."""
    feedback = content_agent.invoke({"essay": state["essay"]})
    return {"individual_feedback": [feedback]}

def run_critical_thinking_evaluator(state: AgentState) -> AgentState:
    """Runs the Critical Thinking Evaluator agent and adds its feedback to the state."""
    feedback = critical_thinking_agent.invoke({"essay": state["essay"]})
    return {"individual_feedback": [feedback]}

def run_style_evaluator(state: AgentState) -> AgentState:
    """Runs the Clarity and Style Reviewer agent and adds its feedback to the state."""
    feedback = style_agent.invoke({"essay": state["essay"]})
    return {"individual_feedback": [feedback]}

def run_synthesis_node(state: AgentState) -> EssayEvaluation:
    """
    Runs the Chief Academic Grader (synthesis) agent to combine all individual feedback
    and produce a final evaluation.
    """
    # Convert list of Pydantic objects to JSON strings for the prompt
    individual_feedback_json_list = [f.json() for f in state["individual_feedback"]]
    
    return synthesis_agent.invoke({
        "essay_text": state["essay"],
        "individual_feedback_json": "\n---\n".join(individual_feedback_json_list) # Use a clear separator
    })

# 8. Graph Construction (LangGraph)
workflow = StateGraph(AgentState)

# Add nodes for input and each evaluator
workflow.add_node("essay_input", essay_input_node)
workflow.add_node("grammar_eval", run_grammar_evaluator)
workflow.add_node("content_eval", run_content_evaluator)
workflow.add_node("critical_eval", run_critical_thinking_evaluator)
workflow.add_node("style_eval", run_style_evaluator)
workflow.add_node("synthesis", run_synthesis_node)

# Set the entry point of the graph
workflow.set_entry_point("essay_input")

# Define edges for parallel execution of evaluators
# After essay_input, all evaluator nodes are triggered.
workflow.add_edge("essay_input", "grammar_eval")
workflow.add_edge("essay_input", "content_eval")
workflow.add_edge("essay_input", "critical_eval")
workflow.add_edge("essay_input", "style_eval")

# All evaluator nodes feed into the synthesis node. LangGraph handles the accumulation
# in 'individual_feedback' due to Annotated[List[Feedback], operator.add].
# The synthesis node will only run after all its predecessors have completed.
workflow.add_edge("grammar_eval", "synthesis")
workflow.add_edge("content_eval", "synthesis")
workflow.add_edge("critical_eval", "synthesis")
workflow.add_edge("style_eval", "synthesis")

# The synthesis node marks the end of the graph
workflow.add_edge("synthesis", END)

# Compile the graph to create the executable application
app = workflow.compile()

# 9. Main execution block (Example Usage)
if __name__ == "__main__":
    example_essay = """
    The impact of artificial intelligence on society is a topic of immense importance, deserving careful consideration from multiple perspectives. While AI promises advancements in healthcare, transportation, and efficiency, its rapid development also raises significant ethical concerns. Job displacement due to automation is a pressing issue, potentially widening economic inequality if not managed proactively. Furthermore, the autonomous nature of advanced AI systems introduces questions of accountability and control, particularly in critical sectors. Bias embedded in training data can perpetuate and even amplify existing societal prejudices, leading to unfair outcomes. However, the potential for AI to solve complex global challenges, such as climate change or disease eradication, cannot be understated. Striking a balance between innovation and responsible deployment is crucial for harnessing AI's benefits while mitigating its risks. Education and retraining programs are essential to prepare the workforce for an AI-driven future, and robust regulatory frameworks are needed to guide ethical development.
    """

    print("--- Starting Essay Evaluation ---")
    
    # Invoke the graph with the initial state containing the essay
    try:
        final_output: EssayEvaluation = app.invoke({"essay": example_essay})

        print("\n--- Final Essay Evaluation Report ---")
        print(f"Original Essay (excerpt): {final_output.essay_text[:200]}...")
        print(f"Overall Final Score: {final_output.final_score}/100")
        print("\nIndividual Evaluator Feedback:")
        for feedback in final_output.individual_feedback:
            print(f"\nPersona: {feedback.persona} (Score: {feedback.score}/25)")
            for point in feedback.feedback_points:
                print(f"- {point}")
        print("\nComprehensive Overall Report:")
        print(final_output.overall_report)

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        print("Please ensure your OPENAI_API_KEY is correctly set in your .env file or environment variables.")


