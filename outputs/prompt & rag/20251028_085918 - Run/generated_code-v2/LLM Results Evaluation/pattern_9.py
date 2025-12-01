import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# --- Pydantic Models for Structured Output ---
class AgentFeedback(BaseModel):
    agent_name: str = Field(description="Name of the evaluating agent.")
    assessment: str = Field(description="Detailed assessment from the agent's perspective.")
    score: int = Field(description="Score out of 10 for the specific aspect evaluated by the agent.")

class EvaluationReport(BaseModel):
    overall_score: int = Field(description="An overall quality score for the LLM response out of 10.")
    individual_feedback: list[AgentFeedback] = Field(description="List of feedback from each individual agent.")
    conflicting_points: str = Field(description="Identified conflicting points or disagreements among agents, if any.")
    final_recommendations: str = Field(description="Actionable recommendations for improving the LLM response.")

# --- LLM Setup ---
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") # Ensure API key is set
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# --- Agent Definitions ---
accuracy_checker = Agent(
    role="Expert in factual correctness and information alignment.",
    goal="Ensure the LLM response is factually accurate and directly addresses the customer's query without misinterpretations.",
    backstory="With a keen eye for detail, I meticulously cross-reference information to guarantee precision and relevance.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

empathy_assessor = Agent(
    role="Specialist in human-centered communication and emotional intelligence.",
    goal="Evaluate the LLM response for empathetic tone, understanding, and respectful language to ensure a positive customer experience.",
    backstory="I believe every customer deserves to feel heard and understood. My focus is on the warmth and appropriateness of the interaction.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

conciseness_reviewer = Agent(
    role="Master of brevity and clarity.",
    goal="Assess the LLM response for conciseness, clarity, and ease of understanding, eliminating jargon and superfluous details.",
    backstory="Time is precious. I ensure responses are direct, to the point, and deliver information efficiently without sacrificing completeness.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

policy_adherence_validator = Agent(
    role="Guardian of company guidelines and compliance.",
    goal="Verify that the LLM response strictly adheres to all predefined company policies, legal requirements, and best practices.",
    backstory="Rules are there for a reason. I ensure every word aligns with our operational standards and legal obligations.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

debate_moderator = Agent(
    role="Chief adjudicator and feedback consolidator.",
    goal="Synthesize feedback from all evaluative agents, identify conflicting points, and produce a final, comprehensive quality score and actionable improvement recommendations.",
    backstory="My job is to bring all perspectives together, resolve discrepancies, and deliver a holistic judgment with clear paths for enhancement.",
    verbose=True,
    allow_delegation=True, # Can delegate for deeper analysis if needed
    llm=llm,
    output_pydantic=EvaluationReport # Ensure structured output for the final report
)

# --- Task Definitions ---
accuracy_check_task = Task(
    description=(
        "Analyze the following LLM-generated response to a customer query for factual accuracy, completeness, and direct relevance to the query. "
        "Provide a summary of your findings and a score out of 10 for accuracy. "
        "Customer Query: {customer_query}\nLLM Response: {llm_generated_response}"
    ),
    agent=accuracy_checker,
    expected_output="A string summarizing accuracy findings and a score out of 10 (e.g., 'Accuracy: The response is highly accurate and directly answers the question. Score: 9/10')."
)

empathy_assessment_task = Task(
    description=(
        "Evaluate the empathetic tone, understanding, and respectful language of the LLM response. "
        "Provide feedback on whether the response acknowledges customer's feelings and maintains a positive interaction. "
        "Provide a summary of your findings and a score out of 10 for empathy. "
        "Customer Query: {customer_query}\nLLM Response: {llm_generated_response}"
    ),
    agent=empathy_assessor,
    expected_output="A string summarizing empathy findings and a score out of 10 (e.g., 'Empathy: The response is empathetic and acknowledges the customer's frustration. Score: 8/10')."
)

conciseness_review_task = Task(
    description=(
        "Assess the LLM response for conciseness, clarity, and ease of understanding. "
        "Identify any jargon, redundant information, or areas where the response could be more direct. "
        "Provide a summary of your findings and a score out of 10 for conciseness. "
        "Customer Query: {customer_query}\nLLM Response: {llm_generated_response}"
    ),
    agent=conciseness_reviewer,
    expected_output="A string summarizing conciseness findings and a score out of 10 (e.g., 'Conciseness: The response is clear but a bit lengthy. Score: 7/10')."
)

policy_validation_task = Task(
    description=(
        "Verify the LLM response against simulated company policy guidelines. "
        "For this task, assume the following policy: 'All responses must offer a refund for service outages lasting more than 4 hours, and must never promise specific resolution times unless explicitly authorized.' "
        "State if the response adheres to this policy and provide a score out of 10 for policy adherence. "
        "Customer Query: {customer_query}\nLLM Response: {llm_generated_response}"
    ),
    agent=policy_adherence_validator,
    expected_output="A string summarizing policy adherence and a score out of 10 (e.g., 'Policy Adherence: The response correctly offered a refund and avoided promising specific times. Score: 10/10')."
)

synthesis_and_debate_task = Task(
    description=(
        "Receive the evaluations from the Accuracy Checker, Empathy Assessor, Conciseness Reviewer, and Policy Adherence Validator. "
        "Synthesize their feedback, identify any conflicting points or areas of disagreement, and based on all input, determine an overall quality score out of 10 for the LLM response. "
        "Finally, provide actionable recommendations for improving the LLM response to ensure it meets high standards across all evaluated aspects. "
        "Use the provided Pydantic model 'EvaluationReport' for your final output."
    ),
    agent=debate_moderator,
    context=[accuracy_check_task, empathy_assessment_task, conciseness_review_task, policy_validation_task],
    output_pydantic=EvaluationReport,
    expected_output="A JSON object adhering to the EvaluationReport Pydantic model, containing overall_score, individual_feedback, conflicting_points, and final_recommendations."
)

# --- Simulated Data ---
customer_query = "My internet has been down for 6 hours! What are you going to do about this? I'm very frustrated."
llm_generated_response = (
    "I apologize for the inconvenience you've experienced with your internet service being down for 6 hours. "
    "We understand how frustrating this can be. As per our company policy for outages exceeding 4 hours, "
    "we will be issuing a full refund for today's service charges to your account. "
    "Our technical team is actively working to restore service, and we anticipate it will be resolved within the next 2-3 hours. "
    "We appreciate your patience as we work to resolve this for you."
)

# --- Crew Definition and Execution ---
customer_support_eval_crew = Crew(
    agents=[
        accuracy_checker,
        empathy_assessor,
        conciseness_reviewer,
        policy_adherence_validator,
        debate_moderator
    ],
    tasks=[
        accuracy_check_task,
        empathy_assessment_task,
        conciseness_review_task,
        policy_validation_task,
        synthesis_and_debate_task
    ],
    process=Process.sequential,
    verbose=2  # Set to 1 for less verbose output, 2 for full output
)

print("### Initiating LLM Response Quality Assurance Debate ###")
result = customer_support_eval_crew.kickoff(
    inputs={
        "customer_query": customer_query,
        "llm_generated_response": llm_generated_response
    }
)

print("\n### Final Evaluation Report ###")
print(result.model_dump_json(indent=2))
