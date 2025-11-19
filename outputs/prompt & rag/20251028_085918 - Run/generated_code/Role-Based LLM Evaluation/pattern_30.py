import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class CustomerSupportEvaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

        self.frustrated_customer_agent = Agent(
            role="Frustrated Customer Persona",
            goal="Evaluate customer support responses from the perspective of a highly frustrated customer seeking empathy and quick resolution.",
            backstory="You are an extremely frustrated customer who has encountered a recurring issue and is seeking a prompt, empathetic, and effective solution. Your patience is thin, and you value understanding and directness.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        self.technical_expert_agent = Agent(
            role="Technical Expert Persona",
            goal="Assess the technical accuracy, completeness, and clarity of the support response.",
            backstory="You are a seasoned technical support engineer. Your expertise lies in pinpointing technical inaccuracies, identifying missing information, and ensuring solutions are technically sound and easy to follow.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        self.company_policy_enforcer_agent = Agent(
            role="Company Policy Enforcer Persona",
            goal="Ensure the support response adheres strictly to company policies, brand guidelines, and legal compliance.",
            backstory="You are the guardian of company standards. Your primary focus is to verify that all customer interactions align with our terms of service, brand voice, and any relevant legal or regulatory requirements.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        self.efficiency_analyst_agent = Agent(
            role="Efficiency Analyst Persona",
            goal="Evaluate the support response for conciseness, clarity, and overall efficiency in resolving the issue.",
            backstory="You are an efficiency expert. You look for brevity without sacrificing clarity, ensuring the customer gets the information they need quickly and effectively, minimizing unnecessary back-and-forth.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def evaluate_ticket(self, customer_query: str, support_response: str):
        evaluate_empathy_task = Task(
            description=f"""
            Analyze the following customer support interaction:
            Customer Query: "{customer_query}"
            Support Response: "{support_response}"

            Evaluate the support response solely from the perspective of a frustrated customer.
            Assess:
            - Did the agent show empathy and understanding?
            - Was the tone appropriate for a frustrated customer?
            - Does the response make the customer feel heard and valued?
            Provide a score from 1 to 10 for 'Empathy and Understanding' and a brief, concise explanation.
            Example Output: "Empathy Score: 8/10. Explanation: The agent acknowledged the customer's frustration clearly."
            """,
            agent=self.frustrated_customer_agent,
            expected_output="A score for empathy and a brief explanation."
        )

        evaluate_technical_task = Task(
            description=f"""
            Analyze the following customer support interaction:
            Customer Query: "{customer_query}"
            Support Response: "{support_response}"

            Evaluate the support response solely for its technical accuracy, completeness, and clarity.
            Assess:
            - Is the technical information provided correct?
            - Are all relevant technical details included?
            - Is the technical explanation easy to understand for a non-expert?
            Provide a score from 1 to 10 for 'Technical Accuracy and Clarity' and a brief, concise explanation.
            Example Output: "Technical Score: 9/10. Explanation: All technical steps were accurate and well-explained."
            """,
            agent=self.technical_expert_agent,
            expected_output="A score for technical accuracy and clarity and a brief explanation."
        )

        evaluate_policy_task = Task(
            description=f"""
            Analyze the following customer support interaction:
            Customer Query: "{customer_query}"
            Support Response: "{support_response}"

            Evaluate the support response solely based on adherence to general company policies, brand guidelines, and ethical conduct.
            Assess:
            - Does the response maintain a professional and appropriate brand voice?
            - Are there any statements that violate company policy or legal requirements?
            - Is the proposed solution aligned with standard operating procedures?
            Provide a 'Policy Adherence' rating (e.g., 'Fully Compliant', 'Minor Deviation', 'Major Violation') and a brief, concise explanation.
            Example Output: "Policy Adherence: Fully Compliant. Explanation: Response used appropriate brand voice and followed all standard procedures."
            """,
            agent=self.company_policy_enforcer_agent,
            expected_output="A rating for policy adherence and a brief explanation."
        )

        evaluate_efficiency_task = Task(
            description=f"""
            Analyze the following customer support interaction:
            Customer Query: "{customer_query}"
            Support Response: "{support_response}"

            Evaluate the support response solely for its efficiency, conciseness, and effectiveness in reaching a resolution.
            Assess:
            - Is the response concise and to the point?
            - Does it avoid unnecessary jargon or lengthy explanations?
            - Does it clearly guide the customer towards a resolution or next steps without ambiguity?
            Provide a score from 1 to 10 for 'Efficiency and Clarity' and a brief, concise explanation.
            Example Output: "Efficiency Score: 7/10. Explanation: The response was clear but slightly longer than necessary for the given issue."
            """,
            agent=self.efficiency_analyst_agent,
            expected_output="A score for efficiency and clarity and a brief explanation."
        )

        crew = Crew(
            agents=[
                self.frustrated_customer_agent,
                self.technical_expert_agent,
                self.company_policy_enforcer_agent,
                self.efficiency_analyst_agent
            ],
            tasks=[
                evaluate_empathy_task,
                evaluate_technical_task,
                evaluate_policy_task,
                evaluate_efficiency_task
            ],
            process=Process.sequential,
            verbose=2
        )

        result = crew.kickoff()
        return result
