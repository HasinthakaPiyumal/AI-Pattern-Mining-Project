import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

def main():
    load_dotenv()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    if not os.environ["OPENAI_API_KEY"]:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    evaluator_manager = Agent(
        role="Evaluation Manager",
        goal="Facilitate a comprehensive evaluation of customer support resolution quality by orchestrating a multi-agent debate and synthesizing a final report.",
        backstory=(
            "You are the central figure in a customer support evaluation team. "
            "Your primary responsibility is to present customer interactions, "
            "manage the debate among different evaluators, and compile a conclusive report "
            "that highlights areas of improvement for the chatbot."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    frustrated_customer = Agent(
        role="Frustrated Customer Persona",
        goal="Evaluate the chatbot's response from the perspective of a highly frustrated customer, prioritizing empathy, quick resolution, and acknowledgement of their distress.",
        backstory=(
            "You've had a terrible experience and are very upset. "
            "You expect not just a solution, but also genuine understanding and a swift, "
            "hassle-free resolution to your problem. You are quick to point out "
            "any lack of empathy or slow service."
        ),
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    detail_oriented_customer = Agent(
        role="Detail-Oriented Customer Persona",
        goal="Assess the chatbot's response for accuracy, completeness, and clarity of information, ensuring all specific queries are thoroughly addressed.",
        backstory=(
            "You value precision and comprehensive information. "
            "You scrutinize every detail of the chatbot's response, "
            "looking for exact answers, clear instructions, and ensuring all "
            "your specific questions have been fully answered without ambiguity."
        ),
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    impatient_customer = Agent(
        role="Impatient Customer Persona",
        goal="Evaluate the chatbot's response based on efficiency, conciseness, and speed of resolution, favoring direct answers and minimal back-and-forth.",
        backstory=(
            "You have very little time and expect quick, to-the-point answers. "
            "Any unnecessary fluff, delays, or requests for redundant information "
            "will be met with disapproval. Efficiency is your top priority."
        ),
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    efficiency_expert = Agent(
        role="Efficiency Expert Supervisor",
        goal="Evaluate the chatbot's response for operational efficiency, resource utilization, and adherence to streamlined processes, seeking cost-effective and swift resolutions.",
        backstory=(
            "As a supervisor focused on operational metrics, you analyze customer interactions "
            "to ensure the chatbot is resolving issues efficiently, minimizing steps, "
            "and optimizing resource allocation. You look for ways to reduce resolution time and operational costs."
        ),
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    customer_satisfaction_specialist = Agent(
        role="Customer Satisfaction Specialist Supervisor",
        goal="Assess the chatbot's response for its positive impact on customer sentiment, empathy, tone, and long-term customer loyalty.",
        backstory=(
            "Your focus is entirely on the customer's emotional journey and satisfaction. "
            "You evaluate how well the chatbot builds rapport, demonstrates empathy, "
            "and leaves the customer feeling valued and likely to return. Tone and sentiment are key."
        ),
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    customer_query_template = "{customer_query}"
    chatbot_response_template = "{chatbot_response}"
    interaction_context = f"Customer Query: {customer_query_template}\nChatbot Response: {chatbot_response_template}"

    analyze_frustrated = Task(
        description=(
            f"Analyze the following customer support interaction:\n{interaction_context}\n\n"
            "From the perspective of a frustrated customer, provide an initial assessment "
            "of the chatbot's response. Focus on empathy, speed of resolution, and "
            "whether your distress was acknowledged. Be critical if these aspects are lacking. "
            "Your output should be a detailed paragraph on your assessment."
        ),
        agent=frustrated_customer,
        expected_output="A detailed paragraph assessing the chatbot response from a frustrated customer's viewpoint."
    )

    analyze_detailed = Task(
        description=(
            f"Analyze the following customer support interaction:\n{interaction_context}\n\n"
            "From the perspective of a detail-oriented customer, provide an initial assessment "
            "of the chatbot's response. Focus on accuracy, completeness, and clarity "
            "of information provided. Point out any ambiguities or missing details. "
            "Your output should be a detailed paragraph on your assessment."
        ),
        agent=detail_oriented_customer,
        expected_output="A detailed paragraph assessing the chatbot response from a detail-oriented customer's viewpoint."
    )

    analyze_impatient = Task(
        description=(
            f"Analyze the following customer support interaction:\n{interaction_context}\n\n"
            "From the perspective of an impatient customer, provide an initial assessment "
            "of the chatbot's response. Focus on efficiency, conciseness, and directness "
            "of the resolution. Criticize any unnecessary steps or verbosity. "
            "Your output should be a detailed paragraph on your assessment."
        ),
        agent=impatient_customer,
        expected_output="A detailed paragraph assessing the chatbot response from an impatient customer's viewpoint."
    )

    analyze_efficiency_expert = Task(
        description=(
            f"Analyze the following customer support interaction:\n{interaction_context}\n\n"
            "As an Efficiency Expert Supervisor, provide an initial assessment of the chatbot's "
            "response. Focus on operational efficiency, resource utilization, "
            "and adherence to streamlined processes. Identify areas for quicker, "
            "more cost-effective resolutions. "
            "Your output should be a detailed paragraph on your assessment."
        ),
        agent=efficiency_expert,
        expected_output="A detailed paragraph assessing the chatbot response from an efficiency expert's viewpoint."
    )

    analyze_satisfaction_specialist = Task(
        description=(
            f"Analyze the following customer support interaction:\n{interaction_context}\n\n"
            "As a Customer Satisfaction Specialist Supervisor, provide an initial assessment "
            "of the chatbot's response. Focus on its impact on customer sentiment, empathy, "
            "tone, and potential for long-term customer loyalty. Suggest improvements for positive customer experience. "
            "Your output should be a detailed paragraph on your assessment."
        ),
        agent=customer_satisfaction_specialist,
        expected_output="A detailed paragraph assessing the chatbot response from a customer satisfaction specialist's viewpoint."
    )

    debate_resolution_quality = Task(
        description=(
            "Given the initial assessments:\n"
            f"- Frustrated Customer: {{ {analyze_frustrated.output_key} }}\n"
            f"- Detail-Oriented Customer: {{ {analyze_detailed.output_key} }}\n"
            f"- Impatient Customer: {{ {analyze_impatient.output_key} }}\n"
            f"- Efficiency Expert: {{ {analyze_efficiency_expert.output_key} }}\n"
            f"- Customer Satisfaction Specialist: {{ {analyze_satisfaction_specialist.output_key} }}\n\n"
            "And the original customer query: {customer_query}\n"
            "And the chatbot's response: {chatbot_response}\n\n"
            "As a collective group of evaluators (Frustrated Customer, Detail-Oriented Customer, Impatient Customer, "
            "Efficiency Expert, Customer Satisfaction Specialist), critically analyze the chatbot's response. "
            "Each of you should contribute to a 'debate summary' by:\n"
            "1. Stating your persona/role and briefly reiterating your main initial point.\n"
            "2. Responding to points made by other personas/roles that you agree or disagree with, explaining why.\n"
            "3. Highlighting specific strengths and weaknesses of the chatbot's response from your unique perspective.\n"
            "4. Proposing counter-arguments or alternative solutions to improve the resolution.\n"
            "5. Conclude with a collective view on the most significant issues or successes.\n\n"
            "Your final output should be a structured summary of this debate, clearly attributing points to each persona/role, "
            "and demonstrating how different perspectives contribute to a comprehensive understanding of the chatbot's performance."
        ),
        agent=[
            frustrated_customer,
            detail_oriented_customer,
            impatient_customer,
            efficiency_expert,
            customer_satisfaction_specialist
        ],
        context=[
            analyze_frustrated,
            analyze_detailed,
            analyze_impatient,
            analyze_efficiency_expert,
            analyze_satisfaction_specialist
        ],
        expected_output="A comprehensive summary of the multi-agent debate, including arguments for and against the chatbot's performance, highlighting key points of agreement and disagreement among the evaluators, structured by persona."
    )

    synthesize_evaluation_report = Task(
        description=(
            "Based on the customer query: {customer_query}, the chatbot response: {chatbot_response}, "
            f"and the comprehensive debate among various evaluators:\n{{ {debate_resolution_quality.output_key} }}\n\n"
            "As the Evaluation Manager, synthesize a final, detailed evaluation report. "
            "This report should clearly state the overall assessment of the chatbot's resolution, "
            "its strengths, weaknesses, and specific, actionable recommendations for "
            "improving the chatbot's performance and customer support interactions. "
            "Ensure the report is structured and easy to understand."
        ),
        agent=evaluator_manager,
        context=[debate_resolution_quality],
        expected_output="A structured, detailed evaluation report (text/JSON) including overall assessment, strengths, weaknesses, and actionable recommendations for chatbot improvement."
    )

    crew = Crew(
        agents=[
            evaluator_manager,
            frustrated_customer,
            detail_oriented_customer,
            impatient_customer,
            efficiency_expert,
            customer_satisfaction_specialist
        ],
        tasks=[
            analyze_frustrated,
            analyze_detailed,
            analyze_impatient,
            analyze_efficiency_expert,
            analyze_satisfaction_specialist,
            debate_resolution_quality,
            synthesize_evaluation_report
        ],
        process=Process.sequential,
        verbose=True
    )

    customer_query_example = "My order #12345 hasn't arrived after two weeks! The tracking says delivered, but I never received it. This is unacceptable, I need my item or a refund NOW."
    chatbot_response_example = "Thank you for contacting us regarding order #12345. We understand your concern. We show that the package was delivered to your address on [Date]. Please confirm your shipping address. If confirmed, we can initiate a lost package investigation which may take 5-7 business days."

    inputs = {
        "customer_query": customer_query_example,
        "chatbot_response": chatbot_response_example
    }

    print("## Starting Customer Support Resolution Evaluation Crew ##")
    result = crew.kickoff(inputs=inputs)
    print("\n\n## Evaluation Report ##")
    print(result)

if __name__ == "__main__":
    main()