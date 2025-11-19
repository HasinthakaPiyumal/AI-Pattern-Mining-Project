import os
from typing import List, Dict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

# 1. Persona Class
class Persona(BaseModel):
    name: str = Field(description="The name of the persona")
    description: str = Field(description="A detailed description of the persona's role and perspective")
    criteria: str = Field(description="Specific evaluation criteria this persona focuses on")

# Pydantic model for individual persona evaluation output
class PersonaEvaluationOutput(BaseModel):
    persona_name: str = Field(description="The name of the persona who performed this evaluation")
    score: int = Field(description="A score from 1 to 10 for the response based on the persona's criteria")
    feedback: str = Field(description="Detailed feedback from the persona on the automated response")

# Pydantic model for synthesized evaluation output
class SynthesizedEvaluationOutput(BaseModel):
    overall_score: int = Field(description="An overall quality score for the response, synthesized from all personas")
    improvement_suggestions: str = Field(description="Actionable suggestions to improve the automated response")

class PersonaEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.personas: List[Persona] = []

    def add_persona(self, persona: Persona):
        self.personas.append(persona)

    def evaluate_response(self, customer_query: str, automated_response: str) -> List[PersonaEvaluationOutput]:
        evaluations: List[PersonaEvaluationOutput] = []
        parser = PydanticOutputParser(pydantic_object=PersonaEvaluationOutput)

        for persona in self.personas:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", f"You are {persona.name}. {persona.description}. Your task is to evaluate a customer support response based on the following criteria: {persona.criteria}.\n{parser.get_format_instructions()}"),
                    ("human", f"Customer Query: {customer_query}\nAutomated Response: {automated_response}\n\nProvide your evaluation as a JSON object.")
                ]
            )
            chain = prompt | self.llm | parser
            result = chain.invoke({"customer_query": customer_query, "automated_response": automated_response})
            result.persona_name = persona.name # Assign persona name to the output object
            evaluations.append(result)
        return evaluations

class EvaluationSynthesizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def synthesize_feedback(self,
                            customer_query: str,
                            automated_response: str,
                            persona_evaluations: List[PersonaEvaluationOutput]
                           ) -> SynthesizedEvaluationOutput:
        
        feedback_summary = "\n".join(
            [f"- {eval.persona_name} (Score: {eval.score}): {eval.feedback}" for eval in persona_evaluations]
        )

        parser = PydanticOutputParser(pydantic_object=SynthesizedEvaluationOutput)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"You are an Evaluation Synthesizer. Your role is to aggregate feedback from various AI personas to provide a comprehensive quality assessment and actionable improvement suggestions for a customer support response.\nBased on the individual persona evaluations, provide an overall score and detailed suggestions for improvement.\n{parser.get_format_instructions()}"),
                ("human", f"Customer Query: {customer_query}\nAutomated Response: {automated_response}\n\nIndividual Persona Evaluations:\n{feedback_summary}\n\nProvide the synthesized evaluation as a JSON object.")
            ]
        )
        chain = prompt | self.llm | parser
        result = chain.invoke({"customer_query": customer_query, "automated_response": automated_response, "feedback_summary": feedback_summary})
        return result

if __name__ == "__main__":
    # Set up OpenAI API key
    # Ensure OPENAI_API_KEY is set in your environment variables
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    # Sample data
    customer_query = "My internet is not working. I've tried restarting my router, but it didn't help. What should I do?"
    automated_response = "We apologize for the inconvenience. Please restart your router and modem. If the issue persists, visit our troubleshooting guide at example.com/troubleshoot or contact technical support."

    # 1. Define Personas
    frustrated_customer = Persona(
        name="Frustrated Customer",
        description="Evaluates the response from the perspective of a customer who is already annoyed and wants a quick, effective solution.",
        criteria="Empathy, directness of solution, avoidance of repetitive steps, speed of resolution."
    )
    technical_expert = Persona(
        name="Technical Expert",
        description="Assesses the technical accuracy and completeness of the solution provided.",
        criteria="Technical accuracy, clarity of instructions, depth of troubleshooting."
    )
    company_policy_enforcer = Persona(
        name="Company Policy Enforcer",
        description="Checks if the response adheres to company guidelines, terms of service, and escalates appropriately.",
        criteria="Adherence to policy, proper escalation paths, legal compliance."
    )
    empathy_analyst = Persona(
        name="Empathy Analyst",
        description="Focuses on the tone, language, and overall empathetic quality of the response.",
        criteria="Compassionate language, active listening (implied), reassurance, customer sentiment improvement."
    )

    # 2. Instantiate PersonaEvaluator and add personas
    evaluator = PersonaEvaluator(llm=llm)
    evaluator.add_persona(frustrated_customer)
    evaluator.add_persona(technical_expert)
    evaluator.add_persona(company_policy_enforcer)
    evaluator.add_persona(empathy_analyst)

    # 3. Evaluate the response
    print("\n--- Running Persona Evaluations ---")
    persona_evals = evaluator.evaluate_response(customer_query, automated_response)

    for pe in persona_evals:
        print(f"\nPersona: {pe.persona_name}")
        print(f"Score: {pe.score}/10")
        print(f"Feedback: {pe.feedback}")

    # 4. Synthesize the evaluations
    print("\n--- Synthesizing Evaluations ---")
    synthesizer = EvaluationSynthesizer(llm=llm)
    final_report = synthesizer.synthesize_feedback(customer_query, automated_response, persona_evals)

    print("\n--- Final Report ---")
    print(f"Overall Quality Score: {final_report.overall_score}/10")
    print(f"Improvement Suggestions: {final_report.improvement_suggestions}")
