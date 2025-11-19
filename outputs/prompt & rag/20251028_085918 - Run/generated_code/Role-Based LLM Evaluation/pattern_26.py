import os
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Ensure OPENAI_API_KEY is set in your environment variables
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

class PersonaEvaluation(BaseModel):
    persona: str = Field(description="The name of the persona that performed the evaluation.")
    score: int = Field(description="A numerical score from 1 to 5 (1 = very poor, 5 = excellent) given by the persona.")
    rationale: str = Field(description="A detailed explanation from the persona justifying their score and evaluation.")

class ConsolidatedEvaluation(BaseModel):
    overall_score: float = Field(description="The average or synthesized overall score across all personas.")
    summary: str = Field(description="A consolidated summary of all persona evaluations, highlighting key strengths and weaknesses.")
    actionable_feedback: List[str] = Field(description="A list of actionable feedback points for improving the chatbot's responses.")

class PrimaryChatbotSimulator:
    def __init__(self):
        pass

    def simulate_response(self, query: str) -> str:
        # A simple simulator for an e-commerce customer support chatbot
        query_lower = query.lower()
        if "order status" in query_lower or "where is my order" in query_lower:
            return "Your order #12345 is currently being processed and is expected to ship within 2 business days. You will receive a tracking number via email soon."
        elif "return policy" in query_lower or "how to return" in query_lower:
            return "Our return policy allows returns within 30 days of purchase for a full refund, provided the item is unused and in its original packaging. Please visit our returns portal on the website to initiate a return."
        elif "product details" in query_lower or "specs" in query_lower:
            return "Could you please specify which product you are interested in? I can provide more detailed information once I know the product name or ID."
        elif "payment issue" in query_lower or "card declined" in query_lower:
            return "We apologize for any inconvenience. Please double-check your payment details or try another payment method. If the issue persists, contact your bank or try again later. For security reasons, we cannot see specific decline reasons."
        else:
            return "Thank you for contacting us! How can I assist you further?"

class PersonaAgent:
    def __init__(self, persona_name: str, persona_description: str, llm: ChatOpenAI):
        self.persona_name = persona_name
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"You are {persona_description}. Your task is to evaluate a customer support chatbot's response to a customer query. Provide a score from 1 (very poor) to 5 (excellent) and a detailed rationale from your specific persona's perspective. Your output must be in JSON format, strictly adhering to the PersonaEvaluation Pydantic model structure."),
            ("human", "Customer Query: {query}\nChatbot Response: {chatbot_response}")
        ])
        self.evaluation_chain = LLMChain(prompt=self.prompt_template, llm=self.llm)

    def evaluate(self, query: str, chatbot_response: str) -> PersonaEvaluation:
        response = self.evaluation_chain.invoke({"query": query, "chatbot_response": chatbot_response})
        # Assuming the LLM returns a string that can be parsed as JSON
        import json
        try:
            eval_dict = json.loads(response["text"])
            return PersonaEvaluation(**eval_dict)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for {self.persona_name}: {e}")
            print(f"Raw LLM output: {response['text']}")
            return PersonaEvaluation(persona=self.persona_name, score=1, rationale=f"JSON parsing error: {e}. Raw output: {response['text']}")

class EvaluationOrchestrationModule:
    def __init__(self, llm: ChatOpenAI):
        self.chatbot_simulator = PrimaryChatbotSimulator()
        self.llm = llm
        self.persona_agents = [
            PersonaAgent("The Empathetic Customer", "an empathetic customer who cares about tone, understanding of emotion, and helpfulness, and wants to feel heard and understood", self.llm),
            PersonaAgent("The Technical Support Expert", "a technical support expert who focuses on the accuracy, technical correctness, and completeness of solutions provided", self.llm),
            PersonaAgent("The Sales & Marketing Analyst", "a sales and marketing analyst who assesses upselling/cross-selling opportunities, brand alignment, and customer retention potential of the response", self.llm),
            PersonaAgent("The Policy & Compliance Officer", "a policy and compliance officer who checks adherence to company policies, legal guidelines, and data privacy regulations in the response", self.llm)
        ]
        self.consolidation_chain = self._setup_consolidation_chain()

    def _setup_consolidation_chain(self) -> LLMChain:
        consolidation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI tasked with consolidating evaluations from multiple personas regarding a customer support chatbot's response. Synthesize their feedback into an overall score, a summary, and actionable feedback points. Your output must be in JSON format, strictly adhering to the ConsolidatedEvaluation Pydantic model structure."),
            ("human", "Original Customer Query: {original_query}\nChatbot Response: {chatbot_response}\nIndividual Persona Evaluations: {persona_evaluations}")
        ])
        return LLMChain(prompt=consolidation_prompt, llm=self.llm)

    def run_evaluation(self, customer_query: str) -> ConsolidatedEvaluation:
        print(f"\n--- Evaluating Query: '{customer_query}' ---")
        chatbot_response = self.chatbot_simulator.simulate_response(customer_query)
        print(f"Chatbot Response: '{chatbot_response}'")

        individual_evaluations: List[PersonaEvaluation] = []
        for agent in self.persona_agents:
            print(f"\n--- {agent.persona_name} is evaluating ---")
            evaluation = agent.evaluate(customer_query, chatbot_response)
            individual_evaluations.append(evaluation)
            print(f"  Score: {evaluation.score}, Rationale: {evaluation.rationale}")

        print("\n--- Consolidating Evaluations ---")
        # Prepare individual evaluations for the consolidation prompt
        evals_for_consolidation = [eval.model_dump_json() for eval in individual_evaluations]

        consolidation_response = self.consolidation_chain.invoke({
            "original_query": customer_query,
            "chatbot_response": chatbot_response,
            "persona_evaluations": "; ".join(evals_for_consolidation)
        })

        import json
        try:
            consolidated_dict = json.loads(consolidation_response["text"])
            return ConsolidatedEvaluation(**consolidated_dict)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for consolidation: {e}")
            print(f"Raw LLM output: {consolidation_response['text']}")
            return ConsolidatedEvaluation(overall_score=0.0, summary=f"JSON parsing error: {e}. Raw output: {consolidation_response['text']}", actionable_feedback=[])

if __name__ == "__main__":
    # Initialize LLM - ensure OPENAI_API_KEY is set in your environment
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    except Exception as e:
        print(f"Error initializing ChatOpenAI: {e}")
        print("Please ensure 'OPENAI_API_KEY' is set as an environment variable and you have access to the 'gpt-4o' model.")
        exit()

    orchestrator = EvaluationOrchestrationModule(llm=llm)

    test_queries = [
        "Where is my order #12345?",
        "How do I return a faulty product I bought last week?",
        "Tell me more about the new 'Echo Smart Speaker'",
        "I can't complete my purchase, my credit card keeps getting declined."
    ]

    for query in test_queries:
        final_assessment = orchestrator.run_evaluation(query)
        print("\n--- Final Consolidated Assessment ---")
        print(f"Overall Score: {final_assessment.overall_score}")
        print(f"Summary: {final_assessment.summary}")
        print("Actionable Feedback:")
        for feedback in final_assessment.actionable_feedback:
            print(f"  - {feedback}")
        print("\n" + "="*80 + "\n")
