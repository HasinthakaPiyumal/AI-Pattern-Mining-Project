import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class LLMService:
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(openai_api_key=api_key, model_name=model_name)

    def zero_shot_prompt(self, query: str) -> str:
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are a helpful customer support assistant."),
            HumanMessagePromptTemplate.from_template("{query}")
        ])
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        response = chain.run(query=query)
        return response

    def few_shot_prompt(self, query: str, examples: list) -> str:
        example_str = "\n".join([f"Customer: {ex['input']}\nAssistant: {ex['output']}" for ex in examples])
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(f"You are a helpful customer support assistant. Here are some examples of good interactions:\n{example_str}\nNow, respond to the following customer query:"),
            HumanMessagePromptTemplate.from_template("{query}")
        ])
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        response = chain.run(query=query)
        return response

    def role_based_prompt(self, query: str, role: str) -> str:
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(f"You are a customer support assistant with the role of a {role}. Provide a helpful and relevant response."),
            HumanMessagePromptTemplate.from_template("{query}")
        ])
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        response = chain.run(query=query)
        return response

    def prompt_chain_for_troubleshooting(self, issue_description: str) -> str:
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an expert customer support agent specializing in technical troubleshooting. "
                "Analyze the customer's issue step-by-step. First, identify the product or service mentioned. "
                "Second, based on the identified product/service, provide a concise list of 2-3 initial troubleshooting steps. "
                "If the product/service is unclear, ask for clarification. "
                "Respond clearly and helpfully."
            ),
            HumanMessagePromptTemplate.from_template("Customer issue: {issue}")
        ])
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        response = chain.run(issue=issue_description)
        return response

class ValidationService:
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.llm_evaluator = ChatOpenAI(openai_api_key=api_key, model_name=model_name)

    class EvaluationResult(BaseModel):
        score: int = Field(..., description="A score from 1 to 5, where 5 is excellent and 1 is very poor.")
        feedback: str = Field(..., description="Detailed feedback on the response quality, relevance, and helpfulness.")

    def evaluate_response_quality(self, query: str, llm_response: str) -> EvaluationResult:
        parser = PydanticOutputParser(pydantic_object=self.EvaluationResult)
        
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an impartial AI evaluator. Assess the quality of the AI assistant's response to the customer query. "
                "Provide a score from 1 to 5 (5 being excellent) for quality, relevance, and helpfulness. "
                "Also provide detailed feedback. {format_instructions}"
            ),
            HumanMessagePromptTemplate.from_template("Customer Query: {query}\nAI Assistant Response: {response}")
        ])
        
        prompt_with_parser = prompt_template.partial(format_instructions=parser.get_format_instructions())
        chain = LLMChain(llm=self.llm_evaluator, prompt=prompt_with_parser)
        
        output = chain.run(query=query, response=llm_response)
        return parser.parse(output)

    def round_trip_consistency_check(self, original_query: str, ai_solution: str) -> bool:
        summarize_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("Summarize the following solution concisely in one sentence:"),
            HumanMessagePromptTemplate.from_template("{solution}")
        ])
        summarize_chain = LLMChain(llm=self.llm_evaluator, prompt=summarize_prompt)
        summarized_solution = summarize_chain.run(solution=ai_solution)

        check_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("Does the following summarized solution address the original customer query? Respond with 'Yes' or 'No' and a brief reason."),
            HumanMessagePromptTemplate.from_template("Original Query: {query}\nSummarized Solution: {summarized_solution}")
        ])
        check_chain = LLMChain(llm=self.llm_evaluator, prompt=check_prompt)
        check_result = check_chain.run(query=original_query, summarized_solution=summarized_solution)

        return "yes" in check_result.lower()

class EthicalGuidelinesService:
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(openai_api_key=api_key, model_name=model_name)
        self.constitutional_principles = [
            "Always be helpful and harmless.",
            "Avoid generating content that is hateful, abusive, or promotes discrimination.",
            "Do not spread misinformation or provide harmful advice.",
            "Maintain privacy and security of user data.",
            "Be transparent about being an AI assistant."
        ]
        self.system_prompt_prefix = "You are a helpful and ethical AI customer support assistant. Adhere to the following principles:\n" + "\n".join([f"- {p}" for p in self.constitutional_principles]) + "\n"

    def apply_ethical_guidelines(self, query: str) -> str:
        return self.system_prompt_prefix

    def generate_ethically(self, query: str) -> str:
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.apply_ethical_guidelines(query)),
            HumanMessagePromptTemplate.from_template("{query}")
        ])
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        response = chain.run(query=query)
        return response

def run_customer_support_simulation():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    llm_service = LLMService(api_key=openai_api_key)
    validation_service = ValidationService(api_key=openai_api_key)
    ethical_service = EthicalGuidelinesService(api_key=openai_api_key)

    print("--- Customer Support AI Simulation ---")

    print("\n[Scenario 1: Zero-shot Prompting]")
    customer_query1 = "Where is my order #12345?"
    print(f"Customer: {customer_query1}")
    response1 = llm_service.zero_shot_prompt(customer_query1)
    print(f"AI Assistant (Zero-shot): {response1}")
    evaluation1 = validation_service.evaluate_response_quality(customer_query1, response1)
    print(f"Evaluation Score: {evaluation1.score}, Feedback: {evaluation1.feedback}")
    print(f"Consistency Check: {validation_service.round_trip_consistency_check(customer_query1, response1)}")

    print("\n[Scenario 2: Few-shot Prompting]")
    few_shot_examples = [
        {"input": "How do I return a product?", "output": "To return a product, please visit our 'Returns' page and follow the instructions to initiate a return. You'll need your order number."},
        {"input": "What are your shipping options?", "output": "We offer standard and expedited shipping. Details and costs can be found on our 'Shipping Information' page."}
    ]
    customer_query2 = "My item arrived damaged, what should I do?"
    print(f"Customer: {customer_query2}")
    response2 = llm_service.few_shot_prompt(customer_query2, few_shot_examples)
    print(f"AI Assistant (Few-shot): {response2}")
    evaluation2 = validation_service.evaluate_response_quality(customer_query2, response2)
    print(f"Evaluation Score: {evaluation2.score}, Feedback: {evaluation2.feedback}")
    print(f"Consistency Check: {validation_service.round_trip_consistency_check(customer_query2, response2)}")

    print("\n[Scenario 3: Role-based Prompting (Empathetic Agent)]")
    customer_query3 = "I'm very upset, my package is late for my daughter's birthday!"
    print(f"Customer: {customer_query3}")
    response3 = llm_service.role_based_prompt(customer_query3, "highly empathetic and understanding agent")
    print(f"AI Assistant (Empathetic): {response3}")
    evaluation3 = validation_service.evaluate_response_quality(customer_query3, response3)
    print(f"Evaluation Score: {evaluation3.score}, Feedback: {evaluation3.feedback}")
    print(f"Consistency Check: {validation_service.round_trip_consistency_check(customer_query3, response3)}")

    print("\n[Scenario 4: Prompt Chaining for Troubleshooting]")
    customer_query4 = "My new wireless headphones won't connect to my phone. I have an iPhone 13."
    print(f"Customer: {customer_query4}")
    response4 = llm_service.prompt_chain_for_troubleshooting(customer_query4)
    print(f"AI Assistant (Troubleshooting Chain): {response4}")
    evaluation4 = validation_service.evaluate_response_quality(customer_query4, response4)
    print(f"Evaluation Score: {evaluation4.score}, Feedback: {evaluation4.feedback}")
    print(f"Consistency Check: {validation_service.round_trip_consistency_check(customer_query4, response4)}")

    print("\n[Scenario 5: Ethically Aligned Response]")
    customer_query5 = "Can you tell me how to bypass the return policy for a used item?"
    print(f"Customer: {customer_query5}")
    response5 = ethical_service.generate_ethically(customer_query5)
    print(f"AI Assistant (Ethical): {response5}")
    evaluation5 = validation_service.evaluate_response_quality(customer_query5, response5)
    print(f"Evaluation Score: {evaluation5.score}, Feedback: {evaluation5.feedback}")
    print(f"Consistency Check: {validation_service.round_trip_consistency_check(customer_query5, response5)}")

if __name__ == "__main__":
    run_customer_support_simulation()