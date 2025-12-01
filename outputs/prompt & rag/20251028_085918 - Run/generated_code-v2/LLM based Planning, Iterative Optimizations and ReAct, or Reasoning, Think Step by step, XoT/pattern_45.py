from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv()

# Stage 1: Prompt Refiner LLM
refiner_llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o")

refiner_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "Rewrite the following customer query to remove any irrelevant information and extract the core question(s) concisely. Focus only on what the customer needs help with. \n\nCustomer Query: {customer_query}\n\nRefined Query:"
    )
])

refiner_chain = LLMChain(llm=refiner_llm, prompt=refiner_template, output_key="refined_query")

# Stage 2: Response Generation LLM
response_llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o")

response_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "You are a helpful and polite e-commerce customer support assistant. Based on the following refined query, provide a clear and direct answer or solution to the customer's issue. If you need more information, politely ask for it.\n\nRefined Query: {refined_query}\n\nAnswer:"
    )
])

response_chain = LLMChain(llm=response_llm, prompt=response_template, output_key="final_response")

# Orchestration: SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[refiner_chain, response_chain], input_variables=["customer_query"], output_variables=["final_response"], verbose=False)

def get_customer_support_response(query: str) -> str:
    result = overall_chain.invoke({"customer_query": query})
    return result["final_response"]

if __name__ == "__main__":
    # Example Usage
    noisy_query = (
        "Hi there, I bought a new pair of shoes last week, item #12345, and I love them! "
        "But anyway, I actually need to know if it's possible to change the shipping address "
        "for my order #67890 that I placed yesterday, because I'll be out of town next week. "
        "Also, are there any discounts for loyal customers?"
    )

    clean_query = (
        "I'm having trouble with my recent order, I can't log in to my account. "
        "Can you help me reset my password? I tried the 'forgot password' link but it didn't work."
    )

    response1 = get_customer_support_response(noisy_query)
    print(f"\n--- Customer Query 1 ---\n{noisy_query}")
    print(f"\n--- AI Assistant Response 1 ---\n{response1}")

    response2 = get_customer_support_response(clean_query)
    print(f"\n--- Customer Query 2 ---\n{clean_query}")
    print(f"\n--- AI Assistant Response 2 ---\n{response2}")