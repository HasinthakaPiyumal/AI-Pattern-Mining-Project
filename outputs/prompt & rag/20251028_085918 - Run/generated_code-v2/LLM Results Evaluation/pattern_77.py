import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

load_dotenv()

class ProductDescriptionABTest:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o") # Using a capable model for better results

    def generate_product_descriptions(self, product_name: str, features: str, target_audience: str) -> tuple[str, str]:
        # Prompt Template for Description A (Feature-focused, formal)
        prompt_a_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert copywriter specializing in clear, detailed, and SEO-friendly product descriptions for e-commerce. Your goal is to highlight features and benefits."),
                ("human", "Generate a concise and informative product description for '{product_name}'. Focus on the following features: {features}. The target audience is: {target_audience}.")
            ]
        )
        chain_a = LLMChain(llm=self.llm, prompt=prompt_a_template)

        # Prompt Template for Description B (Benefit-focused, engaging)
        prompt_b_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a creative copywriter skilled in crafting engaging and persuasive product descriptions that resonate with the target audience. Focus on emotional appeal and user experience."),
                ("human", "Write an enticing and persuasive product description for '{product_name}'. Emphasize how the product solves problems or enhances the user's life, considering these features: {features}. The target audience is: {target_audience}.")
            ]
        )
        chain_b = LLMChain(llm=self.llm, prompt=prompt_b_template)

        description_a_result = chain_a.run(product_name=product_name, features=features, target_audience=target_audience)
        description_b_result = chain_b.run(product_name=product_name, features=features, target_audience=target_audience)

        return description_a_result, description_b_result

    def evaluate_descriptions(self, product_name: str, description_a: str, description_b: str) -> dict:
        evaluation_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", 
                 "You are an impartial and expert evaluator of e-commerce product descriptions. Your task is to compare two descriptions (A and B) for the same product based on clarity, persuasiveness, SEO-friendliness, and completeness. "
                 "Provide a detailed rationale for your choice and clearly state which description is superior. Output your response in a JSON format with keys 'preferred_description' (A or B), 'rationale', and 'clarity_score', 'persuasiveness_score', 'seo_friendliness_score', 'completeness_score' for each description (out of 10)."),
                ("human", 
                 "Product: {product_name}\n\nDescription A:\n{description_a}\n\nDescription B:\n{description_b}\n\nWhich description is superior based on the given criteria? Provide scores and a rationale.")
            ]
        )
        evaluation_chain = LLMChain(llm=self.llm, prompt=evaluation_prompt_template)

        evaluation_result = evaluation_chain.run(
            product_name=product_name,
            description_a=description_a,
            description_b=description_b
        )
        
        # Attempt to parse the JSON output from the LLM
        try:
            import json
            parsed_result = json.loads(evaluation_result)
            return parsed_result
        except json.JSONDecodeError:
            print("Warning: LLM did not return valid JSON. Returning raw string.")
            return {"error": "Invalid JSON from LLM", "raw_output": evaluation_result}

def main():
    ab_tester = ProductDescriptionABTest()

    print("--- E-commerce Product Description A/B Testing Platform ---")
    product_name = input("Enter product name: ")
    features = input("Enter key features (e.g., 'waterproof, 10-hour battery life'): ")
    target_audience = input("Enter target audience (e.g., 'young professionals, outdoor enthusiasts'): ")

    print("\nGenerating two product descriptions...")
    desc_a, desc_b = ab_tester.generate_product_descriptions(product_name, features, target_audience)

    print("\n--- Generated Description A ---")
    print(desc_a)
    print("\n--- Generated Description B ---")
    print(desc_b)

    print("\nEvaluating descriptions...")
    evaluation = ab_tester.evaluate_descriptions(product_name, desc_a, desc_b)

    print("\n--- Evaluation Result ---")
    if "error" in evaluation:
        print(evaluation["raw_output"])
    else:
        print(f"Preferred Description: {evaluation.get('preferred_description', 'N/A')}")
        print(f"Rationale: {evaluation.get('rationale', 'No rationale provided.')}")
        print("\nScores (out of 10) for Description A:")
        print(f"  Clarity: {evaluation.get('clarity_score_A', 'N/A')}")
        print(f"  Persuasiveness: {evaluation.get('persuasiveness_score_A', 'N/A')}")
        print(f"  SEO-friendliness: {evaluation.get('seo_friendliness_score_A', 'N/A')}")
        print(f"  Completeness: {evaluation.get('completeness_score_A', 'N/A')}")
        print("\nScores (out of 10) for Description B:")
        print(f"  Clarity: {evaluation.get('clarity_score_B', 'N/A')}")
        print(f"  Persuasiveness: {evaluation.get('persuasiveness_score_B', 'N/A')}")
        print(f"  SEO-friendliness: {evaluation.get('seo_friendliness_score_B', 'N/A')}")
        print(f"  Completeness: {evaluation.get('completeness_score_B', 'N/A')}")

if __name__ == "__main__":
    main()