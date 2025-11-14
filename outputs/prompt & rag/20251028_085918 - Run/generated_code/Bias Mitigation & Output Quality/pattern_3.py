
import os
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Placeholder for actual LLM interaction and aggregation
# In a real scenario, you'd use a robust LLM provider.

class CustomerQuery(BaseModel):
    query: str
    customer_id: str = "anonymous"
    cultural_context: str = "general" # e.g., "US", "India", "Germany"

app = FastAPI(
    title="Intelligent Customer Support Assistant",
    description="An AI assistant leveraging advanced LLM prompting for accuracy, bias mitigation, and cultural sensitivity."
)

class CustomerSupportAssistant:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0.7)
        self.output_parser = StrOutputParser()

        # 1. Few-Shot Demonstrations for general queries (DENSE & Balanced Demonstrations)
        # In a real application, you'd have a much larger and more diverse set of examples.
        # For DENSE, you'd create multiple FewShotPromptTemplates with different subsets.
        # For Balanced, ensure examples cover diverse demographics, sentiments, etc.
        self.general_examples = [
            {"query": "How do I reset my password?", "answer": "To reset your password, navigate to the login page and click 'Forgot Password'. Follow the on-screen instructions."},
            {"query": "What are your return policies?", "answer": "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. Please see our FAQs for full details."},
            {"query": "My order hasn't arrived yet.", "answer": "Please provide your order number. We will track its status for you. Standard delivery usually takes 3-5 business days."},
            {"query": "I want to cancel my subscription.", "answer": "You can cancel your subscription anytime from your account settings under 'Manage Subscription'. Confirm your cancellation, and it will be effective at the end of your current billing cycle."},
            {"query": "Is product X available in blue?", "answer": "Product X is currently available in red, green, and black. We do not have blue in stock at the moment."}
        ]

        # Template for general queries
        self.general_prompt_template = PromptTemplate.from_template(
            """You are a helpful and polite customer support assistant. Answer the user's query accurately and concisely.

            {cultural_instruction}

            Here are some examples of how to answer customer queries:
            {examples}

            Customer Query: {query}
            Assistant Answer:"""
        )

        # 2. Cultural Awareness Component
        # This will be injected into various prompts.
        self.cultural_instructions = {
            "US": "Maintain a friendly and direct tone. Focus on efficiency and clear solutions.",
            "India": "Be polite and use respectful language. Emphasize thoroughness and offer multiple solutions if applicable.",
            "Germany": "Be precise and factual. Focus on accuracy and adherence to policies. Avoid overly informal language.",
            "general": "Be polite, helpful, and clear."
        }

        # 3. Debate-Style Evidence Aggregation for complex/policy queries
        # This is a simplified version, simulating internal "pros and cons" generation.
        self.debate_template = PromptTemplate.from_template(
            """You are an expert analyst tasked with providing a balanced perspective on a complex topic.
            For the following topic, identify arguments FOR and AGAINST, and then provide a synthesized, balanced conclusion.

            Topic: {query}

            Arguments FOR:
            -

            Arguments AGAINST:
            -

            Balanced Conclusion:
            """
        )

    def _get_cultural_instruction(self, context: str) -> str:
        return self.cultural_instructions.get(context, self.cultural_instructions["general"])

    def _generate_response_with_dense(self, query: str, cultural_context: str) -> str:
        # Simulate DENSE by generating responses from multiple 'subsets' and taking a simple aggregation (e.g., first plausible)
        # In a real DENSE implementation, you'd have distinct FewShotPromptTemplates.
        # For simplicity, we'll iterate through a shuffled set of examples and simulate multiple runs.

        responses = []
        for i in range(2): # Simulate two distinct "prompt runs"
            # Simulate selecting a subset of balanced demonstrations
            selected_examples = self.general_examples[i:i+3] if len(self.general_examples) >= i+3 else self.general_examples
            few_shot_prompt = FewShotPromptTemplate(
                examples=selected_examples,
                example_prompt=PromptTemplate(input_variables=["query", "answer"], template="Customer Query: {query}\nAssistant Answer: {answer}"),
                prefix=self.general_prompt_template.format(
                    cultural_instruction=self._get_cultural_instruction(cultural_context),
                    examples="" # examples are handled by FewShotPromptTemplate
                ),
                suffix="Customer Query: {query}\nAssistant Answer:",
                input_variables=["query"]
            )
            chain = few_shot_prompt | self.llm | self.output_parser
            response = chain.invoke({"query": query})
            responses.append(response)

        # Simple aggregation: take the first non-empty response.
        # A more sophisticated DENSE would analyze consistency, confidence scores, etc.
        return responses[0] if responses else "I'm sorry, I couldn't generate a response."

    def _generate_debate_style_response(self, query: str) -> str:
        chain = self.debate_template | self.llm | self.output_parser
        response = chain.invoke({"query": query})
        return response

    async def get_assistant_response(self, query_data: CustomerQuery) -> Dict[str, str]:
        query = query_data.query
        cultural_context = query_data.cultural_context

        # Simple heuristic to decide if it's a "complex/debate-style" query
        if "policy" in query.lower() or "controversy" in query.lower() or "compare" in query.lower() or "pros and cons" in query.lower():
            response = self._generate_debate_style_response(query)
            response_type = "Debate-Style Aggregation"
        else:
            response = self._generate_response_with_dense(query, cultural_context)
            response_type = "DENSE with Balanced Demonstrations & Cultural Awareness"

        return {
            "query": query,
            "response": response,
            "response_type": response_type,
            "cultural_context_applied": cultural_context
        }

assistant = CustomerSupportAssistant(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/chat")
async def chat_with_assistant(query: CustomerQuery):
    response = await assistant.get_assistant_response(query)
    return response
