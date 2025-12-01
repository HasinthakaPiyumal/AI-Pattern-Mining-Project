from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from langchain_core.prompts import PromptTemplate

app = FastAPI()

class InContextExample(BaseModel):
    source_query: str
    source_language: str
    target_response: str
    target_language: str

class ExampleManager:
    def __init__(self):
        self.examples: List[InContextExample] = [
            InContextExample(source_query="¿Dónde está mi pedido?", source_language="es", target_response="Your order is on its way and should arrive within 2 business days.", target_language="en"),
            InContextExample(source_query="My package hasn't arrived.", source_language="en", target_response="Ihr Paket ist auf dem Weg und sollte innerhalb von 2 Werktagen ankommen.", target_language="de"),
            InContextExample(source_query="J'ai un problème avec ma facture.", source_language="fr", target_response="We apologize for the inconvenience. Please provide your order number so we can investigate.", target_language="en"),
            InContextExample(source_query="I need to return an item.", source_language="en", target_response="Um einen Artikel zurückzusenden, besuchen Sie bitte unsere Rückgabeseite und folgen Sie den Anweisungen.", target_language="de")
        ]

    def get_examples(self, count: int = 2) -> List[InContextExample]:
        return self.examples[:count] # For simplicity, just return the first 'count' examples

class InCLTPromptGenerator:
    def __init__(self, example_manager: ExampleManager):
        self.example_manager = example_manager
        self.prompt_template = PromptTemplate.from_template(
            """You are a multilingual customer support agent. Generate a response in {target_language} based on the customer's query.

Here are some examples of customer queries and appropriate responses, showing cross-lingual transfer:
{in_context_examples}

Customer query in {source_language}: {customer_query}
Agent response in {target_language}:"""
        )

    def generate_prompt(self, customer_query: str, source_language: str, target_language: str) -> str:
        examples = self.example_manager.get_examples()
        formatted_examples = []
        for ex in examples:
            formatted_examples.append(f"Query in {ex.source_language}: {ex.source_query}\nResponse in {ex.target_language}: {ex.target_response}")
        
        in_context_examples_str = "\n".join(formatted_examples)

        return self.prompt_template.format(
            customer_query=customer_query,
            source_language=source_language,
            target_language=target_language,
            in_context_examples=in_context_examples_str
        )

class MultilingualLLM:
    def __init__(self):
        pass # In a real scenario, initialize a transformers model here

    def generate_response(self, prompt: str) -> str:
        # This is a simulated LLM response for demonstration
        if "¿Dónde está mi pedido?" in prompt and "en" in prompt:
            return "Your order is being processed and will be shipped soon."
        elif "My package hasn't arrived." in prompt and "de" in prompt:
            return "Ihr Paket wurde versandt und sollte in Kürze ankommen."
        elif "J'ai un problème avec ma facture." in prompt and "en" in prompt:
            return "Could you please provide your invoice number? We will assist you promptly."
        else:
            return "This is a placeholder response based on the generated prompt. Actual LLM would generate a more nuanced answer."

example_manager = ExampleManager()
inclt_prompt_generator = InCLTPromptGenerator(example_manager)
llm_model = MultilingualLLM()

class AgentAssistRequest(BaseModel):
    customer_query: str
    source_language: str
    target_language: str

@app.post("/agent-assist")
async def agent_assist(request: AgentAssistRequest):
    prompt = inclt_prompt_generator.generate_prompt(
        customer_query=request.customer_query,
        source_language=request.source_language,
        target_language=request.target_language
    )
    response = llm_model.generate_response(prompt)
    return {"generated_response": response, "debug_prompt": prompt}
