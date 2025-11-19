from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from guardrails.hub import Toxicity
from guardrails import Guard
import os

# Environment variables for API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")

# Initialize FastAPI app
app = FastAPI(title="Globalized & Fair Customer Support AI Assistant")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

# Guardrails setup for output safety
# guard = Guard().use(Toxicity(threshold=0.5, on_fail="fix"))
# For simplicity and to avoid external API calls if Toxicity is not configured,
# we will use a placeholder for guardrails validation, or directly apply output parser.
# A full implementation would involve guard.validate() around the LLM output.

output_parser = StrOutputParser()

# --- Prompt Engineering Patterns ---

# 1. Cultural Awareness Pattern
cultural_awareness_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant for a multinational e-commerce company. Adapt your language, tone, and recommendations to the cultural context provided. Be polite, empathetic, and always aim to resolve customer issues effectively."),
    ("human", "Culture: {culture_context}\nCustomer Query: {query}")
])

cultural_awareness_chain = cultural_awareness_template | llm | output_parser

# 2. Demonstration Ensembling (DENSE) Pattern - Simplified with two variations
# In a real scenario, exemplars would be dynamically selected and aggregated.

dense_prompt_variation_1 = ChatPromptTemplate.from_messages([
    ("system", "Based on the customer's query, provide a concise and direct answer. Example: Query: 'Where is my order?', Answer: 'Please provide your order number to track your shipment.'"),
    ("human", "Customer Query: {query}")
])

dense_prompt_variation_2 = ChatPromptTemplate.from_messages([
    ("system", "Respond to the customer's query with a slightly more detailed explanation and offer next steps. Example: Query: 'I want to return an item.', Answer: 'You can initiate a return from your order history. Please ensure the item is in its original condition. Would you like me to guide you through the process?'"),
    ("human", "Customer Query: {query}")
])

dense_chain_1 = dense_prompt_variation_1 | llm | output_parser
dense_chain_2 = dense_prompt_variation_2 | llm | output_parser

# A simple aggregation function for DENSE
def aggregate_dense_responses(responses):
    return f"Option 1: {responses['resp1']}\nOption 2: {responses['resp2']}\n\nConsider both options for a comprehensive answer."

# 3. Debate-Style Evidence Aggregation Pattern
debate_pro_template = ChatPromptTemplate.from_messages([
    ("system", "Present arguments and evidence *supporting* the following claim or action related to the customer query. Focus on benefits and positive aspects."),
    ("human", "Customer Query: {query}")
])

debate_con_template = ChatPromptTemplate.from_messages([
    ("system", "Present arguments and potential drawbacks or counter-evidence *against* the following claim or action related to the customer query. Focus on risks and negative aspects."),
    ("human", "Customer Query: {query}")
])

debate_chain_pro = debate_pro_template | llm | output_parser
debate_chain_con = debate_con_template | llm | output_parser

# Aggregate pro and con arguments
def aggregate_debate_responses(responses):
    return f"Arguments For:\n{responses['pro_args']}\n\nArguments Against:\n{responses['con_args']}\n\nConsider all perspectives for a balanced view."

# --- Main Orchestration Chain ---

# This chain orchestrates the different patterns based on the query type or flags.
# For simplicity, we'll run them sequentially or in parallel for demonstration.
# In a real app, conditional logic would determine which patterns to apply.

class CustomerQuery(BaseModel):
    query: str
    culture_context: str = "General (English)"
    enable_dense: bool = False
    enable_debate: bool = False

@app.post("/query")
async def handle_customer_query(input: CustomerQuery):
    try:
        # Initial cultural awareness response
        cultural_response = cultural_awareness_chain.invoke({"query": input.query, "culture_context": input.culture_context})

        final_response_parts = [f"Culturally adapted initial response: {cultural_response}"]

        # DENSE pattern integration
        if input.enable_dense:
            dense_responses = await RunnableParallel({
                "resp1": dense_chain_1,
                "resp2": dense_chain_2
            }).invoke({"query": input.query})
            aggregated_dense = aggregate_dense_responses(dense_responses)
            final_response_parts.append(f"\n\nDemonstration Ensembled (DENSE) options:\n{aggregated_dense}")

        # Debate-Style Evidence Aggregation integration
        if input.enable_debate:
            debate_responses = await RunnableParallel({
                "pro_args": debate_chain_pro,
                "con_args": debate_chain_con
            }).invoke({"query": input.query})
            aggregated_debate = aggregate_debate_responses(debate_responses)
            final_response_parts.append(f"\n\nDebate-Style Evidence Aggregation:\n{aggregated_debate}")

        full_response = " ".join(final_response_parts)

        # Placeholder for Bias-Aware Design & Mitigation / Guardrails validation
        # In a full implementation, the 'guard' object would validate 'full_response'
        # e.g., validated_response = guard.validate(full_response)
        # For now, we assume the LLM output is generally safe.

        return {"response": full_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "AI Assistant is running"}

# To run the app:
# 1. Save this code as `customer_support_ai.py`
# 2. Install dependencies: `pip install fastapi "uvicorn[standard]" langchain_openai`
#    For guardrails, `pip install guardrails-ai` (and potentially `openai` for `Toxicity`)
# 3. Set your OpenAI API key in environment variables or directly in the code (not recommended for production)
# 4. Run from your terminal: `uvicorn customer_support_ai:app --reload`
# 5. Access at http://127.0.0.1:8000/docs for API documentation and testing.

# Example usage with requests (if running outside Uvicorn):
# import requests

# if __name__ == "__main__":
#     # This block is just for demonstrating how to call the API
#     # In a real setup, uvicorn would run the app separately.

#     # Example 1: Basic query with cultural awareness
#     payload1 = {"query": "I need help with my recent purchase.", "culture_context": "Japanese - formal and polite"}
#     response1 = requests.post("http://127.0.0.1:8000/query", json=payload1)
#     print("\n--- Response 1 (Cultural Awareness) ---")
#     print(response1.json())

#     # Example 2: Query with DENSE enabled
#     payload2 = {"query": "My item hasn't arrived yet.", "culture_context": "American - direct and efficient", "enable_dense": True}
#     response2 = requests.post("http://127.0.0.1:8000/query", json=payload2)
#     print("\n--- Response 2 (DENSE) ---")
#     print(response2.json())

#     # Example 3: Query with Debate enabled
#     payload3 = {"query": "Should I cancel my subscription?", "culture_context": "British - thoughtful and balanced", "enable_debate": True}
#     response3 = requests.post("http://127.0.0.1:8000/query", json=payload3)
#     print("\n--- Response 3 (Debate) ---")
#     print(response3.json())

#     # Example 4: All patterns enabled
#     payload4 = {"query": "Tell me about the refund policy.", "culture_context": "German - precise and factual", "enable_dense": True, "enable_debate": True}
#     response4 = requests.post("http://127.0.0.1:8000/query", json=payload4)
#     print("\n--- Response 4 (All Enabled) ---")
#     print(response4.json())

