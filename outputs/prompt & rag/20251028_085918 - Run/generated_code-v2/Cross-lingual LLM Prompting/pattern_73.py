from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline

app = FastAPI()

product_database = {
    "electronics_en": "High-end noise-cancelling headphones with 20-hour battery life.",
    "electronics_fr": "Casque audio haut de gamme à réduction de bruit avec 20 heures d'autonomie.",
    "apparel_en": "Comfortable, breathable running shoes, perfect for long distances.",
    "apparel_fr": "Chaussures de course confortables et respirantes, parfaites pour les longues distances."
}

cross_lingual_examples = [
    {
        "query_lang": "fr",
        "query_text": "J'ai besoin d'un casque pour voyager. Avez-vous quelque chose avec une bonne autonomie et réduction de bruit ?",
        "product_context_lang": "en",
        "product_context_text": "Product: High-end noise-cancelling headphones with 20-hour battery life.",
        "expected_answer_lang": "fr",
        "expected_answer_text": "Oui, nous avons des casques haut de gamme à réduction de bruit avec une autonomie de 20 heures, parfaits pour les voyages."
    },
    {
        "query_lang": "en",
        "query_text": "I need shoes for running long distances.",
        "product_context_lang": "fr",
        "product_context_text": "Produit : Chaussures de course confortables et respirantes, parfaites pour les longues distances.",
        "expected_answer_lang": "en",
        "expected_answer_text": "Our comfortable and breathable running shoes are perfect for long distances."
    }
]

model_name = "google/mt5-base"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        device=-1
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
except Exception as e:
    class MockLLM:
        def invoke(self, prompt: str) -> dict:
            if "Casque audio" in prompt or "headphones" in prompt:
                return {"text": "Mock Response: Yes, we have noise-cancelling headphones for travel with 20 hours battery life. (Simulated mt5 response)"}
            elif "Chaussures de course" in prompt or "running shoes" in prompt:
                return {"text": "Mock Response: Our comfortable running shoes are ideal for long distances. (Simulated mt5 response)"}
            else:
                return {"text": "Mock Response: I'm sorry, I cannot assist with that specific query. (Simulated mt5 response)"}
    llm = MockLLM()

prompt_template_string = """You are a helpful multilingual customer support assistant.\nPlease answer the user's question based on the provided product information and examples.\n\nExamples for cross-lingual transfer (Query, Product Context, Expected Answer):\n{examples}\n\nProduct Information (may be in a different language than the query):\n{product_info}\n\nUser Query: {query}\nAnswer in the same language as the User Query, being concise and helpful."""

prompt = PromptTemplate(
    input_variables=["examples", "product_info", "query"],
    template=prompt_template_string,
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

def get_relevant_product_info_text(topic: str, target_lang_for_retrieval: str = "en"):
    product_key = f"{topic}_{target_lang_for_retrieval}"
    if product_key in product_database:
        return product_database[product_key]
    
    fallback_key = f"{topic}_en"
    if fallback_key in product_database:
        return product_database[fallback_key]
    
    if "electronics" in topic and "electronics_en" in product_database:
        return product_database["electronics_en"]
    if "apparel" in topic and "apparel_en" in product_database:
        return product_database["apparel_en"]

    return "No specific product information found."


def format_cross_lingual_examples_for_prompt() -> str:
    formatted_examples = []
    for ex in cross_lingual_examples:
        formatted_examples.append(
            f"Query ({ex['query_lang']}): {ex['query_text']}\n"
            f"Product Context ({ex['product_context_lang']}): {ex['product_context_text']}\n"
            f"Expected Answer ({ex['expected_answer_lang']}): {ex['expected_answer_text']}"
        )
    return "\n\n".join(formatted_examples)

class ChatMessage(BaseModel):
    query: str
    query_language: str
    topic: str = "general"

@app.post("/chat")
async def chat_with_bot(message: ChatMessage):
    product_info_text = get_relevant_product_info_text(message.topic)

    formatted_examples = format_cross_lingual_examples_for_prompt()

    full_prompt_input = {
        "examples": formatted_examples,
        "product_info": product_info_text,
        "query": message.query
    }
    
    try:
        response_dict = llm_chain.invoke(full_prompt_input)
        response = response_dict['text']
    except Exception as e:
        response = f"Sorry, I am currently experiencing issues. Please try again later. (Error: {e})"

    return {"response": response}