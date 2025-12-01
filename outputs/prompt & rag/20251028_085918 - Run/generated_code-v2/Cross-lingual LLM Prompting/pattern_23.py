import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Libraries for LLM and embeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

app = FastAPI()

# --- Configuration and Initialization ---

# LLM Setup
# Use OpenAI by default, fall back to a dummy if no key is provided
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM = None
if OPENAI_API_KEY:
    LLM = ChatOpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY, model_name="gpt-4")
else:
    print("Warning: OPENAI_API_KEY not found. LLM responses will be simulated.")

# Embedding Model
EMBEDDING_MODEL = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Dummy Knowledge Base and Examples ---

# In a real application, this would be loaded from a persistent store
dummy_knowledge_base_data = [
    {
        "id": "kb_1_en",
        "lang": "en",
        "text": "Customer: My order #123 hasn't arrived. Agent: Apologies for the delay. We are tracking it and will provide an update within 24 hours.",
        "intent": "package_delay"
    },
    {
        "id": "kb_1_es",
        "lang": "es",
        "text": "Cliente: Mi pedido #123 no ha llegado. Agente: Disculpe la demora. Lo estamos rastreando y le daremos una actualización en 24 horas.",
        "intent": "package_delay"
    },
    {
        "id": "kb_2_en",
        "lang": "en",
        "text": "Customer: How can I return an item? Agent: You can initiate a return within 30 days of purchase through your account's order history.",
        "intent": "return_policy"
    },
    {
        "id": "kb_2_es",
        "lang": "es",
        "text": "Cliente: ¿Cómo puedo devolver un artículo? Agente: Puede iniciar una devolución dentro de los 30 días posteriores a la compra a través del historial de pedidos de su cuenta.",
        "intent": "return_policy"
    }
]

# Initialize ChromaDB
# For simplicity, we'll re-initialize and populate it on startup
# In a real scenario, this would be persistent and loaded from disk.
chroma_client = Chroma(
    collection_name="global_care_kb",
    embedding_function=EMBEDDING_MODEL,
    persist_directory="./chroma_db"
)

# Populate ChromaDB if empty
if chroma_client._collection.count() == 0:
    texts = [item["text"] for item in dummy_knowledge_base_data]
    metadatas = [{
        "id": item["id"],
        "lang": item["lang"],
        "intent": item["intent"]
    } for item in dummy_knowledge_base_data]
    chroma_client.add_texts(texts=texts, metadatas=metadatas)
    chroma_client.persist()
    print("ChromaDB populated with dummy data.")

# --- Helper Functions ---

def detect_language(text: str) -> str:
    # A very basic dummy language detection for demonstration
    # In a real system, use a robust library like 'langdetect' or a cloud service.
    text_lower = text.lower()
    if "paquete" in text_lower or "pedido" in text_lower or "devolver" in text_lower:
        return "es"
    return "en"

def translate_text(text: str, target_language: str, source_language: str = "en") -> str:
    # Placeholder for a real machine translation service (e.g., Google Cloud Translation API)
    print(f"[DUMMY TRANSLATION] Translating '{text}' from {source_language} to {target_language}")
    # In a real application, integrate with a service like:
    # from google.cloud import translate_v2 as translate
    # translate_client = translate.Client()
    # result = translate_client.translate(text, target_language=target_language)
    # return result['translatedText']

    # For this demo, we'll simply indicate it's a translation or return original if no specific translation available
    if target_language == "es" and "order" in text.lower():
        return text.replace("order", "pedido").replace("arrived", "llegado").replace("Apologies for the delay", "Disculpe la demora")
    if target_language == "en" and "pedido" in text.lower():
        return text.replace("pedido", "order").replace("llegado", "arrived").replace("Disculpe la demora", "Apologies for the delay")
    return f"[Translated to {target_language}: {text}]" # Generic placeholder

def retrieve_cross_lingual_examples(
    query: str,
    query_lang: str,
    num_examples: int = 2,
    source_lang: str = "en"
) -> List[Dict[str, str]]:
    
    retrieved_examples = []

    # 1. Try to retrieve examples directly in the query language
    results_in_query_lang = chroma_client.similarity_search_with_score(
        query,
        k=num_examples,
        where={"lang": query_lang}
    )

    for doc, score in results_in_query_lang:
        # Assuming doc.page_content is the full interaction text
        # and doc.metadata contains lang and intent
        retrieved_examples.append({
            "source_lang_problem_solution": doc.page_content if doc.metadata["lang"] == source_lang else translate_text(doc.page_content, source_lang, doc.metadata["lang"]),
            "target_lang_problem_solution": doc.page_content if doc.metadata["lang"] == query_lang else translate_text(doc.page_content, query_lang, doc.metadata["lang"])
        })

    # 2. If not enough, retrieve from the source language and translate
    if len(retrieved_examples) < num_examples:
        needed = num_examples - len(retrieved_examples)
        results_in_source_lang = chroma_client.similarity_search_with_score(
            query,
            k=needed,
            where={"lang": source_lang}
        )

        for doc, score in results_in_source_lang:
            # Ensure we don't duplicate if an English example was already retrieved
            if not any(doc.page_content in ex["source_lang_problem_solution"] for ex in retrieved_examples):
                retrieved_examples.append({
                    "source_lang_problem_solution": doc.page_content,
                    "target_lang_problem_solution": translate_text(doc.page_content, query_lang, source_lang)
                })

    return retrieved_examples


# --- Prompt Engineering for InCLT ---

INCLT_PROMPT_TEMPLATE = """
Your task is to act as a helpful and accurate customer support agent for a multinational e-commerce company.

Here are some examples of customer interactions and agent responses in both English and {target_language} to guide you. These examples demonstrate how to handle various customer queries and respond appropriately.

{in_context_examples}

--- Start of New Customer Interaction ---

Customer query in {target_language}: "{customer_query}"

Agent response in {target_language}:"""

# --- FastAPI Models ---

class CustomerQuery(BaseModel):
    query: str

class AgentResponse(BaseModel):
    response: str
    language: str
    context_used: List[Dict[str, str]]

# --- API Endpoint ---

@app.post("/query", response_model=AgentResponse)
async def handle_customer_query(customer_query: CustomerQuery):
    try:
        query_text = customer_query.query
        detected_lang = detect_language(query_text)
        source_lang = "en" # Our primary knowledge base language

        # Retrieve cross-lingual in-context examples
        in_context_examples_list = retrieve_cross_lingual_examples(
            query=query_text,
            query_lang=detected_lang,
            num_examples=2, # Get 2 examples
            source_lang=source_lang
        )

        formatted_examples = []
        for ex in in_context_examples_list:
            formatted_examples.append(f"English: {ex['source_lang_problem_solution']}\n{detected_lang.upper()}: {ex['target_lang_problem_solution']}")
        
        in_context_examples_str = "\n\n".join(formatted_examples)

        # Create the prompt using LangChain's PromptTemplate
        prompt = PromptTemplate(
            template=INCLT_PROMPT_TEMPLATE,
            input_variables=["target_language", "in_context_examples", "customer_query"]
        )

        final_prompt = prompt.format(
            target_language=detected_lang,
            in_context_examples=in_context_examples_str,
            customer_query=query_text
        )

        agent_response_text = ""
        if LLM:
            print(f"\n--- Sending Prompt to LLM ---\n{final_prompt}\n-----------------------------")
            llm_response = LLM.invoke(final_prompt)
            agent_response_text = llm_response.content
        else:
            # Simulate LLM response if no API key
            agent_response_text = (
                f"[SIMULATED LLM RESPONSE in {detected_lang.upper()}] "
                f"Thank you for your query about '{query_text}'. "
                f"We are processing your request based on the provided context. "
                f"Context examples used: {json.dumps(in_context_examples_list)}"
            )
        
        return AgentResponse(
            response=agent_response_text.strip(),
            language=detected_lang,
            context_used=in_context_examples_list
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run this application:
# 1. Make sure you have the required libraries installed: 
#    pip install fastapi uvicorn python-dotenv langchain-openai langchain-community sentence-transformers chromadb
# 2. Create a .env file in the same directory with OPENAI_API_KEY="YOUR_OPENAI_KEY" (optional)
# 3. Run: uvicorn global_care_agent:app --reload
# 4. Access the API at http://127.0.0.1:8000/docs for Swagger UI
