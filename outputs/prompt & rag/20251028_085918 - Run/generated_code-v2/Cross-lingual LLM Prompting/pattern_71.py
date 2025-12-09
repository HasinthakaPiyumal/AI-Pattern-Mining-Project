from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

# Set seed for reproducibility in langdetect
DetectorFactory.seed = 0

app = FastAPI()

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    detected_language: str

# Initialize Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="customer_support_kb")

# Populate Knowledge Base (InCLT examples)
# Each example has a source query in one language and a target answer in another (or same)
# For InCLT, we explicitly store both source_query and target_answer languages.
kb_entries = [
    {
        "id": "kb_en_1",
        "source_query": "What is the shipping cost?",
        "target_answer": "Shipping costs vary based on your location and the selected delivery method. Please refer to our shipping policy page for details.",
        "source_lang": "en",
        "target_lang": "en"
    },
    {
        "id": "kb_es_1",
        "source_query": "¿Cuál es el costo de envío?",
        "target_answer": "El costo de envío varía según su ubicación y el método de entrega seleccionado. Consulte nuestra página de política de envío para obtener más detalles.",
        "source_lang": "es",
        "target_lang": "es"
    },
    {
        "id": "kb_fr_1",
        "source_query": "Comment puis-je suivre ma commande?",
        "target_answer": "Vous pouvez suivre votre commande en utilisant le numéro de suivi fourni dans votre email de confirmation d'expédition.",
        "source_lang": "fr",
        "target_lang": "fr"
    },
    {
        "id": "kb_en_2",
        "source_query": "How do I return an item?",
        "target_answer": "To return an item, please visit our returns portal within 30 days of purchase and follow the instructions.",
        "source_lang": "en",
        "target_lang": "en"
    },
    {
        "id": "kb_de_1",
        "source_query": "Wie kann ich meine Bestellung stornieren?",
        "target_answer": "Um Ihre Bestellung zu stornieren, kontaktieren Sie bitte unseren Kundenservice so schnell wie möglich.",
        "source_lang": "de",
        "target_lang": "de"
    }
]

# Embed and add to ChromaDB if not already present
if collection.count() == 0:
    documents = [entry["source_query"] for entry in kb_entries]
    metadatas = [entry for entry in kb_entries]
    ids = [entry["id"] for entry in kb_entries]
    embeddings = embedding_model.encode(documents).tolist()
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

# Initialize LLM (using a small local model for demonstration)
# For a real application, consider a larger multilingual model like mBERT, XLM-R, or an API-based LLM (OpenAI, Cohere)
llm_pipeline = pipeline("text-generation", model="distilgpt2", tokenizer="distilgpt2", max_new_tokens=100)

# InCLT Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant for an e-commerce platform. Your goal is to answer customer questions concisely and accurately, leveraging the provided context."),
    ("user", "Here are some examples of customer questions and their answers in different languages for context:"),
    ("user", "{examples}"),
    ("user", "Now, answer the following customer query in English based on the context, if applicable:"),
    ("user", "Customer query ({query_lang}): {customer_query}")
])

# Helper function for LLM generation with a simple pipeline
def generate_response_with_llm(prompt_text: str) -> str:
    # The distilgpt2 model is not designed for instruction following or cross-lingual tasks.
    # This is a placeholder for demonstration. A proper multilingual LLM or an API-based LLM
    # integrated via LangChain's LLM class would be used here.
    response = llm_pipeline(prompt_text, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)[0]["generated_text"]
    # Clean up the response to only include the part after the prompt
    return response[len(prompt_text):].strip()

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: QueryRequest):
    query_text = request.query

    # 1. Language Detection
    try:
        detected_lang = detect(query_text)
    except:
        detected_lang = "unknown"

    # 2. Retrieve relevant KB entries (documents from ChromaDB)
    # We search based on the embedding of the incoming query
    query_embedding = embedding_model.encode(query_text).tolist()
    retrieved_docs = collection.query(
        query_embeddings=[query_embedding],
        n_results=3, # Retrieve top 3 relevant examples
        include=["documents", "metadatas"]
    )

    examples_for_prompt = []
    if retrieved_docs and retrieved_docs["metadatas"][0]:
        for meta in retrieved_docs["metadatas"][0]:
            examples_for_prompt.append(
                f"Customer ({meta['source_lang']}): {meta['source_query']}\n"
                f"Answer ({meta['target_lang']}): {meta['target_answer']}"
            )
    
    # Format examples for the prompt
    formatted_examples = "\n\n".join(examples_for_prompt)

    # 3. InCLT Prompt Engineering
    full_prompt_chain = prompt_template | StrOutputParser()
    final_prompt = full_prompt_chain.invoke({
        "examples": formatted_examples,
        "query_lang": detected_lang,
        "customer_query": query_text
    })

    # 4. Multilingual Large Language Model (LLM) processing
    # The actual LLM call will be simplified for this demonstration using a local pipeline
    llm_response_text = generate_response_with_llm(final_prompt)

    # 5. Response Output
    return ChatResponse(answer=llm_response_text, detected_language=detected_lang)

# To run the app:
# 1. Make sure you have the required libraries installed:
#    pip install fastapi uvicorn langdetect sentence-transformers chromadb langchain-core transformers pydantic
# 2. Run from your terminal: uvicorn main:app --reload
# 3. Access the API at http://127.0.0.1:8000/docs for Swagger UI. 