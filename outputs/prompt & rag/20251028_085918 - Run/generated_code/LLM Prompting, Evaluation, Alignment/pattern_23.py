from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import LLMChain, SequentialChain
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from guardrails.hub import ProfanityFree, ToxicityFree
from guardrails import Guard
from loguru import logger
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

app = FastAPI()

# --- 1. LLM and Embeddings Initialization ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
embeddings = OpenAIEmbeddings()

# --- 2. Vector Store (ChromaDB) for RAG ---
# In a real application, this would be loaded from persistent storage.
dummy_docs = [
    "Our refund policy states that items can be returned within 30 days for a full refund if unused.",
    "To reset your password, visit the 'Forgot Password' link on the login page and follow the instructions.",
    "Shipping usually takes 5-7 business days for standard delivery within the country.",
    "For technical support, please contact our dedicated IT team at support@example.com.",
    "Our premium subscription includes ad-free experience and exclusive content."
]

vectorstore = Chroma.from_texts(texts=dummy_docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

# --- 3. Prompt Engineering Module ---
def create_prompt(query: str, context: str) -> str:
    base_template = """
    You are an empathetic, calm, and helpful customer support agent. 
    Your goal is to assist customers accurately and ethically. 
    Based on the following context, please answer the customer's query.

    Context: {context}

    Customer Query: {query}

    Please provide a concise and polite response. If you cannot find the answer in the context, 
    kindly state that you don't have enough information and suggest alternative actions.
    """
    
    prompt = PromptTemplate(
        template=base_template,
        input_variables=["context", "query"]
    )
    return prompt.format(context=context, query=query)

# --- 4. Reasoning and Problem-Solving Module (Rephrase and Respond) ---
rephrase_prompt = PromptTemplate(
    template="""Rephrase the following customer query to clarify its intent. 
    Original query: {query}
    Rephrased query:""",
    input_variables=["query"]
)

rephrase_chain = LLMChain(llm=llm, prompt=rephrase_prompt, output_key="rephrased_query")

def get_rephrased_query(inputs: dict) -> str:
    return rephrase_chain.invoke({"query": inputs["query"]})["rephrased_query"]

# --- 5. Validation and Quality Assurance (Guardrails-AI) & Ethical Considerations ---
# Define a simple Pydantic model for the expected output structure
class AgentResponse(BaseModel):
    response: str
    is_helpful: bool

# Define a Guard for validation and ethical checks
guard = Guard.from_pydantic(output_class=AgentResponse, 
                           validators=[
                               ProfanityFree(on_fail="reask"), 
                               ToxicityFree(on_fail="reask"),
                            ])

# --- 6. Langchain Orchestration ---
def process_query(query: str):
    logger.info(f"Received query: {query}")

    # Step 1: Rephrase the query for clarity
    rephrased_query = get_rephrased_query({"query": query})
    logger.info(f"Rephrased query: {rephrased_query}")

    # Step 2: Retrieve relevant context
    context_docs = retriever.invoke(rephrased_query)
    context = "\n".join([doc.page_content for doc in context_docs])
    logger.info(f"Retrieved context: {context[:200]}...")

    # Step 3: Create the final prompt with context and ethical instructions
    final_prompt_text = create_prompt(query=rephrased_query, context=context)

    # Step 4: Generate LLM response
    raw_llm_response = llm.invoke(final_prompt_text).content
    logger.info(f"Raw LLM response: {raw_llm_response}")

    # Step 5: Validate and refine response using Guardrails
    try:
        validated_response = guard.validate(raw_llm_response)
        logger.info(f"Validated response: {validated_response.rail.output}")
        # For simplicity, we assume if validation passes, it's helpful and extract the response string
        final_agent_response = validated_response.rail.output.response
    except Exception as e:
        logger.error(f"Guardrails validation failed: {e}")
        final_agent_response = "I apologize, but I encountered an issue while processing your request. Please try again or contact a human agent."
    
    return final_agent_response

# --- 7. FastAPI Endpoint ---
class QueryRequest(BaseModel):
    query: str

class AgentResponseModel(BaseModel):
    response: str

@app.post("/ask", response_model=AgentResponseModel)
async def ask_agent(request: QueryRequest):
    try:
        response_text = process_query(request.query)
        return AgentResponseModel(response=response_text)
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# To run this file:
# 1. Install dependencies: pip install fastapi uvicorn langchain langchain-openai chromadb sentence-transformers guardrails-ai pydantic loguru
# 2. Replace "YOUR_OPENAI_API_KEY" with your actual OpenAI API key.
# 3. Run the application: uvicorn customer_support_agent:app --reload
# 4. Access the API at http://127.0.0.1:8000/docs