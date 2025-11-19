from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Initialize FastAPI app
app = FastAPI()

# --- LLM and vLLM Setup ---
# Replace with your desired LLM from Hugging Face. Ensure it's compatible with vLLM.
# For example, 'mistralai/Mistral-7B-Instruct-v0.2' or 'HuggingFaceH4/zephyr-7b-beta'
LLM_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
llm = LLM(model=LLM_MODEL_NAME)
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)

# --- Embedding Model Setup ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embeddings_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# --- Chroma DB Setup ---
# For a real application, you'd load your persistent Chroma DB here.
# For demonstration, we'll create an in-memory one with some dummy data.
dummy_documents = [
    "How do I track my order? You can track your order using the tracking number provided in your shipping confirmation email.",
    "What is your return policy? We offer a 30-day return policy on most items, provided they are in their original condition.",
    "How can I change my shipping address? Please contact customer support immediately after placing your order to change the shipping address.",
    "Do you offer international shipping? Yes, we ship to over 100 countries worldwide. Shipping fees and times vary by destination.",
    "How do I apply a discount code? Discount codes can be applied during checkout in the designated 'Discount Code' field."
]

vectorstore = Chroma.from_texts(
    texts=dummy_documents,
    embedding=embeddings_model,
    collection_name="ecommerce_faqs"
)
retriever = vectorstore.as_retriever()

# --- LangChain RAG Setup ---
# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an helpful e-commerce customer support assistant. Answer the user's question based on the provided context. If you cannot find the answer, politely state that you don't have enough information."),
    ("human", "Context: {context}\nQuestion: {question}")
])

# Custom runnable for vLLM integration
class VLLMRunnable:
    def __init__(self, llm_engine, sampling_params_config):
        self.llm_engine = llm_engine
        self.sampling_params_config = sampling_params_config

    def invoke(self, input_text: str) -> str:
        outputs = self.llm_engine.generate(prompts=[input_text], sampling_params=self.sampling_params_config)
        # Assuming single output for simplicity in this chain
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return ""

vllm_runnable = VLLMRunnable(llm, sampling_params)

# Construct the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | vllm_runnable
    | StrOutputParser()
)

# --- FastAPI Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# --- FastAPI Endpoint ---
@app.post("/chat", response_model=QueryResponse)
async def chat_with_bot(request: QueryRequest):
    try:
        response_text = rag_chain.invoke(request.query)
        return QueryResponse(response=response_text)
    except Exception as e:
        print(f"Error processing request: {e}")
        return QueryResponse(response="Sorry, I am currently experiencing technical difficulties. Please try again later.")

# To run this application:
# 1. Install necessary libraries: pip install fastapi uvicorn vllm "langchain_community[chroma]" sentence-transformers langchain-core
# 2. Make sure you have a CUDA-enabled GPU and appropriate drivers for vLLM.
# 3. Download the LLM model specified by LLM_MODEL_NAME.
# 4. Run: uvicorn chatbot_service:app --host 0.0.0.0 --port 8000
