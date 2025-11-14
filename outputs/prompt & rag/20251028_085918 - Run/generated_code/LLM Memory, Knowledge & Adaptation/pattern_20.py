import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain and related imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings # More realistic than a dummy, but requires download
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory

import os
import collections

# --- Configuration and Mock Data ---

# For demonstration, we'll use a simple in-memory ChromaDB and mock LLM.
# In a real application, you'd configure a persistent ChromaDB and an actual LLM (e.g., OpenAI, HuggingFace).

# Mock Knowledge Base for Long-Term Memory
KNOWLEDGE_BASE_DOCUMENTS = [
    "Our shipping policy states that orders are processed within 1-2 business days and delivery takes 5-7 business days.",
    "To return an item, please visit our returns portal on the website and follow the instructions. Returns are accepted within 30 days of purchase.",
    "You can track your order using the tracking number provided in your shipping confirmation email.",
    "Our customer service hours are Monday to Friday, 9 AM to 5 PM EST.",
    "We offer a wide range of electronics including laptops, smartphones, and smartwatches.",
    "The 'ProGadget X' features a 12-hour battery life, 128GB storage, and a 1080p display.",
    "For technical support, please describe your issue in detail, and we will connect you with a specialist.",
    "Payment methods accepted include Visa, Mastercard, American Express, and PayPal.",
    "To reset your password, click on the 'Forgot Password' link on the login page.",
    "Our loyalty program offers 10% off on your first purchase and exclusive discounts for members."
]

# --- Mock LLM and Embeddings for Demo Purposes ---
# In a real scenario, replace these with actual implementations (e.g., ChatOpenAI, HuggingFaceEmbeddings)

class MockChatModel(BaseChatModel):
    """A mock LLM for demonstration purposes."""
    def _generate(self, messages, stop=None, callbacks=None, **kwargs):
        last_message = messages[-1].content
        
        # Simple keyword-based responses for demonstration
        if "shipping" in last_message.lower():
            response = "According to our shipping policy, orders are processed within 1-2 business days and delivered in 5-7 business days."
        elif "return" in last_message.lower():
            response = "You can initiate a return via our returns portal within 30 days of purchase."
        elif "track order" in last_message.lower():
            response = "Please use the tracking number from your shipping confirmation email to track your order."
        elif "hello" in last_message.lower() or "hi" in last_message.lower():
            response = "Hello! How can I assist you today?"
        elif "product" in last_message.lower() or "gadget" in last_message.lower():
            response = "Please tell me more about the product you are interested in. For example, the 'ProGadget X' has a 12-hour battery life."
        elif "payment" in last_message.lower():
            response = "We accept Visa, Mastercard, American Express, and PayPal."
        elif "password" in last_message.lower() or "reset" in last_message.lower():
            response = "To reset your password, please use the 'Forgot Password' link on the login page."
        elif "technical" in last_message.lower():
            response = "For technical support, please describe your issue in detail."
        else:
            response = "I'm sorry, I couldn't find a direct answer to your question. Can you please rephrase or provide more details?"

        return AIMessage(content=response)

    @property
    def _llm_type(self) -> str:
        return "mock-chat"

# Initialize Mock LLM
llm = MockChatModel()

# Initialize Embeddings for ChromaDB
# Note: This will download a model the first time it's run.
# For a truly isolated demo without external downloads, you would mock this.
# from langchain_community.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# For this example, we'll use a placeholder for SentenceTransformerEmbeddings
# but acknowledge that a real setup would download a model.
class DummyEmbeddings:
    def embed_documents(self, texts):
        return [[1.0] * 384 for _ in texts] # Dummy 384-dim vectors
    def embed_query(self, text):
        return [1.0] * 384

embeddings = DummyEmbeddings() # Using dummy for self-contained code

# --- Long-Term Memory (ChromaDB) ---

# Split documents into chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.create_documents(KNOWLEDGE_BASE_DOCUMENTS)

# Initialize ChromaDB with the knowledge base
vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, collection_name="ecommerce_kb")
retriever = vectorstore.as_retriever()

# --- Short-Term Memory (LangChain Conversational Memory) ---
# Using a dictionary to hold memory for multiple sessions (users)
conversation_memories = collections.defaultdict(lambda: ConversationBufferWindowMemory(k=5, return_messages=True, output_key="answer"))

# --- Query Classifier Module ---
# A simple rule-based classifier for demonstration. 
# In a real app, this would be an ML model (e.g., fine-tuned BERT).
def classify_query(query: str) -> str:
    query_lower = query.lower()
    if "shipping" in query_lower or "delivery" in query_lower or "track" in query_lower:
        return "Order Status"
    elif "return" in query_lower or "refund" in query_lower:
        return "Returns"
    elif "product" in query_lower or "item" in query_lower or "specs" in query_lower:
        return "Product Inquiry"
    elif "tech" in query_lower or "technical" in query_lower or "issue" in query_lower:
        return "Technical Support"
    elif "hello" in query_lower or "hi" in query_lower or "greeting" in query_lower:
        return "Greeting"
    else:
        return "General Inquiry"

# --- LLM Orchestration (LangChain Chains) ---

# Prompt for RAG (Retrieval Augmented Generation)
ragn_template = """You are an AI customer support agent for an e-commerce store.
Answer the user's question based on the provided context and the chat history.
If you don't know the answer, state that you don't have enough information.

Context: {context}
Chat History: {chat_history}
Human: {question}
AI:"""
rag_prompt = ChatPromptTemplate.from_template(ragn_template)

# Prompt for general inquiries (when RAG might not be explicitly needed or context is sufficient from history)
general_template = """You are an AI customer support agent for an e-commerce store.
Maintain a friendly and helpful tone. Use the chat history to understand context.

Chat History: {chat_history}
Human: {question}
AI:"""
general_prompt = ChatPromptTemplate.from_template(general_template)

# --- FastAPI Application ---

app = FastAPI(
    title="E-commerce Intelligent Customer Support Agent",
    description="An AI-powered customer support agent for e-commerce, featuring adaptive LLM augmentation and memory management."
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    agent_response: str
    query_category: str

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Get short-term memory for the current user
    memory = conversation_memories[user_id]
    current_chat_history = memory.load_memory_variables({})["history"]

    # Classify the query
    query_category = classify_query(user_message)

    context = ""
    if query_category in ["Order Status", "Returns", "Product Inquiry", "Technical Support", "General Inquiry"]:
        # Retrieve relevant context from Long-Term Memory (RAG)
        docs = retriever.invoke(user_message)
        context = "\n".join([doc.page_content for doc in docs])

    # Select prompt based on query category (simplified for demo)
    if context:
        prompt = rag_prompt
        input_data = {"context": context, "chat_history": current_chat_history, "question": user_message}
    else:
        prompt = general_prompt
        input_data = {"chat_history": current_chat_history, "question": user_message}
    
    # LLM Chain setup
    rag_chain = (
        RunnablePassthrough.assign(chat_history=lambda x: memory.load_memory_variables({})["history"]) 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    # Invoke the LLM chain
    try:
        agent_response = rag_chain.invoke(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    # Update short-term memory with the new interaction
    memory.save_context({"input": user_message}, {"answer": agent_response})

    return ChatResponse(agent_response=agent_response, query_category=query_category)

# --- Offline Fine-tuning Module Description ---
# This part is described as an *offline* process in the architecture, 
# so it's not directly integrated into the real-time API. 
# It would involve separate scripts and a workflow like this:

# 1. Data Collection: Gather new product data, successful interaction logs, updated FAQs.
# 2. Data Preprocessing: Format data for fine-tuning (e.g., Q&A pairs, conversational turns).
# 3. Model Selection: Choose an LLM and potentially an embedding model to fine-tune.
# 4. Fine-tuning with TRL/PEFT: Use libraries like Hugging Face's `trl` or `peft`.
#    Example (conceptual): 
#    from trl import SFTTrainer
#    from transformers import AutoModelForCausalLM, AutoTokenizer
#    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
#    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=finetune_dataset, ...)
#    trainer.train()
# 5. Evaluation: Evaluate the fine-tuned model's performance on a validation set.
# 6. Deployment: Update the LLM used in the real-time system with the fine-tuned version.


if __name__ == "__main__":
    # To run this FastAPI application:
    # 1. Save the code as 'ecommerce_support_agent.py'
    # 2. Install necessary libraries: pip install fastapi uvicorn pydantic langchain-core langchain-community
    # 3. Run from your terminal: uvicorn ecommerce_support_agent:app --reload
    # Then access the API at http://127.0.0.1:8000/docs
    print("\n--- E-commerce Intelligent Customer Support Agent API ---")
    print("To run the application, save this code as 'ecommerce_support_agent.py' and run:")
    print("  pip install fastapi uvicorn pydantic langchain-core langchain-community")
    print("  uvicorn ecommerce_support_agent:app --reload")
    print("Access the API documentation at http://127.0.0.1:8000/docs after running.\n")

    # This block is for demonstrating the ChromaDB and LLM without running the FastAPI server for testing.
    # In a real setup, uvicorn will handle the app execution.
    # For direct testing, you might instantiate and test components here.

    # Example of direct interaction (without FastAPI):
    # print("\n--- Direct Interaction Example (for testing components) ---")
    # test_user_id = "test_user_123"
    # test_memory = conversation_memories[test_user_id]
    # 
    # query1 = "What is your shipping policy?"
    # print(f"\nUser: {query1}")
    # docs1 = retriever.invoke(query1)
    # context1 = "\n".join([doc.page_content for doc in docs1])
    # 
    # # Using the RAG chain directly
    # rag_chain_test = (
    #     RunnablePassthrough.assign(chat_history=lambda x: test_memory.load_memory_variables({})["history"]) 
    #     | rag_prompt 
    #     | llm 
    #     | StrOutputParser()
    # )
    # 
    # response1 = rag_chain_test.invoke({"context": context1, "chat_history": test_memory.load_memory_variables({})["history"], "question": query1})
    # print(f"Agent: {response1}")
    # test_memory.save_context({"input": query1}, {"answer": response1})
    # 
    # query2 = "And how about returns?"
    # print(f"\nUser: {query2}")
    # docs2 = retriever.invoke(query2)
    # context2 = "\n".join([doc.page_content for doc in docs2])
    # response2 = rag_chain_test.invoke({"context": context2, "chat_history": test_memory.load_memory_variables({})["history"], "question": query2})
    # print(f"Agent: {response2}")
    # test_memory.save_context({"input": query2}, {"answer": response2})

    # Run the FastAPI app if this script is executed directly
    # uvicorn.run(app, host="0.0.0.0", port=8000)



