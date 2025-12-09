import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- Knowledge Base Setup (Long-Term Memory Simulation) ---
KNOWLEDGE_BASE_FILE = "knowledge_base.txt"
CHROMA_PERSIST_DIR = "./chroma_db"

# Create a dummy knowledge base file if it doesn't exist
if not os.path.exists(KNOWLEDGE_BASE_FILE):
    with open(KNOWLEDGE_BASE_FILE, "w") as f:
        f.write("""
Product Information:
- Product A: High-quality smartphone with 128GB storage, 6.1-inch display. Price: $799.
- Product B: Wireless earbuds with noise cancellation, 24-hour battery life. Price: $149.
- Product C: Smartwatch with heart rate monitor, GPS, and waterproof design. Price: $299.

FAQs:
- How can I track my order? You can track your order using the tracking number provided in your shipping confirmation email on our website.
- What is your return policy? We offer a 30-day return policy for unused items in their original packaging. Please visit our returns page for more details.
- How do I contact customer support? You can reach us via email at support@ecommerce.com or by calling 1-800-123-4567 during business hours.

Shipping Information:
- Standard shipping takes 5-7 business days.
- Express shipping takes 2-3 business days.
- Free shipping on all orders over $50.

Past Resolutions:
- Customer 'John Doe' had an issue with a damaged Product A. We offered a full refund or replacement. He chose a replacement.
- Customer 'Jane Smith' inquired about the compatibility of Product B with her device. We confirmed it's compatible with all Bluetooth-enabled devices.
""")

# Load documents
loader = TextLoader(KNOWLEDGE_BASE_FILE)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings()

# Initialize Chroma vector store (Long-Term Memory)
# It will persist the database to disk, so it only needs to be built once.
if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
    print("Initializing Chroma DB for the first time...")
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIR)
    vectordb.persist()
    print("Chroma DB initialized.")
else:
    print("Loading existing Chroma DB...")
    vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    print("Chroma DB loaded.")

retriever = vectordb.as_retriever()

# --- LLM Core ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# --- Short-Term Memory (Working Memory) ---
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", 
    return_messages=True, 
    k=5 # Keep track of the last 5 turns of conversation
)

# --- Prompt Template ---
custom_template = """You are an intelligent customer support agent for an e-commerce platform. Your goal is to provide helpful and accurate information to customers based on the provided context and chat history. 
If you don't know the answer, state that you don't know rather than making up an answer. Always be polite and professional.

Chat History:
{chat_history}

Context from Knowledge Base:
{context}

Customer Question: {question}
Agent Response:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# --- Orchestration Layer (ConversationalRetrievalChain) ---
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": CUSTOM_QUESTION_PROMPT}
)

# --- User Interface (Simple Text-Based) ---
print("\n--- Intelligent Customer Support Agent ---")
print("Hello! How can I assist you today? Type 'exit' to end the conversation.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'exit':
        print("Agent: Goodbye! Have a great day.")
        break

    try:
        # Invoke the chain with the user's question
        result = qa_chain.invoke({"question": user_input})
        print(f"Agent: {result['answer']}")
    except Exception as e:
        print(f"Agent: I apologize, but an error occurred: {e}. Please try again later.")

