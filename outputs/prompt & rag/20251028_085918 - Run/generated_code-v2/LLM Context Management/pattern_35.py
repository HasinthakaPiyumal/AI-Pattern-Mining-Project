from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
import os

# Set up environment variables (replace with your actual API key)
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 1. Core Conversational LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Embedding Model for Long-Term Memory
embeddings = OpenAIEmbeddings()

# ChromaDB Setup for Long-Term Memory and Knowledge Base
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Function to ingest knowledge base documents
def ingest_knowledge_base(filepath):
    loader = TextLoader(filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    vectorstore.add_documents(split_docs)
    print(f"Ingested {len(split_docs)} documents from {filepath} into ChromaDB.")

# Example: Create a dummy knowledge base file and ingest it
with open("knowledge_base.txt", "w") as f:
    f.write("""
    Product A: Features include X, Y, Z. Common issues: issue1, issue2.
    Product B: Features include P, Q, R. Common issues: issue3, issue4.
    Return Policy: Items can be returned within 30 days with a receipt.
    Shipping Information: Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.
    """)
ingest_knowledge_base("knowledge_base.txt")

# Short-Term Memory: ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=500,
    memory_key="chat_history",
    return_messages=True,
    ai_prefix="Agent",
    human_prefix="Customer"
)

# 2. Custom "NotebookWrite" equivalent tool
@tool
def record_key_info(info: str) -> str:
    """Records critical information from the current conversation turn to be remembered for short-term and potentially long-term use."""
    # In a real scenario, this might update a dict in memory, or prepare for long-term storage
    # For this example, we'll just print it and store it in a temporary variable
    global current_session_notes
    current_session_notes.append(info)
    return f"Information recorded: {info}"

current_session_notes = [] # To simulate a scratchpad for short-term recorded info

# 3. Retrieval Tool for Long-Term Memory (RAG)
@tool
def retrieve_from_long_term_memory(query: str) -> str:
    """Retrieves relevant information from the long-term memory (knowledge base and past interactions) based on the query."""
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "No relevant information found in long-term memory."
    return "\n---\n".join([doc.page_content for doc in docs])

# Tool to summarize and store conversation history
@tool
def summarize_and_store_conversation(conversation_transcript: str, customer_id: str = "anonymous") -> str:
    """Summarizes a conversation transcript and stores it in long-term memory associated with a customer ID."""
    docs = [Document(page_content=conversation_transcript)]
    summarize_chain = load_summarize_chain(llm, chain_type="stuff")
    summary = summarize_chain.run(docs)
    
    # Store the summary in the vector database with metadata
    vectorstore.add_documents([
        Document(page_content=summary, metadata={"customer_id": customer_id, "type": "conversation_summary"})
    ])
    return f"Conversation summarized and stored for customer {customer_id}. Summary: {summary}"

# 4. Agent Orchestration
tools = [
    record_key_info,
    retrieve_from_long_term_memory,
    summarize_and_store_conversation
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent customer support agent. Help the customer with their inquiries. Use the provided tools when necessary."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# 5. Simplified Interaction Flow
def run_customer_interaction(customer_query):
    print(f"\nCustomer: {customer_query}")
    response = agent_executor.invoke({"input": customer_query})
    print(f"Agent: {response["output"]}")
    return response["output"]

# Example Interactions
print("\n--- Starting Customer Interaction 1 ---")
run_customer_interaction("Hi, I have a question about Product A. What are its features?")
run_customer_interaction("I also want to know about common issues with Product A.")
run_customer_interaction("Okay, thank you. Can you record that I am interested in troubleshooting for Product A?")
run_customer_interaction("Also, what is your return policy?")

print("\n--- Session Notes after Interaction 1 ---")
print(f"Recorded during session: {current_session_notes}")

# Simulate end of a complex interaction and store summary
print("\n--- Summarizing Interaction 1 for Long-Term Memory ---")
# The actual conversation transcript would come from memory.chat_history
history_for_summary = memory.load_memory_variables({})["chat_history"]
full_transcript = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in history_for_summary])
agent_executor.invoke({"input": f"Summarize the following conversation for customer CUST001 and store it: {full_transcript}"})

# Reset short-term memory for a new customer or new session
memory.clear()
current_session_notes = []

print("\n--- Starting Customer Interaction 2 (New Session/Customer) ---")
run_customer_interaction("Hello, I'm a new customer but I have questions about Product B. How long does standard shipping take?")
run_customer_interaction("If I want faster delivery, what are my options?")
run_customer_interaction("Can you tell me about the features of Product B?")

# Simulate a follow-up query from CUST001 (should retrieve from long-term memory)
print("\n--- Starting Customer Interaction 3 (Follow-up for CUST001) ---")
memory.clear() # Clear for a new interaction context
current_session_notes = []
run_customer_interaction("Hello, I am CUST001. Last time I asked about Product A troubleshooting. Can you remind me of the common issues with Product A?")