
import gradio as gr
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Core LLM (Mock/Placeholder for demonstration)
class MockLLM:
    def __init__(self):
        # Using a simple text generation pipeline from transformers for demonstration
        # In a real application, this would be a more powerful LLM like Llama, GPT, etc.
        self.llm_pipeline = pipeline("text-generation", model="distilgpt2", framework="pt")

    def invoke(self, prompt):
        # For simplicity, just return a generated text based on the prompt
        # Real LLMs would handle more complex reasoning
        response = self.llm_pipeline(prompt, max_new_tokens=100, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
        return response[0]["generated_text"]

mock_llm = MockLLM()

# 2. Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Vector Database (ChromaDB for Long-Term Memory)
client = chromadb.Client()
collection_name = "medical_knowledge"

try:
    medical_knowledge_db = client.get_or_create_collection(name=collection_name)
except Exception as e:
    print(f"Error getting/creating collection: {e}. Attempting to delete and recreate.")
    client.delete_collection(name=collection_name)
    medical_knowledge_db = client.get_or_create_collection(name=collection_name)


# Add some dummy medical documents
dummy_docs = [
    "Symptoms of common cold include runny nose, sore throat, cough, congestion, slight body aches or a mild headache, sneezing, and low-grade fever.",
    "Influenza, or flu, is a contagious respiratory illness caused by flu viruses. Symptoms can include fever, cough, sore throat, runny or stuffy nose, muscle or body aches, headaches, and fatigue. Complications can include pneumonia.",
    "Diabetes is a chronic condition that affects how your body turns food into energy. Most of the food you eat is broken down into sugar (glucose) and released into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin. Insulin acts like a key to let blood sugar into your body’s cells for use as energy.",
    "Hypertension, also known as high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. It often has no symptoms."
]
dummy_ids = [f"doc{i}" for i in range(len(dummy_docs))]
dummy_embeddings = embedding_model.encode(dummy_docs).tolist()

# Add documents if the collection is empty
if medical_knowledge_db.count() == 0:
    medical_knowledge_db.add(
        embeddings=dummy_embeddings,
        documents=dummy_docs,
        metadatas=[{"source": "medical textbook"} for _ in dummy_docs],
        ids=dummy_ids
    )
    print(f"Added {len(dummy_docs)} dummy documents to ChromaDB.")
else:
    print(f"ChromaDB already contains {medical_knowledge_db.count()} documents. Skipping addition of dummy data.")

# 4. Conversational Memory (Short-Term Memory)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 5. Query Complexity Classifier (Rule-based for simplicity)
def classify_query(query: str) -> str:
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in ["symptoms", "diagnose", "what if", "medical advice", "treatment"]):
        return "complex"
    return "simple"

# 6. Orchestration Framework (LangChain-like logic)
# Define a custom retriever for ChromaDB
class ChromaRetriever:
    def __init__(self, chroma_collection, embedding_model, k=3):
        self.chroma_collection = chroma_collection
        self.embedding_model = embedding_model
        self.k = k

    def get_relevant_documents(self, query: str):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=self.k,
            include=['documents']
        )
        return results['documents'][0] if results['documents'] else []

chroma_retriever = ChromaRetriever(medical_knowledge_db, embedding_model)

# Template for LLM prompt
rag_prompt_template = """You are an AI medical assistant. Answer the user's question based on the provided context and conversation history. 
If you don't know the answer from the provided information, state that you don't have enough information. 

Conversation History:
{chat_history}

Retrieved Medical Context:
{context}

User Question: {question}
AI Assistant:"""

simple_prompt_template = """You are an AI medical assistant. Answer the user's question based on the conversation history. 

Conversation History:
{chat_history}

User Question: {question}
AI Assistant:"""

# Function to handle the entire interaction flow
def medical_assistant_chain(user_input: str, chat_history: list) -> str:
    # Update short-term memory with current chat history from Gradio
    # Gradio passes chat_history as a list of tuples, memory expects messages
    memory.clear()
    for human_msg, ai_msg in chat_history:
        memory.save_context({"input": human_msg}, {"output": ai_msg})
    
    current_chat_history = memory.load_memory_variables({})["chat_history"]
    formatted_chat_history = "\n".join([f"Human: {msg.content}" if msg.type == "human" else f"AI: {msg.content}" for msg in current_chat_history])

    query_type = classify_query(user_input)
    context_docs = []

    if query_type == "complex":
        context_docs = chroma_retriever.get_relevant_documents(user_input)
        context_str = "\n".join(context_docs)
        prompt = PromptTemplate.from_template(rag_prompt_template)
        chain = ({"context": RunnablePassthrough(), "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()} 
                 | prompt 
                 | mock_llm.invoke 
                 | StrOutputParser())
        
        response = chain.invoke({"context": context_str, "question": user_input, "chat_history": formatted_chat_history})

    else: # simple query
        prompt = PromptTemplate.from_template(simple_prompt_template)
        chain = ({"question": RunnablePassthrough(), "chat_history": RunnablePassthrough()} 
                 | prompt 
                 | mock_llm.invoke 
                 | StrOutputParser())
        response = chain.invoke({"question": user_input, "chat_history": formatted_chat_history})

    # Update memory after getting LLM response
    memory.save_context({"input": user_input}, {"output": response})

    return response

# Gradio Interface
def greet_and_respond(message, history):
    # `history` is a list of tuples: [(user_message, bot_message), ...]
    # `message` is the current user input
    return medical_assistant_chain(message, history)

# Gradio UI
with gr.Blocks(title="Adaptive AI Medical Assistant") as demo:
    gr.Markdown("# Adaptive AI Medical Assistant")
    gr.Markdown("Ask me anything about common medical conditions or symptoms.")
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Your Question")
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(greet_and_respond, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)

demo.launch()