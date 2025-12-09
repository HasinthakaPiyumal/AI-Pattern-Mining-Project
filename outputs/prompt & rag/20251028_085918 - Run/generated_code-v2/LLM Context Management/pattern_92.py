from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.language_models import BaseLanguageModel


class SimulatedLLM(BaseLanguageModel):
    def invoke(self, prompt, stop=None, config=None):
        # In a real scenario, this would call a large language model API or local model
        # For demonstration, we simply process the prompt content.
        
        # The prompt will typically be a list of messages. We extract the relevant parts.
        # Assuming the last message is the user's question and previous messages
        # contain the context provided by the RAG system.
        
        context = "No relevant context provided." # Default if no context found
        question = ""

        if isinstance(prompt, list):
            for msg in prompt:
                if hasattr(msg, 'type') and msg.type == 'system' and 'context' in msg.content.lower():
                    context = msg.content.replace("You are a helpful medical assistant. Use the following context to answer the question:\n", "").strip()
                elif hasattr(msg, 'type') and msg.type == 'human':
                    question = msg.content
        
        response_content = f"Based on the medical knowledge, and the provided context: '{context}', here is the answer to your question: '{question}'. This is a simulated response."    
        return AIMessage(content=response_content)

    @property
    def _llm_type(self) -> str:
        return "simulated-llm"


# 1. Non-Parametric Memory (Knowledge Base & Retriever)
medical_knowledge_base = [
    "Influenza, commonly known as the flu, is an infectious disease caused by influenza viruses. Symptoms include fever, runny nose, sore throat, muscle pains, headache, coughing, and fatigue.",
    "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain. It should not be given to children with viral infections due to the risk of Reye's syndrome.",
    "Diabetes mellitus is a metabolic disease that causes high blood sugar. The two main types are Type 1 (insulin-dependent) and Type 2 (non-insulin-dependent).",
    "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
    "Antibiotics are medications that destroy or slow down the growth of bacteria. They are used to treat bacterial infections but are ineffective against viral infections like the common cold or flu.",
    "The common cold is a viral infectious disease of the upper respiratory tract that primarily affects the nose. Symptoms include coughing, sore throat, runny nose, sneezing, and fever, which usually resolve in 7 to 10 days.",
    "Asthma is a chronic lung disease that inflames and narrows the airways. It causes recurring periods of wheezing, chest tightness, shortness of breath, and coughing."
]

# Embedding Model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector Store and Retriever
vectorstore = FAISS.from_texts(medical_knowledge_base, embeddings)
retriever = vectorstore.as_retriever()

# 2. Parametric Memory (Generative Language Model - LLM)
llm = SimulatedLLM()

# 3. Orchestration (Retrieval Augmented Generation - RAG Chain)
# Prompt Template
# The context comes from the retriever and is passed into the system message.
# The human message contains the user's question.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medical assistant. Use the following context to answer the question:\n{context}"),
    ("human", "{question}"),
])

# RAG Workflow Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example Usage
if __name__ == "__main__":
    print("\n--- Medical Diagnostic and Information Assistant ---\n")

    query1 = "What are the symptoms of the flu?"
    print(f"User Query: {query1}")
    response1 = rag_chain.invoke(query1)
    print(f"Assistant: {response1}")
    print("\n" + "="*80 + "\n")

    query2 = "Can I give aspirin to a child with a viral infection?"
    print(f"User Query: {query2}")
    response2 = rag_chain.invoke(query2)
    print(f"Assistant: {response2}")
    print("\n" + "="*80 + "\n")

    query3 = "What is diabetes?"
    print(f"User Query: {query3}")
    response3 = rag_chain.invoke(query3)
    print(f"Assistant: {response3}")
    print("\n" + "="*80 + "\n")

    query4 = "Tell me about healthy eating."
    print(f"User Query: {query4}")
    response4 = rag_chain.invoke(query4)
    print(f"Assistant: {response4}")
    print("\n" + "="*80 + "\n")
