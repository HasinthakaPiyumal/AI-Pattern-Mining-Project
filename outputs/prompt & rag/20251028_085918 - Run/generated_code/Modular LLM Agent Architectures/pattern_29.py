from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Knowledge Retrieval Module
knowledge_base_documents = [
    "Our product 'SuperWidget' features include automated scheduling, data analytics, and real-time reporting.",
    "To troubleshoot SuperWidget, first check your internet connection, then restart the device. If issues persist, contact support.",
    "The price of SuperWidget Pro is $99/month, and the basic version is $49/month.",
    "SuperWidget is compatible with Windows, macOS, and Linux operating systems.",
    "You can find user manuals and tutorials for SuperWidget on our website under the 'Support' section."
]

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_base_embeddings = embedding_model.encode(knowledge_base_documents)

def retrieve_knowledge(query: str) -> str:
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, knowledge_base_embeddings)[0]
    most_similar_idx = np.argmax(similarities)
    if similarities[most_similar_idx] > 0.7:  # Threshold for relevance
        return knowledge_base_documents[most_similar_idx]
    return "I could not find a direct answer in our knowledge base. Would you like to rephrase or escalate?"

# 2. Custom Tools
@tool
def ProductKnowledgeTool(query: str) -> str:
    """Use this tool to search the product knowledge base for answers to specific questions about the product. Input should be a clear, concise question."""
    print(f"\n>>> Using ProductKnowledgeTool for query: {query}")
    return retrieve_knowledge(query)

@tool
def HumanEscalationTool(reason: str) -> str:
    """Use this tool to escalate the conversation to a human agent when the AI cannot resolve the issue or when explicitly requested by the user. Input should be a brief reason for escalation."""
    print(f"\n>>> Escalating to human agent for reason: {reason}")
    return f"I'm escalating your issue to a human agent. Reason: {reason}. A support representative will contact you shortly."

tools = [ProductKnowledgeTool, HumanEscalationTool]

# 3. LLM (using Ollama as an example, replace with your preferred LLM)
# Ensure Ollama is running and you have a model pulled, e.g., 'ollama pull llama2'
llm = ChatOllama(model="llama2") # Or replace with ChatOpenAI(model="gpt-3.5-turbo") or similar

# 4. Conversation Memory Module
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 5. Planning Module / Agent Setup
# Define the prompt for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support AI assistant. Answer user questions to the best of your ability. If you cannot find an answer in your knowledge base, offer to escalate to a human. Be polite and professional."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an AgentExecutor to run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

def run_customer_agent(query: str) -> str:
    response = agent_executor.invoke({"input": query})
    return response["output"]

if __name__ == "__main__":
    print("Welcome to the Composable AI Customer Support Agent! Type 'exit' to end the conversation.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            break
        agent_response = run_customer_agent(user_query)
        print(f"Agent: {agent_response}")
