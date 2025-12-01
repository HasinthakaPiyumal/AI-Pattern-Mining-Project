import os
import getpass

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration and Environment Setup ---

# Ensure OPENAI_API_KEY is set
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Please enter your OpenAI API key:")

# --- Long-Term Memory (RAG Components) ---

class MemoryManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Customer History Database
        self.customer_history_db = Chroma(embedding_function=self.embeddings, collection_name="customer_history")
        self.populate_customer_history()

        # Knowledge Base Database
        self.knowledge_base_db = Chroma(embedding_function=self.embeddings, collection_name="knowledge_base")
        self.populate_knowledge_base()

    def populate_customer_history(self):
        # Dummy customer history data
        docs = [
            ("Customer ID: 1001. Previous issue: Internet slow. Resolution: Router reset. Date: 2023-10-26", {"customer_id": "1001"}),
            ("Customer ID: 1001. Previous query: Billing inquiry about extra charges. Resolution: Explained data overage. Date: 2023-11-15", {"customer_id": "1001"}),
            ("Customer ID: 1002. Previous issue: Cable TV pixelation. Resolution: Signal refresh. Date: 2023-12-01", {"customer_id": "1002"}),
            ("Customer ID: 1003. New plan requested: Fiber Optic 1Gbps. Date: 2024-01-05", {"customer_id": "1003"}),
            ("Customer ID: 1001. Current service plan: Basic Internet, Standard TV. Account status: Active.", {"customer_id": "1001"}),
        ]
        self.customer_history_db.add_texts([doc[0] for doc in docs], metadatas=[doc[1] for doc in docs])
        print("Customer history database populated.")

    def populate_knowledge_base(self):
        # Dummy knowledge base data
        docs = [
            "Troubleshooting internet issues: Check modem lights, restart router, check for outages in your area.",
            "How to reset your modem: Unplug from power for 30 seconds, then plug back in. Wait 2-3 minutes for lights to stabilize.",
            "Fiber Optic 1Gbps plan includes unlimited data and speeds up to 1000 Mbps download.",
            "Our billing cycle runs from the 1st to the 30th of each month. Bills are generated on the 5th.",
            "For cable TV pixelation, try restarting your set-top box. If issue persists, check cable connections."
        ]
        self.knowledge_base_db.add_texts(docs)
        print("Knowledge base database populated.")

# --- Custom Tools ---

class QueryCustomerHistoryTool(BaseTool):
    name: str = "query_customer_history"
    description: str = "Retrieves relevant past interactions, issues, or account details for a given customer ID or general query."
    memory_manager: MemoryManager
    llm: ChatOpenAI

    def _run(self, query: str, customer_id: str = None) -> str:
        if customer_id:
            # Filter by customer ID if provided
            results = self.memory_manager.customer_history_db.similarity_search(query, k=5, filter={"customer_id": customer_id})
        else:
            results = self.memory_manager.customer_history_db.similarity_search(query, k=5)
        
        if not results:
            return "No relevant customer history found."
        
        # Summarize results before returning to save context window space
        history_text = "\n".join([doc.page_content for doc in results])
        summary = self.summarize_text(history_text)
        return f"Customer History Summary:\n{summary}"

    async def _arun(self, query: str, customer_id: str = None) -> str:
        raise NotImplementedError("Async not implemented for QueryCustomerHistoryTool")

    def summarize_text(self, text: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that summarizes customer interaction history concisely."),
            ("user", "Please summarize the following customer history:\n{history}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"history": text})
        return response.content


class QueryKnowledgeBaseTool(BaseTool):
    name: str = "query_knowledge_base"
    description: str = "Retrieves information from the company's knowledge base, including product details, troubleshooting guides, and FAQs."
    memory_manager: MemoryManager
    llm: ChatOpenAI

    def _run(self, query: str) -> str:
        results = self.memory_manager.knowledge_base_db.similarity_search(query, k=3)
        if not results:
            return "No relevant information found in the knowledge base."
        
        # Summarize results before returning if they are too long
        kb_text = "\n".join([doc.page_content for doc in results])
        summary = self.summarize_text(kb_text)
        return f"Knowledge Base Information:\n{summary}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented for QueryKnowledgeBaseTool")
    
    def summarize_text(self, text: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that summarizes knowledge base articles concisely."),
            ("user", "Please summarize the following knowledge base content:\n{content}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"content": text})
        return response.content


# --- Customer Support Agent --- 

class CustomerSupportAgent:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
        self.memory_manager = MemoryManager()
        
        # Short-term memory
        self.short_term_memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

        # Tools
        self.tools = [
            QueryCustomerHistoryTool(memory_manager=self.memory_manager, llm=self.llm),
            QueryKnowledgeBaseTool(memory_manager=self.memory_manager, llm=self.llm),
        ]

        # Agent Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI customer support agent for a telecommunications company. Your goal is to assist customers efficiently by leveraging both short-term conversational context and long-term knowledge and history. Always try to be polite and resolve the customer's issue."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Agent Executor
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, memory=self.short_term_memory, verbose=True, handle_parsing_errors=True)

    def run(self, user_input: str) -> str:
        response = self.agent_executor.invoke({"input": user_input})
        return response["output"]
    
    def get_chat_history(self):
        return self.short_term_memory.load_memory_variables({})


# --- Main Interaction Loop ---

if __name__ == "__main__":
    print("Initializing Customer Support Agent...")
    agent = CustomerSupportAgent(openai_api_key=os.environ["OPENAI_API_KEY"])
    print("Agent initialized. Type 'exit' to quit.")

    # Simulate an initial greeting based on customer history (optional)
    # For a real application, you'd get the customer ID from the session.
    # Let's assume customer ID 1001 for demonstration.
    print("\n--- Initializing for Customer ID: 1001 ---")
    initial_history_query = agent.tools[0]._run(query="initial greeting context", customer_id="1001")
    print(f"Retrieved initial history: {initial_history_query}")
    agent.short_term_memory.save_context({"input": "Customer ID 1001 joined the chat."}, {"output": f"Retrieved customer context: {initial_history_query}"})
    print("Agent: Hello! How can I assist you today?")

    while True:
        user_message = input("You: ")
        if user_message.lower() == 'exit':
            print("Exiting chat. Goodbye!")
            # In a real app, summarize and save short_term_memory to long-term history here.
            break
        
        try:
            agent_response = agent.run(user_message)
            print(f"Agent: {agent_response}")
        except Exception as e:
            print(f"Agent encountered an error: {e}")
            print("Agent: I apologize, but I'm having trouble processing your request right now. Could you please rephrase or try again later?")

