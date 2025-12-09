import os
from typing import Dict, Any, List

import openai
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# --- Environment Setup ---
# Make sure to set your OPENAI_API_KEY environment variable
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# --- Memory Management --- 

class LongTermMemoryManager:
    def __init__(self, collection_name: str = "personal_assistant_memory", embedding_function=None):
        if embedding_function is None:
            embedding_function = OpenAIEmbeddings()
        self.vectorstore = Chroma(collection_name=collection_name, embedding_function=embedding_function, persist_directory="./chroma_db")
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo-preview")
        self.retriever = self.vectorstore.as_retriever()
        self.vectorstore.persist()

    def summarize_text(self, text: str) -> str:
        prompt_template = """Summarize the following text concisely, focusing on key information relevant for a personal assistant to remember for future interactions or tasks:

TEXT: {text}

CONCISE SUMMARY:"""
        summary_prompt = PromptTemplate.from_template(prompt_template)
        summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
        return summary_chain.run(text=text)

    def add_to_long_term_memory(self, content: str, metadata: Dict[str, Any] = None):
        summary = self.summarize_text(content)
        doc = Document(page_content=summary, metadata=metadata or {})
        self.vectorstore.add_documents([doc])
        self.vectorstore.persist()
        print(f"[LTM] Added summarized content to long-term memory: {summary[:50]}...")

    def retrieve_from_long_term_memory(self, query: str, k: int = 3) -> List[str]:
        docs = self.retriever.invoke(query)
        return [doc.page_content for doc in docs]

# --- Short-Term Memory (Conversation Buffer) ---
# Using ConversationBufferWindowMemory to keep recent turns in context
conversation_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5 # Keep the last 5 turns of conversation
)

# --- Initialize Long-Term Memory Manager ---
ltm_manager = LongTermMemoryManager()

# --- Define Tools ---

@tool
def NotebookWrite(information: str) -> str:
    """Use this tool to explicitly record important information or intermediate steps in your working memory.
    This information will be available in the short-term conversation context for a few turns.
    Input should be the information you want to remember."""
    # In a real scenario, this might update a scratchpad or a more persistent short-term store
    # For this implementation, we'll append it to the conversation history if it's not too repetitive
    # or just acknowledge it, relying on ConversationBufferWindowMemory for short-term recall.
    print(f"[NotebookWrite] Agent noted: {information}")
    return f"Information '{information}' has been noted for current task context."

@tool
def LongTermMemoryAdd(information: str, relevant_task: str = "general") -> str:
    """Use this tool to add important, summarized information to the long-term memory.
    This is useful for remembering facts, preferences, or task outcomes across sessions or long interactions.
    Input should be the information to add. Optionally, specify 'relevant_task' for better organization."""
    ltm_manager.add_to_long_term_memory(information, metadata={"task": relevant_task})
    return f"Information related to '{relevant_task}' has been added to long-term memory."

@tool
def LongTermMemoryRetrieve(query: str) -> str:
    """Use this tool to retrieve relevant information from the long-term memory based on a query.
    This is useful for recalling past preferences, historical data, or broader knowledge.
    Input should be the query to search long-term memory for."""
    retrieved_info = ltm_manager.retrieve_from_long_term_memory(query)
    if retrieved_info:
        return "Retrieved from long-term memory: " + "\n".join(retrieved_info)
    return "No relevant information found in long-term memory."

# --- Agent Setup ---

llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo-preview")

tools = [
    NotebookWrite,
    LongTermMemoryAdd,
    LongTermMemoryRetrieve,
]

# Define the agent's prompt
agent_prompt = PromptTemplate.from_template("""You are a Smart Personal Assistant, designed to help users manage complex multi-step tasks.
 You have access to short-term memory (your current conversation context) and long-term memory.
Use the available tools to assist the user effectively.

TOOLS:
{tools}

FORMAT INSTRUCTIONS:
{format_instructions}

CURRENT CONVERSATION HISTORY:
{chat_history}

Relevant past information from long-term memory (if any):
{long_term_context}

USER'S CURRENT REQUEST:
{input}

{agent_scratchpad}""")

# Function to get long-term context for the prompt
def get_long_term_context(query: str) -> str:
    retrieved = ltm_manager.retrieve_from_long_term_memory(query, k=1) # Retrieve a single most relevant item
    return retrieved[0] if retrieved else "None."

# Create the agent
def create_smart_assistant_agent() -> AgentExecutor:
    # A custom RAG chain for the agent to use long-term memory contextually
    class SmartAssistantRAGChain(LLMChain):
        @property
        def _run_output_key(self) -> str:
            return "long_term_context"
        
        def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            query = inputs["input"]
            context = get_long_term_context(query) # Use the helper function
            return {"long_term_context": context}

    # Create a dummy chain that returns the current input as long_term_context, to be replaced
    # This is a placeholder for dynamic long-term context injection into the prompt
    long_term_context_chain = SmartAssistantRAGChain(llm=llm, prompt=PromptTemplate.from_template("{input}"))

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=agent_prompt
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=conversation_memory,
        verbose=True,
        # You can add custom inputs to the agent if needed. 
        # Here, we'll manage long_term_context dynamically in the main loop.
        handle_parsing_errors=True # For robustness
    )
    return agent_executor

agent_executor = create_smart_assistant_agent()

# --- User Interface (CLI) ---

def main():
    print("\n--- Smart Personal Assistant ---\n")
    print("Hello! I'm your Smart Personal Assistant. How can I help you today?")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        try:
            # Dynamically add long-term context to the agent's input
            # This simulates the RAG part augmenting the main prompt
            long_term_context_for_current_query = get_long_term_context(user_input)

            # The agent's `run` method automatically passes chat_history from `conversation_memory`
            # We explicitly pass `long_term_context` as an additional input to the prompt template.
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": conversation_memory.load_memory_variables({}).get("chat_history", []), 
                "long_term_context": long_term_context_for_current_query
            })
            
            print(f"Assistant: {response['output']}")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()