"""
This script implements an AI-Powered Clinical Decision Support System with Real-time Medical Knowledge Augmentation.
It leverages a simulated Large Language Model (LLM), a Retrieval Augmented Generation (RAG) system using ChromaDB 
and Sentence-Transformers, and simulated external tools for medical database lookups, all orchestrated by a LangChain agent.

Key Components:
- MockLLM: A simulated LLM for local demonstration without external API keys.
- search_drug_info: A simulated external tool for retrieving drug information.
- RAG System: Utilizes ChromaDB for vector storage and SentenceTransformer for embeddings of medical documents.
- LangChain Agent: Orchestrates the LLM, RAG system, and external tools to answer clinician queries.

Workflow:
1. Clinician inputs a natural language query.
2. LangChain Agent processes the query.
3. Agent decides whether to use RAG, an external tool, or a combination.
4. LLM synthesizes information and generates a concise, evidence-based response.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field

# LangChain specific imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool as langchain_tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.language_models import BaseChatModel

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1. Simulated LLM Backbone --- 
class MockLLM(BaseChatModel):
    """A simple mock LLM for demonstration purposes."""
    def _generate(self, messages: List[HumanMessage], **kwargs: Any) -> Any:
        # Simple logic to simulate LLM response based on messages
        # In a real scenario, this would call an actual LLM API
        # For agent execution, it needs to parse tool invocations.
        last_message_content = messages[-1].content if messages else ""

        if "tool_code" in last_message_content: # Heuristic to detect tool output from agent
            # If it looks like tool output, just acknowledge or pass it through
            return type('MockResponse', (object,), {'content': f"Okay, I've processed the tool output."})

        if "search_drug_info" in last_message_content and "(" in last_message_content:
            # Simulate tool invocation for the agent. The agent should parse this.
            # For this mock, we assume the agent will correctly parse it.
            pass # Agent will handle tool invocation based on its parsing logic

        # Simple direct response if no tool invocation is detected
        if "side effects of Metformin" in last_message_content:
            return type('MockResponse', (object,), {'content': "Metformin's common side effects include nausea, diarrhea, and abdominal discomfort. Rarely, lactic acidosis can occur. It's important to consult with a healthcare professional."})
        elif "latest guidelines for treating type 2 diabetes" in last_message_content:
            return type('MockResponse', (object,), {'content': "The latest guidelines for type 2 diabetes often emphasize personalized care, early intensive lifestyle interventions, and consideration of newer agents with cardiovascular and renal benefits. retrieved from RAG. Please consult full guidelines for details."})
        elif "drug interactions for Aspirin" in last_message_content:
            return type('MockResponse', (object,), {'content': "Aspirin can interact with anticoagulants like Warfarin, increasing bleeding risk. It may also interact with NSAIDs, certain blood pressure medications, and antacids. Always check with a pharmacist or physician for specific interactions."})
        elif "what is insulin" in last_message_content:
            return type('MockResponse', (object,), {'content': "Insulin is a hormone produced by the pancreas that helps regulate blood sugar. It allows glucose to enter cells for energy. In diabetes, insulin production is impaired or its action is ineffective."})
        else:
            return type('MockResponse', (object,), {'content': f"I understand you're asking about '{last_message_content}'. I will use my tools or knowledge base to find more information."})

    @property
    def _llm_type(self) -> str:
        return "mock_llm"
    
    def invoke(self, input: List[HumanMessage], **kwargs: Any) -> Any:
        return self._generate(input, **kwargs)
    
    async def _acall(self, messages: List[HumanMessage], **kwargs: Any) -> Any:
        # For async compatibility, though this mock is synchronous
        return self._generate(messages, **kwargs)


# --- 2. Simulated Medical Database Tool --- 
class DrugInfoInput(BaseModel):
    drug_name: str = Field(description="Name of the drug to search for.")

@langchain_tool("search_drug_info", args_schema=DrugInfoInput)
def search_drug_info(drug_name: str) -> str:
    """Searches a simulated medical database for information about a given drug."""
    drug_data = {
        "metformin": {"dosage": "500-2550 mg daily", "class": "Biguanide", "uses": "Type 2 Diabetes", "mechanism": "Decreases glucose production by the liver", "side_effects": ["Nausea", "Diarrhea", "Abdominal discomfort", "Lactic acidosis (rare)"]},
        "aspirin": {"dosage": "81-325 mg daily", "class": "NSAID, Antiplatelet", "uses": "Pain relief, Fever reduction, Cardiovascular prophylaxis", "mechanism": "Inhibits prostaglandin synthesis and platelet aggregation", "side_effects": ["Gastric upset", "Bleeding", "Reye's syndrome (in children)"]},
        "lisinopril": {"dosage": "5-40 mg daily", "class": "ACE Inhibitor", "uses": "Hypertension, Heart Failure", "mechanism": "Blocks the conversion of angiotensin I to angiotensin II", "side_effects": ["Cough", "Dizziness", "Fatigue", "Hyperkalemia"]},
        "insulin": {"dosage": "Variable", "class": "Hormone", "uses": "Type 1 and Type 2 Diabetes", "mechanism": "Facilitates glucose uptake by cells", "side_effects": ["Hypoglycemia", "Weight gain", "Injection site reactions"]},
    }
    
    normalized_drug_name = drug_name.lower()
    if normalized_drug_name in drug_data:
        info = drug_data[normalized_drug_name]
        return f"Information for {drug_name.capitalize()}: Class - {info['class']}, Uses - {info['uses']}, Side Effects - {', '.join(info['side_effects'])}."
    else:
        return f"No detailed information found for {drug_name} in the simulated database."


# --- 3. RAG System Setup --- 
def setup_rag_system():
    """Sets up the RAG system with ChromaDB and Sentence-Transformers."""
    # Sample medical documents
    medical_docs = [
        "Type 2 diabetes is a chronic condition that affects the way the body processes blood sugar (glucose). The body either doesn't produce enough insulin, or it resists the effects of insulin. Treatment often involves lifestyle changes, oral medications, and sometimes insulin injections.",
        "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Lifestyle modifications like diet and exercise are crucial, alongside medications like ACE inhibitors, ARBs, diuretics, and beta-blockers.",
        "Aspirin (acetylsalicylic acid) is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce pain, fever, and inflammation. It is also used in low doses to prevent heart attacks and strokes in people at high risk. Common side effects include gastrointestinal irritation and an increased risk of bleeding.",
        "Metformin is a first-line medication for type 2 diabetes, particularly in people who are overweight. It works by decreasing glucose production in the liver and improving insulin sensitivity. It can cause gastrointestinal side effects.",
        "Insulin therapy is essential for people with type 1 diabetes and often necessary for those with type 2 diabetes whose bodies do not produce enough insulin or do not use insulin effectively. Various types of insulin exist, categorized by their onset, peak, and duration of action.",
        "The American Diabetes Association (ADA) updates its Standards of Medical Care in Diabetes annually. These guidelines cover diagnosis, treatment goals, and management strategies, including pharmacotherapy and lifestyle interventions.",
        "Cardiovascular disease (CVD) is a general term for conditions affecting the heart or blood vessels. It is often associated with a build-up of fatty deposits in the arteries (atherosclerosis) and an increased risk of blood clots. Managing risk factors like hypertension, high cholesterol, and diabetes is key.",
    ]

    # Initialize embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize ChromaDB and add documents
    print("\nInitializing ChromaDB and ingesting documents...")
    vectorstore = Chroma.from_texts(medical_docs, embeddings, collection_name="medical_knowledge")
    print(f"Ingested {len(medical_docs)} documents into ChromaDB.")

    # Create a retriever
    retriever = vectorstore.as_retriever()
    return retriever


# --- 4. Orchestration Layer (LangChain Agent) --- 
def setup_agent(llm: BaseChatModel, retriever: Any, tools: List[Tool]):
    """Sets up the LangChain agent for orchestrating LLM, RAG, and tools."""
    # Define the agent's system prompt
    system_prompt = (
        "You are a highly intelligent and helpful AI assistant for clinical decision support. "
        "You have access to a RAG system containing medical documents and a tool to look up drug information. "
        "Your goal is to provide accurate, evidence-based responses to clinician queries. "
        "Prioritize using your tools to gather factual information when appropriate, then synthesize a concise answer. "
        "If a query can be answered directly from your internal knowledge or general medical understanding, do so. "
        "Always be cautious and advise consulting a healthcare professional for definitive medical advice."
    )

    # Create the agent prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            ("placeholder", "{chat_history}"), # For future chat history integration
            HumanMessage(content="{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Add the RAG functionality as a tool
    rag_tool = Tool(
        name="medical_document_search",
        func=lambda query: "\n".join([doc.page_content for doc in retriever.invoke(query)]),
        description="Useful for searching general medical documents and guidelines related to diseases, treatments, or conditions."
    )
    tools.append(rag_tool)

    # Create the React Agent
    print("\nSetting up LangChain Agent...")
    agent = create_react_agent(llm, tools, prompt)

    # Create the Agent Executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    print("LangChain Agent setup complete.")
    return agent_executor


# --- Main Execution --- 
if __name__ == "__main__":
    # Initialize simulated LLM
    llm = MockLLM()

    # Setup RAG system
    retriever = setup_rag_system()

    # Define tools
    available_tools = [
        search_drug_info,
    ]

    # Setup LangChain Agent
    agent_executor = setup_agent(llm, retriever, available_tools)

    print("\n--- AI-Powered Clinical Decision Support System Ready ---")
    print("Enter your medical queries below. Type 'exit' to quit.\n")

    while True:
        query = input("Clinician Query: ")
        if query.lower() == 'exit':
            print("Exiting Clinical Decision Support System. Goodbye!")
            break

        try:
            print("\nProcessing query...")
            # The agent executor expects a dictionary with 'input'
            response = agent_executor.invoke({"input": query, "chat_history": []})
            print(f"\nSystem Response: {response['output']}\n")
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")
            print("Please try rephrasing your query or contact support.")

