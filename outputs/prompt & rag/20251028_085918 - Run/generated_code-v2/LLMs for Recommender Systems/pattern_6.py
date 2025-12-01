import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
import uuid

load_dotenv()

PRODUCT_CATALOG = {
    "laptop": {"name": "Super Fast Laptop", "price": 1200, "category": "Electronics", "description": "Powerful laptop for work and gaming."},
    "headphone": {"name": "Noise Cancelling Headphones", "price": 250, "category": "Audio", "description": "Immersive sound with active noise cancellation."},
    "smartwatch": {"name": "Fitness Smartwatch", "price": 180, "category": "Wearables", "description": "Track your health and receive notifications."},
    "keyboard": {"name": "Mechanical Keyboard", "price": 100, "category": "Electronics", "description": "Tactile and responsive typing experience."},
    "mouse": {"name": "Gaming Mouse", "price": 70, "category": "Electronics", "description": "Precise tracking and customizable buttons."}
}

class ProductSearchInput(BaseModel):
    query: str = Field(description="The product name or category to search for.")

class ProductSearchTool(BaseTool):
    name: str = "product_search"
    description: str = "Searches the product catalog for a given query."
    args_schema: type[BaseModel] = ProductSearchInput

    def _run(self, query: str) -> str:
        results = []
        query_lower = query.lower()
        for product_id, product_info in PRODUCT_CATALOG.items():
            if query_lower in product_id.lower() or \
               query_lower in product_info["name"].lower() or \
               query_lower in product_info["category"].lower():
                results.append(f"{product_info['name']} ({product_info['category']}): ${product_info['price']:.2f}. {product_info['description']}")
        if results:
            return "Found products: " + "; ".join(results)
        return "No products found matching your query."

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Asynchronous product search not implemented")

class GetUserProfileInput(BaseModel):
    user_id: str = Field(description="The unique identifier for the user whose profile facts are to be retrieved.")
    query: str = Field(description="A query or keyword to retrieve relevant facts about the user.")

class GetUserProfileTool(BaseTool):
    name: str = "get_user_profile"
    description: str = "Retrieves relevant facts and preferences about a specific user from their profile."
    args_schema: type[BaseModel] = GetUserProfileInput
    vectorstore: Chroma = None
    embeddings: SentenceTransformerEmbeddings = None

    def _run(self, user_id: str, query: str) -> str:
        if not self.vectorstore or not self.embeddings:
            return "User profile memory is not initialized."

        docs = self.vectorstore.similarity_search(query, k=3)
        if docs:
            facts = [doc.page_content for doc in docs]
            return f"Relevant user profile facts: {'; '.join(facts)}"
        return "No specific user profile facts found matching the query."

    async def _arun(self, user_id: str, query: str) -> str:
        raise NotImplementedError("Asynchronous user profile retrieval not implemented")


class StoreUserProfileFactInput(BaseModel):
    user_id: str = Field(description="The unique identifier for the user to whom the fact belongs.")
    fact: str = Field(description="A new fact or preference about the user to store in their profile.")

class StoreUserProfileFactTool(BaseTool):
    name: str = "store_user_profile_fact"
    description: str = "Stores a new fact or preference about a user in their profile for long-term memory."
    args_schema: type[BaseModel] = StoreUserProfileFactInput
    vectorstore: Chroma = None
    embeddings: SentenceTransformerEmbeddings = None

    def _run(self, user_id: str, fact: str) -> str:
        if not self.vectorstore or not self.embeddings:
            return "User profile memory is not initialized."
        
        try:
            self.vectorstore.add_texts(
                texts=[f"User {user_id} preference/fact: {fact}"],
                metadatas=[{"user_id": user_id, "type": "preference"}]
            )
            self.vectorstore.persist()
            return f"Successfully stored fact for user {user_id}: '{fact}'"
        except Exception as e:
            return f"Error storing fact: {e}"

    async def _arun(self, user_id: str, fact: str) -> str:
        raise NotImplementedError("Asynchronous fact storage not implemented")

def main():
    st.title("AI-Powered Personalized Shopping Assistant")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.user_id = "default_user"
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you find something today?"}]

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    chroma_client = Chroma(
        collection_name="user_profiles",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    product_search_tool = ProductSearchTool()
    get_user_profile_tool = GetUserProfileTool(vectorstore=chroma_client, embeddings=embeddings)
    store_user_profile_fact_tool = StoreUserProfileFactTool(vectorstore=chroma_client, embeddings=embeddings)

    tools = [product_search_tool, get_user_profile_tool, store_user_profile_fact_tool]

    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500, memory_key="chat_history", return_messages=True)

    prompt_template = PromptTemplate.from_template("""
    You are an AI-powered personalized shopping assistant. You help users find products, answer questions, and remember their preferences.
    The current user's ID is: {user_id}.
    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action (if a tool requires 'user_id', make sure to include the current user's ID from the context above)
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Previous conversation history:
    {chat_history}

    User: {input}
    Thought:{agent_scratchpad}
    """)

    agent = create_react_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What are you looking for?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = agent_executor.invoke({"input": prompt, "user_id": st.session_state.user_id})
                    assistant_response = response["output"]
                except Exception as e:
                    assistant_response = f"An error occurred: {e}"
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    main()