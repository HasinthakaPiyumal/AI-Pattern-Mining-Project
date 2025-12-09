import os
from collections import deque
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

load_dotenv()

class ShortTermMemoryManager:
    def __init__(self, max_interactions=5):
        self.memory = deque(maxlen=max_interactions)

    def add_interaction(self, role, content):
        self.memory.append({"role": role, "content": content})

    def get_history(self):
        return list(self.memory)

    def clear(self):
        self.memory.clear()

class LongTermMemoryManager:
    def __init__(self, persist_directory="chroma_db", collection_name="customer_support_memory", embeddings=None):
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.vectorstore = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings, collection_name=collection_name)

    def add_documents(self, documents: list[Document]):
        self.vectorstore.add_documents(documents)

    def retrieve_relevant_info(self, query: str, k: int = 3) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def summarize_memory(self, text: str, llm=None) -> str:
        if not llm:
            return f"[SUMMARY OF: {text[:100]}...]"
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are a helpful assistant that summarizes text concisely."),
            HumanMessagePromptTemplate.from_template("Please summarize the following text: {text}")
        ])
        chain = prompt | llm
        response = chain.invoke({"text": text})
        return response.content

class CustomerSupportAgent:
    def __init__(self, llm, short_term_memory_manager, long_term_memory_manager):
        self.llm = llm
        self.short_term_memory = short_term_memory_manager
        self.long_term_memory = long_term_memory_manager

    def handle_inquiry(self, user_query: str):
        self.short_term_memory.add_interaction("user", user_query)
        
        conversation_history_str = "\n".join([f"{d['role'].capitalize()}: {d['content']}" for d in self.short_term_memory.get_history()])
        
        combined_query_for_long_term = f"Current conversation:\n{conversation_history_str}\nUser's last question: {user_query}"
        
        relevant_long_term_info = self.long_term_memory.retrieve_relevant_info(combined_query_for_long_term)
        long_term_context_str = "\n".join([doc.page_content for doc in relevant_long_term_info])

        prompt_messages = [
            SystemMessagePromptTemplate.from_template(
                "You are a helpful and knowledgeable customer support agent. "
                "Use the provided context and conversation history to answer the user's questions accurately and assist them. "
                "If you don't know the answer, state that you cannot provide it based on the available information. "
                "Keep your responses concise and to the point."
            ),
            SystemMessagePromptTemplate.from_template(f"Long-Term Context:\n{long_term_context_str}"),
            SystemMessagePromptTemplate.from_template(f"Conversation History:\n{conversation_history_str}"),
            HumanMessagePromptTemplate.from_template(user_query)
        ]
        
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        
        chain = prompt | self.llm
        response = chain.invoke({"input": user_query})
        agent_response = response.content

        self.short_term_memory.add_interaction("agent", agent_response)
        return agent_response

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

    # Initialize Memory Managers
    short_term_memory = ShortTermMemoryManager(max_interactions=5)
    long_term_memory = LongTermMemoryManager(embeddings=embeddings)

    # Populate Long-Term Memory with sample data
    sample_faqs = [
        Document(page_content="Our return policy allows returns within 30 days of purchase with a valid receipt.", metadata={"source": "FAQ", "topic": "returns"}),
        Document(page_content="Shipping usually takes 3-5 business days for standard delivery within the country.", metadata={"source": "FAQ", "topic": "shipping"}),
        Document(page_content="You can track your order using the tracking number provided in your shipping confirmation email.", metadata={"source": "FAQ", "topic": "tracking"}),
    ]
    sample_customer_history = [
        Document(page_content="Customer John Doe purchased product A on Jan 1, 2023. Order ID: 12345.", metadata={"customer_id": "john_doe", "type": "purchase_history"}),
        Document(page_content="Customer John Doe inquired about product A's features on Jan 5, 2023. Issue resolved.", metadata={"customer_id": "john_doe", "type": "support_ticket"}),
    ]
    long_term_memory.add_documents(sample_faqs + sample_customer_history)

    # Initialize Agent
    agent = CustomerSupportAgent(llm, short_term_memory, long_term_memory)

    print("\n--- Starting Customer Support Chat ---")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        response = agent.handle_inquiry(user_input)
        print(f"Agent: {response}")

    print("\n--- Chat Ended ---")