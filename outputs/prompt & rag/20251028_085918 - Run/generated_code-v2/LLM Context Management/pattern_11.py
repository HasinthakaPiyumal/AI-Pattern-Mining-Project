from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage

import os

# Ensure you have your OpenAI API key set as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

class EcommerceSupportAssistant:
    def __init__(self, openai_api_key=None):
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found. Please set it as an environment variable or pass it to the constructor.")

        self.llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
        self.embeddings = OpenAIEmbeddings()

        # 1. Simulate E-commerce Data Sources
        self.product_info = [
            "Product A: High-quality headphones with noise cancellation, 20-hour battery life. Price: $199.",
            "Product B: Ergonomic office chair, adjustable lumbar support, breathable mesh. Price: $299.",
            "Product C: Smartwatch with heart rate monitor, GPS, water-resistant. Price: $149.",
            "Product D: Portable Bluetooth speaker, 10W output, 12-hour playtime. Price: $79."
        ]
        self.faq_knowledge_base = [
            "FAQ: Shipping usually takes 3-5 business days for standard delivery.",
            "FAQ: Returns are accepted within 30 days of purchase, provided the item is in original condition.",
            "FAQ: To reset your password, click on the 'Forgot Password' link on the login page.",
            "FAQ: Our customer support hours are Monday to Friday, 9 AM to 5 PM EST.",
            "FAQ: We offer a one-year warranty on all electronic products."
        ]
        self.order_history_data = [
            "Order #12345: Placed on 2023-10-26, Product A. Status: Shipped.",
            "Order #67890: Placed on 2023-11-01, Product C. Status: Processing."
        ]

        all_long_term_data = self.product_info + self.faq_knowledge_base + self.order_history_data

        # 2. Long-Term Memory (RAG System)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = self.text_splitter.create_documents(all_long_term_data)
        self.vectorstore = Chroma.from_documents(docs, self.embeddings, collection_name="ecommerce_knowledge")
        self.retriever = self.vectorstore.as_retriever()

        # Initialize RAG chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )

        # 3. Short-Term Memory Module
        self.short_term_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, human_prefix="Customer", ai_prefix="Assistant")

        # 4. Memory Summarization Module
        # Max tokens for the buffer, after which it summarizes to stay within context limits
        self.summary_memory = ConversationSummaryBufferMemory(
            llm=self.llm, max_token_limit=500, memory_key="chat_history", return_messages=True, human_prefix="Customer", ai_prefix="Assistant"
        )

        # Combine short-term memory (for recent turns) and long-term memory (RAG) using ConversationalRetrievalChain
        # The chat_history from summary_memory will be used to pass relevant context to the QA chain.
        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.summary_memory, # Use summary memory for the conversational chain
            combine_docs_chain_kwargs={"prompt": "Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Also, use the chat history to maintain conversation flow.\n\n{context}\n\nChat History:\n{chat_history}\nHuman: {question}\nAssistant:"}
        )

    def process_query(self, query: str) -> str:
        # First, add the user's query to the short-term memory (for direct context access if needed by the prompt)
        # Note: ConversationalRetrievalChain manages its own memory, but we can update short_term_memory for custom logic
        # For this example, ConversationalRetrievalChain's memory (summary_memory) will handle the history directly.

        response = self.conversational_chain.run(query)

        # Manual update for demonstration, though summary_memory is already updated by the chain
        # self.short_term_memory.chat_memory.add_user_message(query)
        # self.short_term_memory.chat_memory.add_ai_message(response)

        print(f"\n--- Short-Term Memory (Current Buffer) ---\n{self.summary_memory.load_memory_variables({})} ")
        # print(f"\n--- Long-Term Memory Retrieval ---\n{self.qa_chain({"query": query})["source_documents"]}") # For debugging retrieved docs

        return response


# --- Example Usage ---
if __name__ == "__main__":
    # Replace with your actual OpenAI API key or ensure it's set as an environment variable
    # assistant = EcommerceSupportAssistant(openai_api_key="YOUR_OPENAI_API_KEY")
    assistant = EcommerceSupportAssistant() 

    print("E-commerce Customer Support Assistant (Type 'exit' to quit)")
    print("Hello! How can I assist you today?")

    while True:
        user_input = input("Customer: ")
        if user_input.lower() == 'exit':
            break

        assistant_response = assistant.process_query(user_input)
        print(f"Assistant: {assistant_response}")

        # Simulate a long conversation to trigger summarization
        # For demonstration, you might need to input many long sentences to reach max_token_limit
        # You can manually check the summary_memory content during a long interaction.
