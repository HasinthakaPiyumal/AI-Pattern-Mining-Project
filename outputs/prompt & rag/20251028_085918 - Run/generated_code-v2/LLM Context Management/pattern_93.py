import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document


load_dotenv()

class ECommerceChatbot:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.short_term_memory = ConversationBufferWindowMemory(
            memory_key="chat_history", return_messages=True, k=5
        )
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize ChromaDB with a persistent client
        self.long_term_memory = Chroma(
            collection_name="ecommerce_memory", 
            embedding_function=self.embedding_function,
            persist_directory="./chroma_db"
        )
        self._initialize_long_term_memory()

        self.qa_chain = self._create_qa_chain()

    def _initialize_long_term_memory(self):
        # Add some dummy data if the collection is empty
        if self.long_term_memory._collection.count() == 0:
            print("Ingesting initial data into ChromaDB...")
            customer_interactions = [
                "Customer John Doe had an issue with a defective 'Wireless Headphones' order #12345 on 2023-10-26. Resolution: Replacement sent.",
                "Customer Jane Smith asked about the return policy for 'Smartwatch X'. Provided link to returns page.",
                "Customer Michael Brown inquired about the shipping status of order #67890 (Laptop Pro). It's scheduled for delivery tomorrow.",
                "Customer John Doe, email john.doe@example.com, prefers express shipping and has previously purchased 'Gaming Mouse Z'.",
            ]
            product_knowledge = [
                "The 'Wireless Headphones' feature active noise cancellation and 20 hours of battery life.",
                "The return policy allows returns within 30 days of purchase for a full refund, provided the item is in its original condition.",
                "'Smartwatch X' is waterproof up to 50 meters and includes a heart rate monitor.",
                "'Gaming Mouse Z' has customizable RGB lighting and a DPI range of 800-16000.",
                "Shipping takes 3-5 business days for standard delivery and 1-2 business days for express delivery."
            ]
            
            docs = []
            for text in customer_interactions + product_knowledge:
                docs.append(Document(page_content=text))
            
            self.long_term_memory.add_documents(docs)
            print("Initial data ingestion complete.")

    def _create_qa_chain(self):
        _template = """You are an AI customer support assistant for an e-commerce platform. Your goal is to provide helpful, accurate, and personalized responses to customer queries.
        Use the following pieces of retrieved context and your short-term conversational history to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        Chat History: {chat_history}
        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=_template, input_variables=["context", "chat_history", "question"]
        )

        return ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.long_term_memory.as_retriever(),
            memory=self.short_term_memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            verbose=False # Set to True for debugging chain execution
        )

    def chat(self, user_query: str) -> str:
        # The ConversationalRetrievalChain automatically handles:
        # 1. Retrieving relevant documents from long-term memory (ChromaDB).
        # 2. Incorporating short-term conversational memory.
        # 3. Passing both to the LLM for response generation.
        result = self.qa_chain({"question": user_query})
        return result["answer"]

if __name__ == "__main__":
    print("Initializing E-commerce Chatbot...")
    chatbot = ECommerceChatbot()
    print("Chatbot initialized. Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        response = chatbot.chat(user_input)
        print(f"Chatbot: {response}")
