import os
import gradio as gr
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Data Ingestion & Indexing

# Create mock knowledge sources
product_db_content = """
Product A: High-end smartphone with 128GB storage, 6.1-inch OLED display, and a dual-camera system. Price: $999.
Product B: Mid-range laptop with 8GB RAM, 256GB SSD, 13.3-inch Full HD display. Price: $799.
Product C: Smartwatch with heart rate monitor, GPS, and water resistance up to 50m. Price: $249.
"""

faq_content = """
Q: How to reset my password?
A: Go to the login page and click 'Forgot Password'. Follow the instructions.
Q: What is your return policy?
A: We offer a 30-day no-questions-asked return policy for all products.
Q: How do I track my order?
A: You can track your order using the tracking number provided in your shipping confirmation email on our website.
"""

user_manual_content = """
Smartwatch Manual: 
1. To turn on, press and hold the side button for 3 seconds.
2. To pair with phone, open the companion app and follow the on-screen instructions.
3. Charging: Use the magnetic charger included in the box. Battery life up to 7 days.
"""

with open("product_db.txt", "w") as f: f.write(product_db_content)
with open("faq.txt", "w") as f: f.write(faq_content)
with open("user_manual.txt", "w") as f: f.write(user_manual_content)

# Load documents
loaders = [
    TextLoader("product_db.txt"),
    TextLoader("faq.txt"),
    TextLoader("user_manual.txt")
]
documents = []
for loader in loaders:
    documents.extend(loader.load())

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Embedding Model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector Store (Chroma)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
vectorstore.persist()

# 2. Retrieval Module
retriever = vectorstore.as_retriever()

# 3. Generation Module
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

# Prompt Template
# The prompt is simplified for direct use with RetrievalQA chain for brevity.
# For more complex LCEL, a ChatPromptTemplate would be defined explicitly.
# However, RetrievalQA implicitly handles context insertion.

# 4. Orchestration & Chatbot Logic
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def chatbot_response(message, history):
    result = qa_chain({"query": message})
    return result["result"]

# Gradio Interface
if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chatbot_response,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Ask me anything about products, orders, or policies!", container=False, scale=7),
        title="E-commerce Support Chatbot (RAG)",
        description="I can answer your questions by retrieving information from our product database, FAQs, and user manuals.",
        theme="soft",
        examples=["Tell me about Product A", "How do I track my order?", "How to turn on my smartwatch?"],
        cache_examples=True,
    )

    print("Starting Gradio interface...")
    demo.launch(share=False)

    # Clean up mock files after use (optional, for demonstration purposes)
    # import os
    # os.remove("product_db.txt")
    # os.remove("faq.txt")
    # os.remove("user_manual.txt")
