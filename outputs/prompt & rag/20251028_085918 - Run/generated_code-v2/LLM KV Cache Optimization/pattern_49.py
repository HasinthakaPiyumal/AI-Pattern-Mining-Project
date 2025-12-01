import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ensure the Chroma DB directory exists
CHROMA_DB_PATH = "./chroma_db"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Sample knowledge base content
KB_CONTENT = """
Product X: This product is designed for enhanced performance and durability. It features a quad-core processor and 16GB RAM.
Product Y: A compact and portable solution with a long-lasting battery life of up to 12 hours. Ideal for travel.
Warranty Information: All products come with a 1-year limited warranty. Extended warranty options are available for purchase.
Return Policy: Items can be returned within 30 days of purchase with a valid receipt. Some exclusions apply.
Technical Support: For technical assistance, please visit our website's support section or call our helpline at 1-800-TECH-HELP.
Shipping: Standard shipping takes 3-5 business days. Expedited shipping is available at an additional cost.
Payment Methods: We accept major credit cards, PayPal, and bank transfers.
"""

# Create a dummy knowledge base file
with open("sample_kb.txt", "w") as f:
    f.write(KB_CONTENT)

# Load and split documents
with open("sample_kb.txt", "r") as f:
    raw_documents = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.create_documents([raw_documents])

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create and persist the Chroma vector store
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=CHROMA_DB_PATH)
vectorstore.persist()

print(f"Knowledge base setup complete. {len(docs)} documents embedded and stored in {CHROMA_DB_PATH}")
