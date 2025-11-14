import os
import shutil
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# Define the directory for the Chroma DB
CHROMA_DB_DIR = "./chroma_db"

def setup_knowledge_base():
    """
    Sets up a Chroma vector store with sample e-commerce customer support documents.
    """
    # Clear existing Chroma DB for a fresh start
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)

    # Sample E-commerce documents for customer support
    documents_content = [
        ("return_policy.txt", "Our return policy allows returns within 30 days of purchase for a full refund. Items must be in their original condition with all tags attached. Electronics have a 15-day return window. Customized items are non-returnable."),
        ("shipping_info.txt", "Standard shipping takes 5-7 business days. Express shipping takes 1-2 business days. Shipping costs vary based on location and order size. Free standard shipping on orders over $50."),
        ("payment_methods.txt", "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay. We do not accept cash on delivery or personal checks."),
        ("product_warranty.txt", "Most electronics come with a 1-year manufacturer's warranty. Please refer to the product page for specific warranty details. Extended warranties are available for purchase."),
        ("account_creation.txt", "To create an account, click 'Sign Up' at the top right corner of the page and follow the prompts. You can also sign up during checkout."),
        ("order_tracking.txt", "You can track your order by logging into your account and visiting the 'My Orders' section. A tracking link will be provided once your order has shipped."),
        ("damaged_item_policy.txt", "If you receive a damaged item, please contact customer support within 48 hours of delivery with photos of the damage. We will arrange for a replacement or refund."),
        ("international_shipping.txt", "We offer international shipping to select countries. Please check our shipping page for a list of supported regions and associated costs and delivery times."),
        ("gift_card_use.txt", "Gift cards can be redeemed at checkout by entering the gift card code. They cannot be exchanged for cash and have no expiration date."),
        ("product_availability.txt", "Product availability is displayed on each product page. If an item is out of stock, you can sign up for email notifications to be alerted when it's back."),
        ("privacy_policy.txt", "Our privacy policy details how we collect, use, and protect your personal information. We are committed to safeguarding your data. For more information, please visit our privacy policy page."),
        ("customer_service_hours.txt", "Our customer service team is available Monday to Friday, 9 AM to 6 PM EST. You can reach us via live chat, email, or phone."),
        ("loyalty_program.txt", "Join our loyalty program to earn points on every purchase, redeemable for discounts and exclusive offers. Membership is free."),
        ("technical_support.txt", "For technical issues with our website or app, please contact our technical support team. Provide details of the issue, including screenshots if possible."),
        ("bulk_orders.txt", "For bulk or wholesale orders, please contact our sales team directly for special pricing and arrangements."),
    ]

    # Create Document objects
    documents = [Document(page_content=content, metadata={"source": filename}) 
                 for filename, content in documents_content]

    # Initialize embeddings model
    # Using a common sentence transformer model suitable for general text embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    print(f"Knowledge base setup complete. {len(documents)} documents added to Chroma DB.")
    return vectorstore

if __name__ == "__main__":
    vector_store = setup_knowledge_base()
    print("\nTesting retrieval from the knowledge base:")
    query = "What is the return policy for electronics?"
    retrieved_docs = vector_store.similarity_search(query, k=2)
    print(f"Query: {query}")
    for i, doc in enumerate(retrieved_docs):
        print(f"\nRetrieved Document {i+1} (Source: {doc.metadata['source']}):")
        print(doc.page_content[:200] + "...") # Print first 200 chars

    query_shipping = "How long does standard shipping take?"
    retrieved_docs_shipping = vector_store.similarity_search(query_shipping, k=1)
    print(f"\nQuery: {query_shipping}")
    for i, doc in enumerate(retrieved_docs_shipping):
        print(f"\nRetrieved Document {i+1} (Source: {doc.metadata['source']}):")
        print(doc.page_content[:200] + "...")
