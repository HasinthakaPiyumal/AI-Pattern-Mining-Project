from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import gradio as gr
import time

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

class CustomTools:
    @staticmethod
    def reset_password(user_id: str) -> str:
        time.sleep(2)
        return f"Password for user ID {user_id} has been reset successfully. A temporary password has been sent to their registered email."

    @staticmethod
    def check_order_status(order_id: str) -> str:
        time.sleep(2)
        if order_id == "ORDER123":
            return f"Order {order_id} is currently in transit and expected to be delivered by tomorrow."
        elif order_id == "ORDER456":
            return f"Order {order_id} has been delivered on 2023-10-26."
        return f"Could not find details for order ID {order_id}. Please double check the ID."

# Knowledge Base Initialization
# For simplicity, we'll create a dummy text file and load it
kb_content = """
Customer Support FAQ:
Q: How do I change my shipping address?
A: You can change your shipping address by logging into your account, going to 'My Orders', and editing the address for your pending order. If the order has already shipped, please contact support immediately.

Q: What is your return policy?
A: We offer a 30-day return policy for most items, provided they are in their original condition. Some exclusions apply. Please visit our 'Returns' page for more details.

Q: How can I contact customer service?
A: You can reach customer service via phone at 1-800-555-0199, or by email at support@example.com. Our live chat is also available 24/7 on our website.

Q: My product is damaged, what should I do?
A: Please take photos of the damaged product and its packaging and contact our support team with your order number. We will arrange for a replacement or refund.

Q: Where can I track my order?
A: You can track your order by clicking the 'Track Order' link in your shipping confirmation email or by entering your order number on our website's 'Track Order' page.
"""

with open("knowledge_base.txt", "w") as f:
    f.write(kb_content)

loader = TextLoader("knowledge_base.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vdb = Chroma.from_documents(texts, embeddings)
retriever = vdb.as_retriever()

# Define Tools
reset_password_tool = Tool(
    name="reset_password",
    func=CustomTools.reset_password,
    description="Useful for resetting a user's password given their user ID."
)

check_order_status_tool = Tool(
    name="check_order_status",
    func=CustomTools.check_order_status,
    description="Useful for checking the status of a customer's order given the order ID."
)

knowledge_base_tool = Tool(
    name="knowledge_base_qa",
    func=lambda query: retriever.get_relevant_documents(query),
    description="Useful for answering questions about products, policies, and general customer support topics from the knowledge base."
)

agent_tools = [reset_password_tool, check_order_status_tool, knowledge_base_tool]

# LLM and Memory Setup
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

# Initialize Agent
agent_executor = initialize_agent(
    agent_tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# Gradio Interface
def chat_with_agent(message, history):
    try:
        response = agent_executor.run(input=message)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    demo = gr.ChatInterface(
        chat_with_agent,
        title="Composable AI Customer Support Agent",
        description="Ask me anything about your orders, password resets, or general support questions!",
        examples=[
            "What is your return policy?",
            "Reset my password for user ID 12345.",
            "What is the status of order ORDER123?",
            "How can I contact customer service?"
        ]
    )
    demo.launch()
