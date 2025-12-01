import os
from dotenv import load_dotenv
import gradio as gr

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(temperature=0.7, model="gpt-4o")

model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500, return_messages=True, memory_key="chat_history")

def add_to_vectorstore(text, metadata=None):
    if metadata is None:
        metadata = {}
    vectorstore.add_texts([text], metadatas=[metadata])

def get_chat_response(message, history):
    memory.clear()
    for human_msg, ai_msg in history:
        memory.save_context({"input": human_msg}, {"output": ai_msg})
        add_to_vectorstore(human_msg, {"type": "user_query"})
        add_to_vectorstore(ai_msg, {"type": "ai_response"})

    qa_prompt_with_retrieval = ChatPromptTemplate.from_messages(
        [
            ("system", (
                "You are an intelligent customer support assistant. "
                "Your goal is to provide helpful and accurate information to the user. "
                "Keep your responses concise and to the point. "
                "If the user asks something you don't know, admit it but try to guide them to relevant resources. "
                "Here is some relevant context retrieved from past interactions and knowledge base:\n{context}\n\n"
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever_runnable = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(
            context=retriever_runnable | format_docs
        )
        | qa_prompt_with_retrieval
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke({
        "input": message,
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })

    memory.save_context({"input": message}, {"output": response})
    add_to_vectorstore(message, {"type": "user_query"})
    add_to_vectorstore(response, {"type": "ai_response"})

    return response

iface = gr.ChatInterface(
    get_chat_response,
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(placeholder="Ask me a question...", container=False, scale=7),
    title="Customer Support Chatbot with Context Management",
    description="An AI chatbot that can handle long conversations by managing its context window efficiently.",
    theme="soft",
    examples=["What are your return policies?", "I have a problem with my order #12345.", "Can you help me reset my password?", "Tell me about your new products."],
)

if __name__ == '__main__':
    iface.launch()