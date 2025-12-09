import streamlit as st
from langdetect import detect
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
import os

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

kb_data = [
    {"id": "1", "content": "The price of the red t-shirt is $25.", "language": "en"},
    {"id": "2", "content": "Our return policy allows returns within 30 days of purchase.", "language": "en"},
    {"id": "3", "content": "We ship worldwide, and delivery usually takes 5-7 business days.", "language": "en"},
    {"id": "4", "content": "El precio de la camiseta roja es de $25.", "language": "es"},
    {"id": "5", "content": "Notre politique de retour permet les retours dans les 30 jours suivant l'achat.", "language": "fr"},
    {"id": "6", "content": "Wir versenden weltweit, und die Lieferung dauert in der Regel 5-7 Werktage.", "language": "de"}
]

embeddings_model = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

texts = [d["content"] for d in kb_data]
metadatas = [{"language": d["language"]} for d in kb_data]
vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings_model, metadatas=metadatas, collection_name="ecommerce_kb")

def get_in_context_examples():
    return [
        {"input_query": "Client (Spanish): ¿Cuál es el precio de la camiseta azul?",
         "retrieved_kb": "The price of the blue t-shirt is $30.",
         "desired_response": "El precio de la camiseta azul es de $30."},
        {"input_query": "Client (French): Je voudrais connaître votre politique de remboursement.",
         "retrieved_kb": "Our refund policy allows full refunds within 15 days for unused items.",
         "desired_response": "Notre politique de remboursement permet des remboursements complets dans les 15 jours pour les articles non utilisés."},
        {"input_query": "Client (German): Wie lange dauert der Versand nach Deutschland?",
         "retrieved_kb": "Shipping to Germany takes 7-10 business days.",
         "desired_response": "Der Versand nach Deutschland dauert 7-10 Werktage."}
    ]

example_template = """
{input_query}
Relevant Knowledge Base Info: {retrieved_kb}
Agent Response: {desired_response}
"""
example_prompt = PromptTemplate(input_variables=["input_query", "retrieved_kb", "desired_response"], template=example_template)

few_shot_prompt_template = FewShotPromptTemplate(
    examples=get_in_context_examples(),
    example_prompt=example_prompt,
    prefix="You are a helpful multilingual e-commerce customer support agent. Respond to customer queries in their language, leveraging the provided knowledge base information. Here are some examples of how to respond:\n",
    suffix="\nClient ({detected_language}): {user_query}\nRelevant Knowledge Base Info: {retrieved_kb_info}\nAgent Response:",
    input_variables=["detected_language", "user_query", "retrieved_kb_info"],
)

llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
llm_chain = LLMChain(prompt=few_shot_prompt_template, llm=llm)

st.title("Multilingual E-commerce Customer Support Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about our products or policies...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        try:
            detected_language = detect(user_input)

            docs = vectorstore.similarity_search(user_input, k=1)
            retrieved_kb_info = docs[0].page_content if docs else "No specific information found in the knowledge base."

            response = llm_chain.run(
                detected_language=detected_language,
                user_query=user_input,
                retrieved_kb_info=retrieved_kb_info
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}"})
            with st.chat_message("assistant"):
                st.markdown(f"Sorry, I encountered an error: {e}")
