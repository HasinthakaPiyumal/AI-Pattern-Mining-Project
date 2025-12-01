import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

medical_docs_dir = "./medical_docs"
os.makedirs(medical_docs_dir, exist_ok=True)

with open(os.path.join(medical_docs_dir, "diabetes.txt"), "w") as f:
    f.write("""Diabetes Mellitus is a chronic condition that affects how your body turns food into energy. \nMost of the food you eat is broken down into sugar (also called glucose) and released into your bloodstream. \nWhen your blood sugar goes up, it signals your pancreas to release insulin. \nInsulin acts like a key to let blood sugar into your body’s cells for use as energy. \nIf you have diabetes, your body either doesn’t make enough insulin or can’t use the insulin it makes as well as it should. \nWhen there isn’t enough insulin or cells stop responding to insulin, too much blood sugar stays in your bloodstream. \nOver time, that can cause serious health problems, such as heart disease, vision loss, and kidney disease.\nThere are three main types of diabetes: type 1, type 2, and gestational diabetes. \nType 1 diabetes is an autoimmune disease where the body does not produce insulin. \nType 2 diabetes is when the body doesn't use insulin well and can't keep blood sugar at normal levels. \nGestational diabetes develops in some women during pregnancy.""")

with open(os.path.join(medical_docs_dir, "hypertension.txt"), "w") as f:
    f.write("""Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. \nBlood pressure is determined by the amount of blood your heart pumps and the amount of resistance to blood flow in your arteries. \nThe more blood your heart pumps and the narrower your arteries, the higher your blood pressure. \nYou can have high blood pressure for years without any symptoms. \nUncontrolled high blood pressure increases your risk of serious health problems, including heart attack and stroke. \nFortunately, high blood pressure can be easily detected. And once you know you have high blood pressure, you can work with your doctor to control it. \nTreatments often include lifestyle changes like diet and exercise, and sometimes medications such as diuretics, beta-blockers, or ACE inhibitors.""")

with open(os.path.join(medical_docs_dir, "paracetamol.txt"), "w") as f:
    f.write("""Paracetamol, also known as acetaminophen, is a common pain reliever and fever reducer. \nIt is available over-the-counter and is found in many cold and flu remedies. \nIt works by blocking the production of certain chemicals in the brain called prostaglandins, which are involved in pain and fever. \nDosage should be carefully followed, as exceeding the recommended dose can lead to severe liver damage. \nFor adults, the typical dose is 500mg to 1000mg every 4 to 6 hours, not exceeding 4000mg in 24 hours. \nIt is generally safe for most people, including pregnant women and children, when used as directed. \nHowever, individuals with liver disease or those who consume alcohol regularly should consult a doctor before taking paracetamol.""")


loaders = [TextLoader(os.path.join(medical_docs_dir, fn)) for fn in os.listdir(medical_docs_dir)]
docs = []
for loader in loaders:
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(final_documents, embeddings, persist_directory="./chroma_db")

llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_template(
    """Answer the user's questions based on the below context:
    {context}
    Question: {input}"""
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

def ask_medical_chatbot(query: str) -> str:
    response = rag_chain.invoke({"input": query})
    return response["answer"]

if __name__ == "__main__":
    print("Medical Chatbot Initialized. Ask a question or type 'exit' to quit.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            break
        if not user_query:
            continue
        
        try:
            bot_response = ask_medical_chatbot(user_query)
            print(f"Bot: {bot_response}")
        except Exception as e:
            print(f"An error occurred: {e}. Please ensure your OPENAI_API_KEY is set correctly and you have an active internet connection.")

