from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class MedicalRAGAssistant:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        self.vectorstore = None
        self.llm = self._initialize_llm()
        self.qa_chain = self._initialize_qa_chain()

    def _initialize_llm(self):
        # Use a smaller model for demonstration; 'facebook/bart-large-cnn' can be used for summarization
        # For question answering, flan-t5 is a good choice.
        text_generation_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            tokenizer="google/flan-t5-small",
            max_new_tokens=200,
            device=-1 # -1 for CPU, 0 for GPU
        )
        return HuggingFacePipeline(pipeline=text_generation_pipeline)

    def _initialize_qa_chain(self):
        template = """Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
        prompt = PromptTemplate.from_template(template)
        return LLMChain(llm=self.llm, prompt=prompt)

    def ingest_documents(self, documents):
        texts = self.text_splitter.split_text(documents)
        # FAISS expects a list of documents, so we'll treat each chunk as a separate document
        # and generate embeddings for them.
        embeddings = self.embedding_model.encode(texts)
        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=[(text, embedding) for text, embedding in zip(texts, embeddings)],
            embedding=self.embedding_model # Pass the SentenceTransformer directly as the embedding function
        )

    def answer_question(self, question, k=3):
        if not self.vectorstore:
            return "Knowledge base not initialized. Please ingest documents first."

        retrieved_docs = self.vectorstore.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        response = self.qa_chain.run(context=context, question=question)
        return response.strip()

if __name__ == "__main__":
    assistant = MedicalRAGAssistant()

    # Simulate medical documents (in a real scenario, these would come from parsed PDFs, databases, etc.)
    medical_data = [
        "Aspirin is commonly used as a pain reliever and to reduce fever. It also has anti-inflammatory effects. Low-dose aspirin is often prescribed to prevent heart attacks and strokes due to its antiplatelet properties. Side effects can include stomach upset and increased risk of bleeding.",
        "Type 2 diabetes is a chronic condition that affects the way the body processes blood sugar (glucose). The body either doesn't produce enough insulin, or it resists the effects of insulin. Treatment often involves lifestyle changes, oral medications, and sometimes insulin injections.",
        "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Medications like ACE inhibitors, diuretics, and beta-blockers are frequently used.",
        "The COVID-19 pandemic, caused by the SARS-CoV-2 virus, led to widespread respiratory illness. Symptoms range from mild to severe, including fever, cough, and shortness of breath. Vaccines have been developed to prevent severe disease. Antiviral treatments like Paxlovid are used for certain high-risk patients.",
        "Migraine is a severe headache accompanied by symptoms such as throbbing in the head, nausea, vomiting, and extreme sensitivity to light and sound. It can be triggered by various factors including stress, certain foods, and hormonal changes. Triptans and NSAIDs are common acute treatments, while beta-blockers can be used for prevention.",
        "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. It can be caused by bacteria, viruses, or fungi. Symptoms include cough with phlegm, fever, chills, and difficulty breathing. Antibiotics are used for bacterial pneumonia."
    ]
    
    # Ingest the simulated documents into the knowledge base
    print("Ingesting medical documents...")
    assistant.ingest_documents(" ".join(medical_data))
    print("Documents ingested. Ready to answer questions.")

    # Example questions
    questions = [
        "What is aspirin used for?",
        "What are common treatments for type 2 diabetes?",
        "How is high blood pressure treated?",
        "What causes COVID-19?",
        "Symptoms of migraine?",
        "What are the causes of pneumonia?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer = assistant.answer_question(q)
        print(f"Answer: {answer}")

    print("\nDemonstration complete.")