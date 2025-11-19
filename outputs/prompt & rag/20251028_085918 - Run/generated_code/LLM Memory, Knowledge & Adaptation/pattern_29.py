import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI # Using OpenAI for simplicity, can be replaced by a local LLM

class QueryComplexityClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression()
        self.labels = ['simple', 'moderate', 'complex', 'urgent']
        self._train_dummy_classifier()

    def _train_dummy_classifier(self):
        training_data = [
            ("How often should I take this pill?", 'simple'),
            ("Explain the side effects of Warfarin in detail.", 'moderate'),
            ("What are the latest research findings on CRISPR gene editing for cancer treatment, including ethical considerations?", 'complex'),
            ("I'm experiencing severe chest pain and shortness of breath right now!", 'urgent'),
            ("What is fever?", 'simple'),
            ("Describe the symptoms of diabetes and its common treatments.", 'moderate'),
            ("Compare the efficacy of different immunotherapy approaches for metastatic melanoma.", 'complex'),
        ]
        texts, classes = zip(*training_data)
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, classes)

    def classify(self, query: str) -> str:
        if "right now" in query.lower() or "urgent" in query.lower() or "severe" in query.lower():
            return "urgent"
        query_vec = self.vectorizer.transform([query])
        prediction = self.classifier.predict(query_vec)[0]
        return prediction

class MemoryManagementSystem:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"): 
        self.short_term_memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
        self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        self.long_term_memory_store = FAISS.from_texts(["Initial medical history data."], self.embedding_function)
        self.patient_profile = {}

    def add_to_long_term_memory(self, text: str, metadata: dict = None):
        self.long_term_memory_store.add_texts([text], metadatas=[metadata] if metadata else None)

    def retrieve_long_term_memory(self, query: str, k: int = 2) -> list:
        return self.long_term_memory_store.similarity_search(query, k=k)

    def update_patient_profile(self, key: str, value: any):
        self.patient_profile[key] = value

    def get_patient_profile(self) -> dict:
        return self.patient_profile

    def get_context_for_llm(self, query: str) -> str:
        stm_history = self.short_term_memory.load_memory_variables({})["chat_history"]
        ltm_docs = self.retrieve_long_term_memory(query)
        patient_info = "Patient Profile: " + str(self.patient_profile)

        combined_context = f"""
        Conversation History: {stm_history}
        Patient Information: {patient_info}
        Relevant Medical History: {', '.join([doc.page_content for doc in ltm_docs])}
        """
        return combined_context

class KnowledgeManagementSystem:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        self.current_index = 0
        self.medical_knowledge_bases = [
            FAISS.from_texts(["Medical Fact 1: Aspirin is used to relieve pain, fever, and inflammation.", "Medical Fact 2: Diabetes is a chronic condition that affects how your body turns food into energy."], self.embedding_function),
            FAISS.from_texts(["Medical Fact A: Latest research on cancer immunotherapy involves CAR T-cell therapy.", "Medical Fact B: Hypertension, or high blood pressure, can lead to heart disease."], self.embedding_function)
        ]
        self.human_readable_kb = {
            "drug_interactions": {"aspirin_warfarin": "Increased bleeding risk."},
            "disease_symptoms": {"diabetes": "Frequent urination, increased thirst, unexplained weight loss."}, 
            "general_guidelines": "Always consult a doctor for medical advice."
        }

    def retrieve_knowledge(self, query: str, k: int = 3) -> list:
        vector_knowledge = self.medical_knowledge_bases[self.current_index].similarity_search(query, k=k)
        
        relevant_hr_knowledge = []
        for category, data in self.human_readable_kb.items():
            for key, value in data.items():
                if query.lower() in key.lower() or query.lower() in value.lower():
                    relevant_hr_knowledge.append(f"{key}: {value}")
            if query.lower() in category.lower():
                 relevant_hr_knowledge.append(f"{category}: {data}")

        combined_knowledge = [doc.page_content for doc in vector_knowledge] + relevant_hr_knowledge
        return combined_knowledge

    def update_human_readable_kb(self, category: str, key: str, value: any):
        if category not in self.human_readable_kb:
            self.human_readable_kb[category] = {}
        self.human_readable_kb[category][key] = value

    def hotswap_index(self, new_index_data: list):
        new_faiss_index = FAISS.from_texts(new_index_data, self.embedding_function)
        self.medical_knowledge_bases.append(new_faiss_index)
        self.current_index = len(self.medical_knowledge_bases) - 1
        print(f"Knowledge index hotswapped to version {self.current_index}")

class AdaptiveResponseStrategyModule:
    def __init__(self, llm):
        self.llm = llm
        self.simple_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful medical assistant. Provide concise answers."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}")
        ])
        self.moderate_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a knowledgeable medical assistant. Provide detailed and accurate information, referencing medical facts provided in context."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Context: {context}\nMedical Knowledge: {medical_knowledge}\nQuestion: {question}")
        ])
        self.complex_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert medical consultant. Analyze the provided context and extensive medical knowledge to give a comprehensive, nuanced, and evidence-based answer."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Context: {context}\nMedical Knowledge: {medical_knowledge}\nQuestion: {question}")
        ])
        self.urgent_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an emergency medical assistant. Provide immediate, critical advice. Emphasize seeking professional medical help without delay."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}")
        ])

    def get_chain_for_complexity(self, complexity: str, retriever=None):
        if complexity == 'simple':
            return {"question": RunnablePassthrough(), "chat_history": RunnablePassthrough()} | self.simple_prompt | self.llm | StrOutputParser()
        elif complexity == 'moderate':
            return {"context": RunnablePassthrough(), "medical_knowledge": retriever, "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()} | self.moderate_prompt | self.llm | StrOutputParser()
        elif complexity == 'complex':
             return {"context": RunnablePassthrough(), "medical_knowledge": retriever, "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()} | self.complex_prompt | self.llm | StrOutputParser()
        elif complexity == 'urgent':
            return {"question": RunnablePassthrough(), "chat_history": RunnablePassthrough()} | self.urgent_prompt | self.llm | StrOutputParser()
        else:
            return {"question": RunnablePassthrough(), "chat_history": RunnablePassthrough()} | self.simple_prompt | self.llm | StrOutputParser()

class EfficientLLMFineTuningModule:
    def __init__(self, model_name="llama-2-7b-chat-hf", dataset_path="./medical_finetune_data.json"):
        self.model_name = model_name
        self.dataset_path = dataset_path

    def generate_synthetic_data(self, num_samples=10):
        synthetic_data = []
        prompts = [
            "What are the common symptoms of a cold?",
            "How does insulin work in the body?",
            "Explain the importance of vaccinations.",
            "What are the risks associated with high cholesterol?"
        ]
        answers = [
            "Common cold symptoms include a runny nose, sore throat, cough, and congestion.",
            "Insulin is a hormone that helps regulate blood sugar levels. It allows glucose to enter cells for energy.",
            "Vaccinations are crucial for preventing infectious diseases by building immunity.",
            "High cholesterol can lead to atherosclerosis, increasing the risk of heart attack and stroke."
        ]
        for i in range(num_samples):
            prompt = prompts[i % len(prompts)]
            answer = answers[i % len(answers)]
            synthetic_data.append({"prompt": prompt, "completion": answer})
        return synthetic_data

    def fine_tune_llm(self, training_data: list, config: dict = None):
        print(f"Simulating fine-tuning of LLM '{self.model_name}' with {len(training_data)} samples.")
        print("Using LoRA/QLoRA techniques via Hugging Face TRL and Accelerate.")
        print(f"Training data example: {training_data[0] if training_data else 'N/A'}")
        print("WandB would be used for experiment tracking here.")
        print("Fine-tuning completed (simulated).")

class MedicalAssistant:
    def __init__(self, openai_api_key: str = "YOUR_OPENAI_API_KEY"):
        self.qcc = QueryComplexityClassifier()
        self.memory_manager = MemoryManagementSystem()
        self.knowledge_manager = KnowledgeManagementSystem()
        self.llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
        self.adaptive_strategy = AdaptiveResponseStrategyModule(self.llm)
        self.finetuning_module = EfficientLLMFineTuningModule()

    def process_query(self, query: str) -> str:
        complexity = self.qcc.classify(query)
        print(f"Query classified as: {complexity}")

        long_term_context = self.memory_manager.get_context_for_llm(query)
        medical_knowledge = self.knowledge_manager.retrieve_knowledge(query)

        # Custom retriever for the RAG chain
        def custom_retriever(q: str):
            return "\n".join(self.knowledge_manager.retrieve_knowledge(q))
        
        retrieval_chain = RunnablePassthrough.assign(
            medical_knowledge=custom_retriever
        )

        # Create the LangChain processing chain
        if complexity == 'simple' or complexity == 'urgent':
            chain = self.adaptive_strategy.get_chain_for_complexity(
                complexity,
            )
            response = chain.invoke({"question": query, "chat_history": self.memory_manager.short_term_memory.load_memory_variables({})["chat_history"]})
        else:
            chain = self.adaptive_strategy.get_chain_for_complexity(
                complexity,
                retriever=retrieval_chain.get_graph().invoke({"question": query})["medical_knowledge"] # Manual invocation to get the knowledge for now
            )
            response = chain.invoke({"context": long_term_context, "question": query, "chat_history": self.memory_manager.short_term_memory.load_memory_variables({})["chat_history"]})
        
        self.memory_manager.short_term_memory.save_context({"input": query}, {"output": response})
        return response

    def run_finetuning_example(self):
        print("\n--- Running Fine-tuning Example ---")
        synthetic_data = self.finetuning_module.generate_synthetic_data(num_samples=5)
        self.finetuning_module.fine_tune_llm(synthetic_data)
        print("-----------------------------------")

    def update_knowledge_base(self, new_facts: list):
        print("\n--- Updating Knowledge Base (Hotswap) ---")
        self.knowledge_manager.hotswap_index(new_facts)
        print("-----------------------------------")

    def add_patient_data(self, data: str, metadata: dict = None):
        print("\n--- Adding Patient Data to LTM ---")
        self.memory_manager.add_to_long_term_memory(data, metadata)
        print("-----------------------------------")

if __name__ == "__main__":
    # Replace with your actual OpenAI API key or configure a local LLM
    # For local LLM setup, replace ChatOpenAI with a suitable local model interface
    # e.g., from langchain_community.llms import LlamaCpp, HuggingFacePipeline
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    
    assistant = MedicalAssistant(openai_api_key="sk-YOUR_OPENAI_API_KEY_HERE") 

    print("\n--- Initializing Patient Profile ---")
    assistant.memory_manager.update_patient_profile("name", "John Doe")
    assistant.memory_manager.update_patient_profile("age", 45)
    assistant.memory_manager.update_patient_profile("allergies", "Penicillin")
    assistant.add_patient_data("John Doe has a history of mild hypertension since 2018.", {"date": "2023-01-15"})
    assistant.add_patient_data("Prescribed Lisinopril 10mg daily for hypertension.", {"date": "2023-03-01"})

    print("\n--- Conversational Flow ---")
    queries = [
        "Hello, what are the common symptoms of a flu?",
        "Can you tell me about the potential side effects of Lisinopril?",
        "What is the latest research on preventing Alzheimer's disease, considering genetic factors?",
        "I'm experiencing sudden severe dizziness and blurred vision! What should I do?",
        "How does penicillin allergy manifest?",
        "What is my current blood pressure medication?",
        "Can you summarize my medical history regarding hypertension?"
    ]

    for q in queries:
        print(f"\nPatient: {q}")
        response = assistant.process_query(q)
        print(f"Assistant: {response}")

    assistant.update_knowledge_base([
        "New Medical Fact: Recent studies suggest exercise significantly reduces Alzheimer's risk.",
        "New Medical Fact: Advanced research in genetic predispositions for dementia is ongoing."
    ])
    
    print("\n--- Continuing Conversational Flow After KB Update ---")
    query_after_update = "What are the very latest findings on Alzheimer's prevention, specifically relating to genetic factors and lifestyle?"
    print(f"\nPatient: {query_after_update}")
    response_after_update = assistant.process_query(query_after_update)
    print(f"Assistant: {response_after_update}")

    assistant.run_finetuning_example()