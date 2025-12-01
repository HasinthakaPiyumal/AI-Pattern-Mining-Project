import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

class RAGSequenceMedicalAssistant:
    def __init__(self, model_name="google/flan-t5-small", sbert_model_name="all-MiniLM-L6-v2", collection_name="medical_corpus_collection"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=sbert_model_name)
        self.client = chromadb.Client()
        self.collection_name = collection_name
        
        try:
            self.collection = self.client.get_collection(name=self.collection_name, embedding_function=self.chroma_ef)
            self.collection.delete(ids=self.collection.get()['ids'])
        except Exception:
            self.collection = self.client.create_collection(name=self.collection_name, embedding_function=self.chroma_ef)

        self.medical_corpus = [
            "Influenza, commonly known as the flu, is an infectious disease caused by influenza viruses. Symptoms can range from mild to severe and often include fever, runny nose, sore throat, muscle pains, headache, coughing, and fatigue.",
            "Type 2 diabetes is a long-term medical condition in which your body doesn’t use insulin properly, leading to high blood sugar levels. Symptoms include increased thirst, frequent urination, increased hunger, fatigue, and blurred vision.",
            "Migraine is a primary headache disorder characterized by recurrent headaches that are moderate to severe. Typically, the headaches affect one half of the head, are throbbing in nature, and last from 2 to 72 hours. Associated symptoms may include nausea, vomiting, and sensitivity to light, sound, or smell.",
            "Hypertension, or high blood pressure, is a serious medical condition that significantly increases the risks of heart, brain, kidney and other diseases. It is often asymptomatic.",
            "Common cold is a viral infectious disease of the upper respiratory tract that primarily affects the nose. Symptoms include coughing, sore throat, runny nose, sneezing, and fever. It is generally milder than the flu.",
            "Asthma is a chronic lung disease that inflames and narrows the airways. Symptoms include wheezing, shortness of breath, chest tightness, and coughing. Triggers can include allergens, exercise, and cold air.",
            "Allergic rhinitis, also known as hay fever, is a type of inflammation in the nose which occurs when the immune system overreacts to allergens in the air. Symptoms include a runny nose, sneezing, itchy eyes, and nasal congestion."
        ]
        self._initialize_chroma_db()

    def _initialize_chroma_db(self):
        self.collection.add(
            documents=self.medical_corpus,
            ids=[f"doc_{i}" for i in range(len(self.medical_corpus))]
        )

    def diagnose(self, symptoms: str, medical_history: str = "", k_retrievals: int = 3):
        query_text = f"Patient Symptoms: {symptoms}\nMedical History: {medical_history}"
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=k_retrievals
        )
        
        retrieved_documents = results['documents'][0]
        
        diagnostic_explanations = []
        for i, doc_content in enumerate(retrieved_documents):
            prompt = (
                f"Context: {doc_content}\n"
                f"Patient Symptoms: {symptoms}\n"
                f"Medical History: {medical_history}\n"
                f"Diagnosis Explanation:"
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            outputs = self.llm.generate(
                inputs.input_ids,
                max_new_tokens=200,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            diagnostic_explanations.append(f"Explanation based on Document {i+1}:\n{explanation}\n---")
            
        return "\n".join(diagnostic_explanations)

assistant = RAGSequenceMedicalAssistant()
user_symptoms = "Persistent coughing, wheezing, shortness of breath after exercise"
user_history = "Has a family history of allergies, used to have childhood asthma"
diagnosis = assistant.diagnose(user_symptoms, user_history)
print(diagnosis)