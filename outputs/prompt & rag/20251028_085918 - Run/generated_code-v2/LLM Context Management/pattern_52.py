class MedicalKnowledgeBase:
    def __init__(self):
        self.documents = [
            "Aspirin is commonly used for pain relief, fever reduction, and anti-inflammatory purposes. It can also be used to prevent blood clots.",
            "Diabetes Mellitus is a metabolic disease that causes high blood sugar. Type 1 diabetes is an autoimmune disease, while Type 2 diabetes is often linked to lifestyle factors.",
            "Hypertension, or high blood pressure, is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
            "Common side effects of antibiotics include nausea, diarrhea, and allergic reactions. It's important to complete the full course of antibiotics as prescribed.",
            "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing.",
            "The flu vaccine is recommended annually for most individuals to protect against influenza viruses. It helps prevent serious illness, hospitalizations, and deaths.",
            "Vitamin D is essential for bone health and immune function. It can be obtained through sun exposure, certain foods, and supplements.",
            "Cardiovascular disease refers to a range of conditions that affect your heart. These include coronary artery disease, heart attack, stroke, and heart failure.",
            "Migraine is a type of headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound.",
            "Asthma is a chronic lung disease that inflames and narrows the airways, causing recurring periods of wheezing, chest tightness, shortness of breath, and coughing."
        ]

    def retrieve_documents(self, query, top_k=2):
        query_words = set(query.lower().split())
        scored_documents = []
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                scored_documents.append((overlap, doc))
        
        scored_documents.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_documents[:top_k]]

class SimulatedLLM:
    def generate_response(self, prompt):
        # A very basic simulation of an LLM's response
        if "aspirin" in prompt.lower() and "pain" in prompt.lower():
            return "Aspirin is widely used for pain relief and reducing inflammation. Always consult a healthcare professional for dosage and suitability."
        elif "diabetes" in prompt.lower() and "blood sugar" in prompt.lower():
            return "Diabetes Mellitus is characterized by high blood sugar. Both Type 1 and Type 2 diabetes require careful management, often involving lifestyle changes, medication, or insulin."
        elif "hypertension" in prompt.lower() or "high blood pressure" in prompt.lower():
            return "Hypertension is a serious condition that can lead to heart disease. Regular monitoring and appropriate medical intervention are crucial."
        elif "antibiotics" in prompt.lower() and "side effects" in prompt.lower():
            return "Antibiotics can have side effects like nausea and diarrhea. It's vital to complete the full prescribed course to prevent antibiotic resistance."
        elif "pneumonia" in prompt.lower() and "lungs" in prompt.lower():
            return "Pneumonia is an infection affecting the lungs, causing symptoms like cough, fever, and difficulty breathing. Medical treatment is often necessary."
        elif "flu vaccine" in prompt.lower() or "influenza" in prompt.lower():
            return "The annual flu vaccine helps protect against influenza and can reduce the severity of illness. It's recommended for widespread public health."
        elif "vitamin d" in prompt.lower() and "bone health" in prompt.lower():
            return "Vitamin D is crucial for maintaining bone health and supporting the immune system. Sources include sunlight, certain foods, and supplements."
        elif "cardiovascular disease" in prompt.lower() or "heart" in prompt.lower():
            return "Cardiovascular diseases encompass various conditions affecting the heart and blood vessels. Early diagnosis and management are key to preventing serious complications."
        elif "migraine" in prompt.lower() and "headache" in prompt.lower():
            return "Migraines are severe headaches often accompanied by other symptoms like nausea and sensitivity to light. Treatment options vary depending on the individual."
        elif "asthma" in prompt.lower() and "airways" in prompt.lower():
            return "Asthma is a chronic respiratory condition that causes inflammation and narrowing of the airways, leading to breathing difficulties. Management typically involves medication to control symptoms."
        else:
            return "I am a medical information assistant. Please ask a specific medical question. (Note: My knowledge is limited to simulated data.)"

class MedicalInformationAssistant:
    def __init__(self):
        self.knowledge_base = MedicalKnowledgeBase()
        self.llm = SimulatedLLM()

    def get_medical_answer(self, query):
        # 1. Retrieve relevant documents
        retrieved_docs = self.knowledge_base.retrieve_documents(query)
        
        # 2. Prepare context for LLM
        context = "\n\nRetrieved Medical Information:\n"
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                context += f"- Document {i+1}: {doc}\n"
        else:
            context += "No highly relevant documents found in the knowledge base.\n"
        
        full_prompt = f"Based on the following information and your general medical knowledge, please answer the question:\n\nQuestion: {query}{context}\n\nAnswer:"

        # 3. Generate response using LLM
        answer = self.llm.generate_response(full_prompt)
        return answer

def main():
    assistant = MedicalInformationAssistant()
    print("Welcome to the Medical Information Assistant (Simulated RAG System).")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nEnter your medical question: ")
        if user_query.lower() == 'exit':
            print("Exiting Medical Information Assistant. Goodbye!")
            break
        
        response = assistant.get_medical_answer(user_query)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()