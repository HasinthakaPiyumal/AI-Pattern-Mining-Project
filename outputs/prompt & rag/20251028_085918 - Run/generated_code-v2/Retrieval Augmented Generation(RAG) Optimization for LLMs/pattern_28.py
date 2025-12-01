class MedicalKnowledgeBase:
    def __init__(self):
        self.knowledge = [
            "Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Symptoms include increased thirst, frequent urination, and unexplained weight loss.",
            "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Regular exercise and a balanced diet can help manage it.",
            "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain from conditions such as muscle aches, toothaches, common cold, and headaches. It can also be used to reduce inflammation and as a blood thinner.",
            "Common cold is a viral infection of your nose and throat (upper respiratory tract). Symptoms can include a runny nose, sore throat, cough, congestion, slight body aches or a mild headache, sneezing, and low-grade fever. There is no cure for the common cold, but treatments focus on relieving symptoms.",
            "Influenza (flu) is a contagious respiratory illness caused by influenza viruses. It can cause mild to severe illness, and at times can lead to death. The flu vaccine is recommended annually to prevent infection.",
            "Paracetamol (acetaminophen) is a pain reliever and a fever reducer. It is used to treat many conditions such as headaches, muscle aches, arthritis, backache, toothaches, colds, and fevers.",
            "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out, and shortness of breath. Triggers include allergens, exercise, and cold air.",
            "Antibiotics are medicines that fight bacterial infections in people and animals. They work by killing the bacteria or by making it difficult for the bacteria to grow and multiply. They are not effective against viral infections like the common cold or flu."
        ]

    def get_all_passages(self):
        return self.knowledge


class NeuralRetriever:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def retrieve(self, query, top_k=3):
        query_keywords = [word.lower() for word in query.split() if len(word) > 2]
        relevant_passages = []
        for passage in self.knowledge_base.get_all_passages():
            if any(keyword in passage.lower() for keyword in query_keywords):
                relevant_passages.append(passage)
        return relevant_passages[:top_k]


class GenerativeLanguageModel:
    def generate_answer(self, query, retrieved_passages):
        if not retrieved_passages:
            return "I apologize, but I couldn't find specific information related to your query in my current knowledge base. Please try rephrasing your question or provide more details."

        answer_parts = [
            f"Based on your question about '{query}', here is some information I found:",
            "\n\nRetrieved information:"
        ]

        for i, passage in enumerate(retrieved_passages):
            answer_parts.append(f"- {passage}")

        answer_parts.append("\n\nPlease note: This information is for general knowledge and should not replace professional medical advice. Always consult a healthcare professional for diagnosis and treatment.")

        return "\n".join(answer_parts)


class MedicalQASystem:
    def __init__(self):
        self.knowledge_base = MedicalKnowledgeBase()
        self.retriever = NeuralRetriever(self.knowledge_base)
        self.generator = GenerativeLanguageModel()

    def ask_question(self, query):
        retrieved_passages = self.retriever.retrieve(query)
        answer = self.generator.generate_answer(query, retrieved_passages)
        return answer


if __name__ == "__main__":
    qa_system = MedicalQASystem()

    print("\n--- Medical Information Q&A System ---")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nEnter your medical question: ")
        if user_query.lower() == 'exit':
            break

        response = qa_system.ask_question(user_query)
        print("\n--- Answer ---")
        print(response)
        print("----------------")