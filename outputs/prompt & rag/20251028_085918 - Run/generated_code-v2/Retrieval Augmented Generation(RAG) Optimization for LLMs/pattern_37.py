import os
import openai

# --- 1. Simulated Knowledge Base Module ---
medical_knowledge_base = [
    "Influenza, commonly known as the flu, is an infectious disease caused by influenza viruses. Symptoms can range from mild to severe and often include fever, runny nose, sore throat, muscle pains, headache, coughing, and fatigue. These symptoms begin one to four days after exposure to the virus and usually last for about a week. Complications can include pneumonia.",
    "Common cold symptoms typically include a runny nose, sneezing, sore throat, and cough. Unlike the flu, a cold rarely causes a fever or headache. Colds are caused by various viruses, most commonly rhinoviruses, and usually resolve within 7-10 days.",
    "Diabetes mellitus, commonly known as diabetes, is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored or used for energy. With diabetes, your body either doesn't make enough insulin or can't effectively use the insulin it does make. Symptoms include frequent urination, increased thirst, and increased hunger.",
    "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. It often has no symptoms, but severe hypertension can cause headaches, nosebleeds, or shortness of breath.",
    "Migraines are severe headaches often accompanied by symptoms such as throbbing pain on one side of the head, nausea, vomiting, and extreme sensitivity to light and sound. Triggers can include stress, certain foods, and hormonal changes. Treatment often involves pain relievers and preventative medications."
]

class KnowledgeBase:
    def __init__(self, documents):
        self.documents = documents

    def retrieve_documents(self, query: str, top_k: int = 2) -> list[str]:
        query_keywords = [word.lower() for word in query.split() if len(word) > 2] # Simple keyword extraction
        scored_documents = []

        for doc_id, doc in enumerate(self.documents):
            score = 0
            doc_lower = doc.lower()
            for keyword in query_keywords:
                if keyword in doc_lower:
                    score += doc_lower.count(keyword)
            if score > 0:
                scored_documents.append((score, doc))
        
        # Sort by score in descending order and return top_k documents
        scored_documents.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_documents[:top_k]]

# --- 2. Language Model (LM) Integration Module ---
class LMIntegration:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_lm_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Or any other suitable model
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant. Provide information based on the given context. If the context does not contain enough information, state that you cannot provide a definitive answer."},  # Adjust as needed
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except openai.APIConnectionError as e:
            return f"Could not connect to OpenAI API: {e}"
        except openai.RateLimitError as e:
            return f"OpenAI API rate limit exceeded: {e}"
        except openai.APIStatusError as e:
            return f"OpenAI API error: {e.status_code} - {e.response}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

# --- 3. InContext RALM Orchestration Module (Core Logic) ---
class MedicalDiagnosisAssistant:
    def __init__(self, knowledge_base_docs):
        self.knowledge_base = KnowledgeBase(knowledge_base_docs)
        self.lm_integration = LMIntegration()

    def get_diagnosis_suggestion(self, user_query: str) -> str:
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.knowledge_base.retrieve_documents(user_query)

        # Step 2: Construct the augmented prompt
        context_prefix = ""
        if retrieved_docs:
            context_prefix = "Relevant Medical Information:\n" + "\n---\n".join(retrieved_docs) + "\n\n"
        
        augmented_prompt = f"{context_prefix}User Query: {user_query}\n\nBased on the information provided, what are the potential conditions or relevant medical facts?"
        
        # Step 3: Get response from the Language Model
        lm_response = self.lm_integration.get_lm_response(augmented_prompt)
        return lm_response

# --- Main Application Logic (Conceptual User Interface) ---
if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set. The LM will not function.")
        print("Please set it (e.g., export OPENAI_API_KEY='your_key') before running.")

    assistant = MedicalDiagnosisAssistant(medical_knowledge_base)
    print("\nWelcome to the Medical Diagnosis Assistant (InContext RALM Demo)!")
    print("Ask me about symptoms, conditions, or medical queries. Type 'exit' to quit.")

    while True:
        user_input = input("\nYour Query: ")
        if user_input.lower() == 'exit':
            print("Exiting assistant. Goodbye!")
            break
        
        if not user_input.strip():
            print("Please enter a query.")
            continue

        response = assistant.get_diagnosis_suggestion(user_input)
        print(f"\nAssistant: {response}")
