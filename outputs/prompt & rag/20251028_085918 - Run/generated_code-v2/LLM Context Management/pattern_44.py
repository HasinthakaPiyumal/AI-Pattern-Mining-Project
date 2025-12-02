"""
medical_ai_assistant.py

A simulated Medical AI Assistant leveraging a Retrieval Augmented Generation (RAG) pattern.
It combines a mock Generative Language Model (parametric memory) with a mock external medical knowledge base (non-parametric memory)
to provide evidence-based answers to medical queries.
"""

import re

class MedicalAIAssistant:
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self):
        """
        Simulates loading an external, non-parametric medical knowledge base.
        In a real application, this would be a large database of medical literature,
        patient records, drug information, etc., potentially stored in a vector database.
        """
        print("Loading simulated medical knowledge base...")
        return [
            {
                "id": "doc1",
                "content": """Diabetes Mellitus Type 2 is a chronic condition characterized by high blood sugar levels. 
                              It is often managed with lifestyle changes, oral medications like Metformin, and sometimes insulin.
                              Symptoms include increased thirst, frequent urination, and unexplained weight loss. \nRegular monitoring of blood glucose is crucial."""
            },
            {
                "id": "doc2",
                "content": """Hypertension, or high blood pressure, is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. 
                              Treatment often involves ACE inhibitors, calcium channel blockers, diuretics, and beta-blockers. \nLifestyle modifications like diet and exercise are also vital."""
            },
            {
                "id": "doc3",
                "content": """Malignant Hyperthermia (MH) is a rare, life-threatening condition that is triggered by exposure to certain anesthetic agents. 
                              It is characterized by a rapid rise in body temperature, muscle rigidity, and tachycardia. 
                              Dantrolene is the specific treatment for MH. \nGenetic testing can identify individuals at risk."""
            },
            {
                "id": "doc4",
                "content": """Common symptoms of myocardial infarction (heart attack) include chest pain radiating to the arm, shortness of breath, and sweating. \nImmediate medical attention is necessary. Treatment may involve angioplasty or bypass surgery."""
            },
            {
                "id": "doc5",
                "content": """Metformin is a first-line medication for Type 2 Diabetes. It works by decreasing glucose production in the liver and improving insulin sensitivity. \nCommon side effects include gastrointestinal upset."""
            },
            {
                "id": "doc6",
                "content": """A patient's record indicates a history of uncontrolled hypertension and recent onset of Type 2 Diabetes. \nBlood pressure readings have been consistently above 140/90 mmHg, and HbA1c is 8.5%. \nMedications: Lisinopril 20mg daily, Metformin 1000mg twice daily."""
            },
        ]

    def _retrieve_documents(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Simulates retrieving relevant documents from the knowledge base.
        In a real RAG system, this would involve embedding the query and documents,
        and performing a vector similarity search (e.g., using Chroma, FAISS).
        Here, it uses simple keyword matching.
        """
        print(f"Retrieving documents for query: '{query}'...")
        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
        
        scored_documents = []
        for doc in self.knowledge_base:
            doc_content_lower = doc["content"].lower()
            score = sum(1 for keyword in query_keywords if keyword in doc_content_lower)
            if score > 0:
                scored_documents.append((score, doc))
        
        # Sort by score in descending order and return top_k
        scored_documents.sort(key=lambda x: x[0], reverse=True)
        retrieved_docs = [doc for score, doc in scored_documents[:top_k]]
        
        if not retrieved_docs:
            print("No relevant documents found.")
        else:
            print(f"Found {len(retrieved_docs)} relevant documents.")
        return retrieved_docs

    def _generate_response(self, query: str, context: list[dict]) -> str:
        """
        Simulates a Generative Language Model's response generation.
        In a real system, this would be an actual LLM call (e.g., OpenAI API, local Llama model).
        It combines its 'parametric memory' (inherent reasoning abilities) with the 'non-parametric memory'
        provided by the retrieved context.
        """
        print("Generating response with LLM simulation...")
        context_text = "\n\nRelevant Medical Information:\n" if context else ""
        for doc in context:
            context_text += f"- {doc['content']}\n"
        
        if not context:
            # Fallback if no context is found, showing the LLM's 'parametric' knowledge limitation
            return f"Based on my general medical knowledge, I can tell you that '{query}' is a medical topic. However, without specific external information, I can only provide a general overview. Please consult a specialist for detailed advice."

        # Simulate LLM's synthesis
        simulated_llm_response = (
            f"As a medical AI assistant, based on your query: '{query}', and the following retrieved medical information:\n"
            f"{context_text}\n"
            f"I can provide a synthesized answer. This information should be used for professional reference only and not as a substitute for clinical judgment."
        )

        # Simple conditional logic to make the response seem more 