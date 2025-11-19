import random

class MedQueryAssistant:
    def __init__(self, medical_knowledge_base=None):
        self.conversation_history = []
        self.medical_knowledge_base = medical_knowledge_base if medical_knowledge_base is not None else self._initialize_default_kb()
        # In a real system, an LLM would be loaded here, e.g., from transformers or an API
        self.llm_model = "Placeholder LLM"
        print("MedQuery AI Assistant initialized.")

    def _initialize_default_kb(self):
        # Simulate a simple medical knowledge base
        return {
            "hypertension": "High blood pressure, a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
            "diabetes": "A chronic condition that affects how your body turns food into energy. Most of the food you eat is broken down into sugar (also called glucose) and released into your bloodstream.",
            "headache treatment": "Common treatments include over-the-counter pain relievers (e.g., ibuprofen, acetaminophen), rest, hydration, and avoiding triggers. For severe headaches, prescription medications may be necessary.",
            "common cold symptoms": "Runny or stuffy nose, sore throat, cough, congestion, slight body aches or a mild headache, sneezing, low-grade fever, and a general feeling of being unwell (malaise).",
            "cardiac arrest": "A sudden, unexpected loss of heart function, breathing, and consciousness. This is a medical emergency."
        }

    def _classify_query_complexity(self, query: str) -> str:
        # Simple heuristic for demonstration
        # In a real system, this would be an ML classifier (e.g., scikit-learn Logistic Regression, or a fine-tuned small LLM)
        if any(keyword in query.lower() for keyword in ["explain", "detail", "mechanism", "pathophysiology", "research", "compare"]):
            return "complex"
        elif len(query.split()) > 7 and "?" in query: # Longer questions with a question mark
             return "complex"
        return "simple"

    def _retrieve_short_term_memory(self):
        # Simulate retrieving recent conversation history
        return "\n".join(self.conversation_history[-5:]) # Last 5 turns

    def _retrieve_long_term_memory(self, query: str):
        # Simulate retrieving from a knowledge base
        # In a real system, this would involve embeddings, vector search (e.g., with FAISS, Chroma), and RAG
        retrieved_info = []
        query_lower = query.lower()
        for key, value in self.medical_knowledge_base.items():
            if key.lower() in query_lower or any(word in query_lower for word in key.lower().split()):
                retrieved_info.append(f"Knowledge from KB about '{key}': {value}")
        return "\n".join(retrieved_info) if retrieved_info else "No specific long-term knowledge found."

    def process_query(self, user_query: str) -> str:
        print(f"\nProcessing query: '{user_query}'")
        self.conversation_history.append(f"User: {user_query}")

        complexity = self._classify_query_complexity(user_query)
        print(f"Query classified as: {complexity}")

        context_data = ""
        if complexity == "simple":
            context_data += "Relevant short-term memory:\n" + self._retrieve_short_term_memory()
            print("Using short-term memory strategy.")
        elif complexity == "complex":
            context_data += "Relevant short-term memory:\n" + self._retrieve_short_term_memory()
            context_data += "\nRelevant long-term memory:\n" + self._retrieve_long_term_memory(user_query)
            print("Using long-term memory (KB) strategy.")

        # Simulate LLM response generation
        # In a real application, this would be an API call or local LLM inference
        # The prompt to the LLM would include user_query and context_data
        llm_response_template = "As a MedQuery AI Assistant, given the context:\n---\n{context}\n---\nRegarding the query '{query}', I would respond with: "

        simulated_llm_output = ""
        if "No specific long-term knowledge found" in context_data and complexity == "complex":
            simulated_llm_output = "I couldn't find specific long-term knowledge for that complex query. Please rephrase or provide more details."
        elif "explain hypertension" in user_query.lower():
            simulated_llm_output = self.medical_knowledge_base.get("hypertension", "Information on hypertension is not available.")
        elif "symptoms of common cold" in user_query.lower():
            simulated_llm_output = self.medical_knowledge_base.get("common cold symptoms", "Information on common cold symptoms is not available.")
        elif "what is diabetes" in user_query.lower():
            simulated_llm_output = self.medical_knowledge_base.get("diabetes", "Information on diabetes is not available.")
        elif "how to treat a headache" in user_query.lower():
            simulated_llm_output = self.medical_knowledge_base.get("headache treatment", "Information on headache treatment is not available.")
        else:
             simulated_llm_output = f"Acknowledged query: '{user_query}'. Based on the {complexity} strategy, I have processed the request. (Simulated LLM processing details - Context provided: {context_data[:100]}...)"


        final_response = f"MedQuery Assistant: {simulated_llm_output}"
        self.conversation_history.append(final_response)
        return final_response

# Example Usage:
# assistant = MedQueryAssistant()
# print(assistant.process_query("What are the symptoms of a common cold?"))
# print(assistant.process_query("Explain hypertension in detail."))
# print(assistant.process_query("How do you treat a headache?"))
# print(assistant.process_query("What did I ask before?"))
# print(assistant.process_query("Compare diabetes types 1 and 2, including their pathophysiology and treatment approaches."))
