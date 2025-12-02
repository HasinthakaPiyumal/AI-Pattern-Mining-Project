class MedicalKnowledgeBase:
    """
    Simulates an external, non-parametric medical knowledge base.
    In a real application, this would involve a vector database (e.g., Faiss, Chroma, Pinecone)
    and document embedding using models like Sentence Transformers.
    """
    def __init__(self):
        self.documents = []
        # In a real system, documents would be pre-embedded here.
        self.document_embeddings = [] # Placeholder for actual embeddings

    def add_document(self, text, source="Unknown"):
        """Adds a document to the knowledge base."""
        self.documents.append({"text": text, "source": source})
        # Simulate embedding generation (in a real scenario, an embedding model would be used)
        # For simplicity, we'll just store the text for "retrieval" in this example
        # self.document_embeddings.append(self._generate_embedding(text))

    def _generate_embedding(self, text):
        """
        Placeholder for generating document embeddings.
        In a real system, this would use a pre-trained embedding model
        (e.g., from sentence_transformers library).
        """
        # A very basic, non-semantic "embedding" for demonstration
        return [ord(c) for c in text[:10]] # Example: sum of char codes for first 10 chars

    def retrieve_documents(self, query, top_k=3):
        """
        Retrieves the most relevant documents based on the query.
        In a real application, this would involve:
        1. Embedding the query.
        2. Performing a similarity search (e.g., cosine similarity) against document embeddings.
        3. Returning top_k documents.
        """
        print(f"DEBUG: Retrieving documents for query: '{query}'")
        # For this simplified example, we'll just do a keyword-based search
        # or return a subset if the query is very basic.
        # A proper implementation would use vector similarity.

        relevant_docs = []
        query_words = set(query.lower().split())

        for doc_id, doc in enumerate(self.documents):
            doc_text_lower = doc["text"].lower()
            if any(word in doc_text_lower for word in query_words if len(word) > 2): # Simple keyword match
                relevant_docs.append(doc)
            if len(relevant_docs) >= top_k:
                break
        
        if not relevant_docs and self.documents: # If no keyword match, return some default or first ones
            print("DEBUG: No direct keyword match, returning top documents.")
            return self.documents[:top_k] # Fallback to top_k if no specific match

        return relevant_docs


class LLMService:
    """
    Simulates a Generative Language Model (LLM), representing parametric memory.
    In a real application, this would interact with an actual LLM API (e.g., OpenAI, Gemini, Hugging Face models).
    """
    def generate_answer(self, query, context_documents):
        """
        Generates an answer based on the query and provided context documents.
        """
        print(f"DEBUG: LLM generating answer with query: '{query}' and context: {len(context_documents)} docs")

        if not context_documents:
            # Fallback if no context is provided
            return (f"I'm sorry, I couldn't find specific information regarding '{query}' in my knowledge base. "
                    f"However, as a general language model, I can tell you that...")

        context_text = "\n\n".join([f"Source ({doc['source']}): {doc['text']}" for doc in context_documents])
        
        # Simulate LLM's capability to synthesize information
        # In a real LLM, a prompt engineering approach would be used
        # e.g., "Based on the following context: [CONTEXT], answer the question: [QUERY]"
        
        simulated_response = (
            f"Based on the provided medical context and my general knowledge, here's an insight for your query '{query}':\n\n"
            f"**Synthesis:** {query} is a critical topic in medicine. While I cannot provide medical advice, "
            f"the retrieved information suggests that understanding its nuances from reliable sources is key. "
            f"For instance, one document mentions [extract a key phrase from a doc, or simulate synthesis].\n\n"
            f"**Relevant Information from Sources:**\n"
        )
        for i, doc in enumerate(context_documents):
            simulated_response += f"- Source {doc['source']}: \"{doc['text'][:150]}...\"\n" # Truncate for brevity
        
        simulated_response += "\n**Disclaimer:** This information is for educational purposes only and not medical advice."
        return simulated_response


class ClinicalInsightAssistant:
    """
    Combines parametric (LLM) and non-parametric (Knowledge Base) memory
    to provide factual and grounded medical insights.
    """
    def __init__(self):
        self.knowledge_base = MedicalKnowledgeBase()
        self.llm_service = LLMService()

    def load_medical_data(self):
        """Populates the simulated medical knowledge base."""
        self.knowledge_base.add_document(
            "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce pain, fever, and inflammation. "
            "It is also used as an antiplatelet to prevent blood clots.",
            source="Medical Journal A, 2023"
        )
        self.knowledge_base.add_document(
            "Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels. "
            "Type 1 diabetes is an autoimmune condition, while Type 2 diabetes is often associated with insulin resistance.",
            source="Clinical Guidelines B, 2022"
        )
        self.knowledge_base.add_document(
            "Hypertension, or high blood pressure, is a common condition that can lead to severe health complications "
            "such as heart disease and stroke. Lifestyle modifications and medication are primary treatments.",
            source="Medical Textbook C, 2021"
        )
        self.knowledge_base.add_document(
            "The COVID-19 vaccine works by teaching the immune system to recognize and fight the virus that causes COVID-19. "
            "Different types of vaccines use different approaches, such as mRNA or viral vector technology.",
            source="WHO Report, 2024"
        )
        print("Medical knowledge base loaded with sample data.")

    def ask_medical_question(self, question):
        """
        Processes a medical question by retrieving relevant context and generating an answer.
        """
        print(f"\nUser asked: '{question}'")
        # 1. Retrieve relevant documents from the non-parametric memory
        retrieved_context = self.knowledge_base.retrieve_documents(question)

        # 2. Provide retrieved context to the generative LLM (parametric memory)
        answer = self.llm_service.generate_answer(question, retrieved_context)

        return answer

# Example Usage
if __name__ == "__main__":
    assistant = ClinicalInsightAssistant()
    assistant.load_medical_data()

    # Test questions
    q1 = "What is aspirin used for?"
    print(assistant.ask_medical_question(q1))

    q2 = "Explain hypertension."
    print(assistant.ask_medical_question(q2))
    
    q3 = "How do COVID-19 vaccines work?"
    print(assistant.ask_medical_question(q3))

    q4 = "What is the primary treatment for cancer?" # Question outside loaded context for specific keyword match
    print(assistant.ask_medical_question(q4))
    
    q5 = "Tell me about insulin resistance." # Related to diabetes
    print(assistant.ask_medical_question(q5))