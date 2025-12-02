import uuid

# 1. Nonparametric Knowledge Base (KnowledgeBase Module)
class KnowledgeBase:
    def __init__(self):
        self.chunks = []

    def add_document_chunk(self, content, source, title):
        chunk_id = str(uuid.uuid4())
        chunk = {
            "id": chunk_id,
            "content": content,
            "source": source,
            "title": title
        }
        self.chunks.append(chunk)
        return chunk_id

    def update_document_chunk(self, chunk_id, new_content):
        for chunk in self.chunks:
            if chunk["id"] == chunk_id:
                chunk["content"] = new_content
                return True
        return False

    def get_document_chunk(self, chunk_id):
        for chunk in self.chunks:
            if chunk["id"] == chunk_id:
                return chunk
        return None

    def delete_document_chunk(self, chunk_id):
        initial_len = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk["id"] != chunk_id]
        return len(self.chunks) < initial_len

    def get_all_chunks(self):
        return self.chunks

# 2. Retrieval Mechanism (Retriever Module)
class Retriever:
    def retrieve_relevant_chunks(self, query, knowledge_base_instance, top_n=3):
        query_keywords = set(word.lower() for word in query.split() if len(word) > 2)
        relevant_chunks_with_scores = []

        for chunk in knowledge_base_instance.get_all_chunks():
            chunk_content_lower = chunk["content"].lower()
            score = sum(1 for keyword in query_keywords if keyword in chunk_content_lower)
            if score > 0:
                relevant_chunks_with_scores.append((score, chunk))

        relevant_chunks_with_scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in relevant_chunks_with_scores[:top_n]]

# 3. Language Model (LLM) Integration (LLM_Integration Module)
class LLM_Integration:
    def simulate_llm_response(self, patient_query, retrieved_context):
        diagnosis_suggestions = []
        recommendations = []
        cited_evidence = []

        if not retrieved_context:
            return {
                "diagnosis": "No relevant information found in the knowledge base.",
                "recommendations": "Please consult a medical professional for further evaluation.",
                "cited_evidence": []
            }

        for chunk in retrieved_context:
            cited_evidence.append(f"ID: {chunk['id']}, Source: {chunk['source']}, Title: {chunk['title']}")
            if "fever" in patient_query.lower() and "infection" in chunk["content"].lower():
                diagnosis_suggestions.append("Possible Viral Infection")
                recommendations.append("Rest and hydration, consider over-the-counter fever reducers.")
            if "headache" in patient_query.lower() and "migraine" in chunk["content"].lower():
                diagnosis_suggestions.append("Potential Migraine")
                recommendations.append("Avoid triggers, pain relievers, consult for specific migraine medication.")
            if "rash" in patient_query.lower() and "allergic reaction" in chunk["content"].lower():
                diagnosis_suggestions.append("Likely Allergic Reaction")
                recommendations.append("Identify allergen, antihistamines, topical creams.")
            if "diabetes" in chunk["title"].lower() or "blood sugar" in chunk["content"].lower():
                 diagnosis_suggestions.append("Diabetes management considerations")
                 recommendations.append("Monitor blood sugar, dietary adjustments, medication review.")

        if not diagnosis_suggestions:
            diagnosis_suggestions.append("Further investigation needed. The provided context suggests general medical principles.")
            recommendations.append("Review patient full medical history and perform physical examination.")

        # Deduplicate and format
        diagnosis_suggestions = list(set(diagnosis_suggestions))
        recommendations = list(set(recommendations))

        return {
            "diagnosis": "; ".join(diagnosis_suggestions) if diagnosis_suggestions else "Undetermined",
            "recommendations": "; ".join(recommendations) if recommendations else "No specific recommendations based on context.",
            "cited_evidence": cited_evidence
        }


# 5. Data Loader (DataLoader Module)
class DataLoader:
    def load_dummy_data(self):
        dummy_data = [
            {
                "content": "Fever is often a sign of an underlying infection. Common causes include viral illnesses like the flu or common cold, and bacterial infections such as strep throat or urinary tract infections.",
                "source": "CDC Guidelines 2023",
                "title": "Understanding Fever"
            },
            {
                "content": "Migraine headaches are characterized by severe throbbing pain or a pulsating sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound.",
                "source": "WHO Fact Sheet",
                "title": "Migraine Symptoms and Treatment"
            },
            {
                "content": "Allergic reactions can manifest as skin rashes, hives, itching, swelling, or difficulty breathing. Common allergens include pollen, dust mites, certain foods, and insect stings.",
                "source": "Mayo Clinic Articles",
                "title": "Recognizing Allergic Reactions"
            },
            {
                "content": "Diabetes mellitus is a chronic metabolic disease characterized by elevated levels of blood glucose (or blood sugar), which over time leads to serious damage to the heart, blood vessels, eyes, kidneys and nerves.",
                "source": "International Diabetes Federation",
                "title": "What is Diabetes?"
            },
            {
                "content": "Management of type 2 diabetes often involves lifestyle changes such as diet and exercise, and medications like metformin or insulin to control blood sugar levels.",
                "source": "American Diabetes Association",
                "title": "Type 2 Diabetes Management"
            }
        ]
        return dummy_data


# 4. User Interface (Main Application - CLI)
class MedicalDiagnosticAssistant:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.retriever = Retriever()
        self.llm_integration = LLM_Integration()
        self.data_loader = DataLoader()
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        print("Loading initial medical knowledge...")
        dummy_chunks = self.data_loader.load_dummy_data()
        for chunk_data in dummy_chunks:
            self.knowledge_base.add_document_chunk(
                content=chunk_data["content"],
                source=chunk_data["source"],
                title=chunk_data["title"]
            )
        print(f"Loaded {len(self.knowledge_base.get_all_chunks())} knowledge chunks.")

    def display_menu(self):
        print("\n--- Medical Diagnostic Assistant CLI ---")
        print("1. Get Diagnosis and Recommendations")
        print("2. View All Knowledge Chunks")
        print("3. Add New Knowledge Chunk")
        print("4. Update Knowledge Chunk")
        print("5. Delete Knowledge Chunk")
        print("6. Exit")

    def run(self):
        while True:
            self.display_menu()
            choice = input("Enter your choice: ")

            if choice == "1":
                patient_symptoms = input("Enter patient symptoms and medical history: ")
                print("\nRetrieving relevant medical information...")
                retrieved_context = self.retriever.retrieve_relevant_chunks(
                    patient_symptoms, self.knowledge_base
                )

                if retrieved_context:
                    print("Context found. Generating diagnosis...")
                    llm_output = self.llm_integration.simulate_llm_response(
                        patient_symptoms, retrieved_context
                    )
                    print("\n--- Diagnosis Results ---")
                    print(f"Diagnosis: {llm_output['diagnosis']}")
                    print(f"Recommendations: {llm_output['recommendations']}")
                    print("\nCited Evidence:")
                    for evidence in llm_output["cited_evidence"]:
                        print(f"- {evidence}")
                else:
                    print("No relevant medical information found for the given symptoms in the knowledge base.")

            elif choice == "2":
                print("\n--- All Knowledge Chunks ---")
                all_chunks = self.knowledge_base.get_all_chunks()
                if all_chunks:
                    for i, chunk in enumerate(all_chunks):
                        print(f"\nChunk {i+1} (ID: {chunk['id']})")
                        print(f"  Title: {chunk['title']}")
                        print(f"  Source: {chunk['source']}")
                        print(f"  Content: {chunk['content'][:200]}...") # Show first 200 chars
                else:
                    print("Knowledge base is empty.")

            elif choice == "3":
                content = input("Enter new chunk content: ")
                source = input("Enter source (e.g., 'New Research Paper'): ")
                title = input("Enter title (e.g., 'Novel Treatment for X'): ")
                chunk_id = self.knowledge_base.add_document_chunk(content, source, title)
                print(f"New chunk added with ID: {chunk_id}")

            elif choice == "4":
                chunk_id = input("Enter ID of chunk to update: ")
                new_content = input("Enter new content for the chunk: ")
                if self.knowledge_base.update_document_chunk(chunk_id, new_content):
                    print(f"Chunk {chunk_id} updated successfully.")
                else:
                    print(f"Chunk with ID {chunk_id} not found.")

            elif choice == "5":
                chunk_id = input("Enter ID of chunk to delete: ")
                if self.knowledge_base.delete_document_chunk(chunk_id):
                    print(f"Chunk {chunk_id} deleted successfully.")
                else:
                    print(f"Chunk with ID {chunk_id} not found.")

            elif choice == "6":
                print("Exiting Medical Diagnostic Assistant. Goodbye!")
                break

            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    app = MedicalDiagnosticAssistant()
    app.run()