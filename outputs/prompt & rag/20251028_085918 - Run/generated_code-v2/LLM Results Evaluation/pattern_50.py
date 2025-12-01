import math
import random
import json

class DataIngestion:
    def __init__(self, segment_size=500, overlap=100):
        self.segment_size = segment_size
        self.overlap = overlap

    def load_and_segment_data(self, filepath):
        passages = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            # Simple text segmentation
            for i in range(0, len(content), self.segment_size - self.overlap):
                segment = content[i : i + self.segment_size]
                if segment:
                    passages.append(segment)
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
        return passages

class MockLLM:
    def __init__(self, model_name="mock_llm"):
        self.model_name = model_name

    def generate_qa(self, passage):
        # Simulate QA generation
        question_templates = [
            f"What is discussed about {passage[:50]}...?",
            f"Can you explain the main point regarding {passage[50:100]}...?",
            f"According to the text, what is relevant to {passage[10:60]}...?"
        ]
        answer_templates = [
            f"The passage states that {passage[:70]}...",
            f"It covers details about {passage[70:140]}...",
            f"Key information includes {passage[140:210]}..."
        ]
        q_idx = random.randint(0, len(question_templates) - 1)
        a_idx = random.randint(0, len(answer_templates) - 1)
        return {
            "question": question_templates[q_idx],
            "answer": answer_templates[a_idx],
            "original_passage": passage
        }

    def regenerate_question(self, answer):
        # Simulate question regeneration from answer
        return f"Based on '{answer[:50]}...', what was the question?"

class MockEmbeddingModel:
    def __init__(self, vector_dim=384): # Typical dimension for sentence-transformers
        self.vector_dim = vector_dim

    def encode(self, text):
        # Simulate embedding generation
        return [random.uniform(-1, 1) for _ in range(self.vector_dim)]

class VectorStore:
    def __init__(self, embedding_model):
        self.passages = []
        self.passage_embeddings = []
        self.embedding_model = embedding_model

    def add_passage(self, passage, passage_id):
        embedding = self.embedding_model.encode(passage)
        self.passages.append({"id": passage_id, "text": passage, "embedding": embedding})
        self.passage_embeddings.append(embedding)

    def _cosine_similarity(self, vec1, vec2):
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(v**2 for v in vec1))
        magnitude2 = math.sqrt(sum(v**2 for v in vec2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def retrieve_passage(self, query_embedding, top_k=1):
        similarities = []
        for i, passage_data in enumerate(self.passages):
            sim = self._cosine_similarity(query_embedding, passage_data["embedding"])
            similarities.append((sim, i))
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for sim, idx in similarities[:top_k]:
            results.append({"passage": self.passages[idx]["text"], "similarity": sim})
        return results

class RoundTripConsistencyFilter:
    def __init__(self, embedding_model, qa_generator_llm, similarity_threshold=0.7):
        self.embedding_model = embedding_model
        self.qa_generator_llm = qa_generator_llm
        self.similarity_threshold = similarity_threshold

    def _cosine_similarity(self, vec1, vec2):
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(v**2 for v in vec1))
        magnitude2 = math.sqrt(sum(v**2 for v in vec2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def check_question_regeneration(self, original_question, generated_answer):
        regenerated_question = self.qa_generator_llm.regenerate_question(generated_answer)
        original_q_emb = self.embedding_model.encode(original_question)
        regenerated_q_emb = self.embedding_model.encode(regenerated_question)
        similarity = self._cosine_similarity(original_q_emb, regenerated_q_emb)
        return similarity >= self.similarity_threshold, similarity

    def check_passage_retrieval(self, generated_question, original_answer_text, original_passage_text, vector_store):
        query_embedding = self.embedding_model.encode(generated_question)
        retrieved_passages_info = vector_store.retrieve_passage(query_embedding, top_k=1)

        if not retrieved_passages_info:
            return False, 0.0

        most_similar_retrieved_passage = retrieved_passages_info[0]["passage"]
        
        passage_similarity = self._cosine_similarity(
            self.embedding_model.encode(original_passage_text),
            self.embedding_model.encode(most_similar_retrieved_passage)
        )
        answer_in_retrieved = original_answer_text.lower() in most_similar_retrieved_passage.lower()
        
        is_consistent = passage_similarity >= self.similarity_threshold and answer_in_retrieved
        
        return is_consistent, passage_similarity # Return passage similarity as the metric

def main():
    # Configuration
    data_filepath = "medical_data.txt" # Placeholder for your medical data
    output_filepath = "filtered_medical_qa.json"
    similarity_threshold = 0.65 # Threshold for consistency checks

    # 1. Data Ingestion
    data_ingestion = DataIngestion()
    # Create a dummy medical_data.txt file for demonstration
    dummy_medical_data = """
    Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain from conditions such as muscle aches, toothaches, common cold, and headaches. It may also be used to reduce pain and swelling in conditions such as arthritis. Aspirin is known for its antiplatelet effects, which means it can prevent blood clots. For this reason, it is sometimes prescribed to prevent heart attacks and strokes in high-risk individuals. However, long-term use should be discussed with a doctor due to potential side effects like stomach upset or bleeding.
    Diabetes mellitus, commonly known as diabetes, is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored or used for energy. With diabetes, your body either doesn't make enough insulin or can't effectively use the insulin it does make. Untreated high blood sugar from diabetes can damage your nerves, eyes, kidneys, and other organs. Regular monitoring of blood glucose levels, a balanced diet, and exercise are crucial for managing diabetes. In some cases, medication or insulin therapy may be required.
    """
    with open(data_filepath, "w", encoding="utf-8") as f:
        f.write(dummy_medical_data)

    medical_passages = data_ingestion.load_and_segment_data(data_filepath)
    print(f"Loaded {len(medical_passages)} passages.")

    # Initialize components
    mock_embedding_model = MockEmbeddingModel()
    mock_qa_generator_llm = MockLLM()
    vector_store = VectorStore(mock_embedding_model)
    consistency_filter = RoundTripConsistencyFilter(mock_embedding_model, mock_qa_generator_llm, similarity_threshold)

    # Add original passages to vector store for retrieval check
    for i, passage in enumerate(medical_passages):
        vector_store.add_passage(passage, i)

    synthetic_qa_pairs = []
    filtered_qa_pairs = []

    # 2. Synthetic QA Generation & 3. Round-Trip Consistency Filtering
    print("Generating and filtering QA pairs...")
    for passage in medical_passages:
        generated_qa = mock_qa_generator_llm.generate_qa(passage)
        synthetic_qa_pairs.append(generated_qa)

        # a. Question Regeneration Check
        q_regen_consistent, q_regen_sim = consistency_filter.check_question_regeneration(
            generated_qa["question"], generated_qa["answer"]
        )

        # b. Passage Retrieval Check
        p_retrieval_consistent, p_retrieval_sim = consistency_filter.check_passage_retrieval(
            generated_qa["question"], generated_qa["answer"], generated_qa["original_passage"], vector_store
        )

        # c. Apply filtering criteria
        if q_regen_consistent and p_retrieval_consistent:
            filtered_qa_pairs.append({
                "question": generated_qa["question"],
                "answer": generated_qa["answer"],
                "original_passage": generated_qa["original_passage"],
                "q_regen_similarity": q_regen_sim,
                "p_retrieval_similarity": p_retrieval_sim
            })


    print(f"Generated {len(synthetic_qa_pairs)} synthetic QA pairs.")
    print(f"Accepted {len(filtered_qa_pairs)} high-quality QA pairs.")

    # 4. Filtering Logic and Data Storage
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(filtered_qa_pairs, f, indent=4)
    print(f"Filtered QA pairs saved to {output_filepath}")

if __name__ == "__main__":
    main()