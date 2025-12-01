from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

class Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer()
        self.corpus_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query, top_n=3):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.corpus_vectors).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        return [self.corpus[i] for i in top_indices]

class Reader:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_answer(self, question, passages, top_k=2):
        all_sentences = []
        for passage in passages:
            sentences = [s.strip() for s in passage.split('.') if s.strip()]
            all_sentences.extend(sentences)
        
        if not all_sentences:
            return "No relevant sentences found to generate an answer."

        question_embedding = self.model.encode(question, convert_to_tensor=True)
        sentence_embeddings = self.model.encode(all_sentences, convert_to_tensor=True)

        cosine_scores = util.cos_sim(question_embedding, sentence_embeddings)[0]
        top_k_indices = cosine_scores.argsort(descending=True)[:top_k]
        
        top_k_sentences = [all_sentences[i] for i in top_k_indices]
        return ". ".join(top_k_sentences) + "."

# 1. Knowledge Base (Corpus)
medical_corpus = [
    "Influenza, commonly known as the flu, is an infectious disease caused by influenza viruses. Symptoms can range from mild to severe and often include fever, runny nose, sore throat, muscle pains, headache, coughing, and fatigue.",
    "The common cold is a viral infectious disease of the upper respiratory tract that primarily affects the nose. Symptoms include coughing, sore throat, runny nose, sneezing, and fever, which is usually mild.",
    "Paracetamol (acetaminophen) is a medication used to treat pain and fever. It is commonly sold under brand names such as Tylenol. Overdoses can cause liver damage.",
    "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for treating pain, fever, and inflammation. It is often used for headaches, dental pain, and arthritis.",
    "Diabetes mellitus, commonly known as diabetes, is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells for storage or energy.",
    "Hypertension, or high blood pressure, is a condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Lifestyle changes are often the first line of treatment.",
    "A recent study published in the New England Journal of Medicine suggests a new therapeutic approach for managing Type 2 Diabetes involving SGLT2 inhibitors.",
    "Clinical guidelines for managing hypertension often emphasize dietary changes, regular exercise, and medication such as ACE inhibitors or calcium channel blockers."
]

# 2. Initialize Retriever and Reader
retriever = Retriever(medical_corpus)
reader = Reader()

# 3. User Question
medical_question = "What are the symptoms of flu and how is it different from common cold?"

print(f"Question: {medical_question}\n")

# 4. Retrieval Stage
print("Retrieving relevant passages...")
retrieved_passages = retriever.retrieve(medical_question, top_n=2)
print("Retrieved Passages:")
for i, passage in enumerate(retrieved_passages):
    print(f"  Passage {i+1}: {passage}")
print("\n")

# 5. Reading/Answer Generation Stage
print("Generating answer from retrieved passages...")
answer = reader.generate_answer(medical_question, retrieved_passages, top_k=2)

# 6. Present Final Answer
print("Final Answer:")
print(answer)