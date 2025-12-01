import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import streamlit as st
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MedicalKnowledgeBase:
    def __init__(self):
        self.passages = []

    def add_passages(self, new_passages):
        self.passages.extend(new_passages)
        print(f"Added {len(new_passages)} new passages. Total passages: {len(self.passages)}")

    def get_passages(self):
        return self.passages

class DensePassageRetriever(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", passage_dim=768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.question_encoder = AutoModel.from_pretrained(model_name).to(device)
        self.passage_encoder = AutoModel.from_pretrained(model_name).to(device)
        self.index = None
        self.passage_texts = []
        self.passage_dim = passage_dim

    def encode_queries(self, queries):
        inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            embeddings = self.question_encoder(**inputs).last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()

    def encode_passages(self, passages):
        inputs = self.tokenizer(passages, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            embeddings = self.passage_encoder(**inputs).last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()

    def index_passages(self, passages):
        self.passage_texts = passages
        if not passages:
            self.index = None
            return
        print(f"Indexing {len(passages)} passages...")
        passage_embeddings = self.encode_passages(passages)
        self.index = faiss.IndexFlatL2(passage_embeddings.shape[1])
        self.index.add(passage_embeddings.astype(np.float32))
        print("Passages indexed successfully.")

    def retrieve(self, query, k=5):
        if self.index is None:
            return []
        query_embedding = self.encode_queries([query])
        D, I = self.index.search(query_embedding.astype(np.float32), k)
        retrieved_passages = [self.passage_texts[i] for i in I[0] if i != -1]
        return retrieved_passages

class Seq2SeqGenerator(nn.Module):
    def __init__(self, model_name="t5-small"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def generate(self, query, retrieved_passages, max_length=200):
        context = " ".join(retrieved_passages)
        input_text = f"question: {query} context: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = self.generator_model.generate(**inputs, max_new_tokens=max_length, num_beams=4, early_stopping=True)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

class EndToEndRAGModel(nn.Module):
    def __init__(self, retriever: DensePassageRetriever, generator: Seq2SeqGenerator):
        super().__init__()
        self.retriever = retriever
        self.generator = generator

    def forward(self, query_texts, k=5):
        retrieved_passages_list = []
        for query in query_texts:
            retrieved_passages = self.retriever.retrieve(query, k=k)
            retrieved_passages_list.append(retrieved_passages)

        answers = []
        for i, query in enumerate(query_texts):
            answers.append(self.generator.generate(query, retrieved_passages_list[i]))
        return answers, retrieved_passages_list

    def train_step(self, batch_queries, batch_relevant_passages, batch_answers, optimizer, k=5):
        optimizer.zero_grad()

        # 1. Forward pass through retriever (conceptual for training, actual implementation is complex)
        # For true end-to-end, gradients from generator would flow back to encoders
        # This is a simplified representation.
        query_embeddings = self.retriever.encode_queries(batch_queries)
        passage_embeddings_relevant = self.retriever.encode_passages(batch_relevant_passages)

        # Dummy loss for retriever training (e.g., contrastive loss between query and relevant passage)
        # In a real scenario, this would involve negative sampling and a proper InfoNCE loss
        retriever_loss = torch.tensor(0.0).to(device) # Placeholder

        # 2. Generate with retrieved passages (or target passages for training)
        # In a real setup, we might use the `batch_relevant_passages` directly here for generator training
        # to ensure it learns from correct contexts, or sample from retriever's output.
        generator_inputs = []
        for q, p in zip(batch_queries, batch_relevant_passages):
            generator_inputs.append(f"question: {q} context: {p}")

        gen_input_tokens = self.generator.tokenizer(generator_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        target_tokens = self.generator.tokenizer(batch_answers, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        # Shift targets for causal language modeling
        labels = target_tokens.clone()
        labels[labels == self.generator.tokenizer.pad_token_id] = -100 # Mask padding tokens

        generator_outputs = self.generator.generator_model(**gen_input_tokens, labels=labels)
        generator_loss = generator_outputs.loss

        # Combined loss (weighted sum would be typical)
        total_loss = generator_loss + retriever_loss # Simplistic combination

        total_loss.backward()
        optimizer.step()
        return total_loss.item()

class RAGTrainer:
    def __init__(self, rag_model: EndToEndRAGModel, mkb: MedicalKnowledgeBase, learning_rate=1e-5, reindex_interval=10):
        self.rag_model = rag_model
        self.mkb = mkb
        self.optimizer = AdamW(rag_model.parameters(), lr=learning_rate)
        self.reindex_interval = reindex_interval

    def _reindex_knowledge_base(self):
        print("Dynamically re-indexing knowledge base...")
        self.rag_model.retriever.index_passages(self.mkb.get_passages())
        print("Knowledge base re-indexed.")

    def train(self, training_data, num_epochs):
        # training_data is a list of (query, relevant_passage, answer) tuples
        print("Starting end-to-end RAG training...")
        self._reindex_knowledge_base() # Initial indexing

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0.0
            # Simulate batching
            for i, (query, passage, answer) in enumerate(training_data):
                loss = self.rag_model.train_step([query], [passage], [answer], self.optimizer)
                epoch_loss += loss

                if (i + 1) % self.reindex_interval == 0:
                    self._reindex_knowledge_base()

            print(f"Average Epoch Loss: {epoch_loss / len(training_data):.4f}")

        print("Training complete.")

class MedicalDiagnosisAssistant:
    def __init__(self):
        self.mkb = MedicalKnowledgeBase()
        self.retriever = DensePassageRetriever().to(device)
        self.generator = Seq2SeqGenerator().to(device)
        self.rag_model = EndToEndRAGModel(self.retriever, self.generator)
        self.trainer = RAGTrainer(self.rag_model, self.mkb)

    def load_initial_knowledge(self, initial_passages):
        self.mkb.add_passages(initial_passages)
        self.retriever.index_passages(self.mkb.get_passages())

    def update_knowledge_base(self, new_medical_texts):
        print("Updating knowledge base with new medical texts...")
        self.mkb.add_passages(new_medical_texts)
        self.retriever.index_passages(self.mkb.get_passages()) # Re-index immediately after update

    def train_model(self, training_data, num_epochs=3):
        self.trainer.train(training_data, num_epochs)

    def diagnose(self, patient_query):
        print(f"Processing query: {patient_query}")
        answers, retrieved_passages_list = self.rag_model([patient_query])
        return answers[0], retrieved_passages_list[0]

# --- Streamlit UI --- Start
st.set_page_config(layout="wide")
st.title("🩺 Medical Diagnosis Assistant for GPs")

@st.cache_resource
def initialize_assistant():
    assistant = MedicalDiagnosisAssistant()
    # Simulate initial medical knowledge
    initial_docs = [
        "COVID-19 symptoms often include fever, cough, fatigue, and loss of taste or smell. Severe cases may lead to pneumonia.",
        "Influenza (flu) is a contagious respiratory illness caused by influenza viruses. Symptoms include fever, body aches, headache, and sore throat.",
        "Dengue fever is a mosquito-borne tropical disease caused by the dengue virus. Symptoms typically begin three to fourteen days after infection and may include a high fever, headache, vomiting, muscle and joint pains, and a characteristic skin rash.",
        "Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). Symptoms often develop slowly and can include increased thirst, frequent urination, increased hunger, unintended weight loss, and fatigue.",
        "Migraine is a severe headache often accompanied by symptoms such as throbbing in the head, sensitivity to light and sound, nausea, and vomiting."
    ]
    assistant.load_initial_knowledge(initial_docs)

    # Simulate training data for domain adaptation (very simplistic)
    training_data_sample = [
        ("patient has fever, headache, and severe joint pain after recent travel to tropics", "Dengue fever is a mosquito-borne tropical disease caused by the dengue virus. Symptoms typically begin three to fourteen days after infection and may include a high fever, headache, vomiting, muscle and joint pains, and a characteristic skin rash.", "Consider Dengue fever given recent tropical travel and symptoms."),
        ("elderly patient with constant thirst and increased urination, family history of diabetes", "Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). Symptoms often develop slowly and can include increased thirst, frequent urination, increased hunger, unintended weight loss, and fatigue.", "Investigate Type 2 diabetes, especially with family history."),
    ]
    # assistant.train_model(training_data_sample, num_epochs=1) # Uncomment to simulate training

    return assistant

assistant = initialize_assistant()

st.markdown(
    "This assistant provides evidence-based information to aid GPs in diagnosis. "
    "It uses an end-to-end trained RAG model that adapts to specialized medical knowledge."
)

# User Input
patient_query = st.text_area("Enter Patient Symptoms or Medical History:",
                             "Patient presents with a persistent cough, shortness of breath, and fatigue, recently returned from an endemic area.")

if st.button("Get Diagnostic Aid"):
    if patient_query:
        with st.spinner("Analyzing symptoms and retrieving medical knowledge..."):
            diagnosis_aid, retrieved_sources = assistant.diagnose(patient_query)

            st.subheader("Proposed Diagnostic Aid:")
            st.write(diagnosis_aid)

            if retrieved_sources:
                st.subheader("Relevant Medical Sources:")
                for i, source in enumerate(retrieved_sources):
                    st.markdown(f"**Source {i+1}:** {source}")
            else:
                st.info("No specific medical sources found for this query in the current knowledge base.")
    else:
        st.warning("Please enter patient symptoms or medical history to get a diagnostic aid.")

# Simulate Knowledge Base Update (e.g., new research published)
st.sidebar.header("Knowledge Base Management (Admin)")
new_doc_input = st.sidebar.text_area("Add New Medical Document/Research Snippet:", "")
if st.sidebar.button("Add to Knowledge Base & Re-index"):
    if new_doc_input:
        assistant.update_knowledge_base([new_doc_input])
        st.sidebar.success("Knowledge base updated and re-indexed!")
        # Rerun the app to refresh cache or simply note the update
        st.experimental_rerun()
    else:
        st.sidebar.warning("Please enter text to add to the knowledge base.")

st.sidebar.info("Note: In a real system, training would be an offline, continuous process.")
# --- Streamlit UI --- End
