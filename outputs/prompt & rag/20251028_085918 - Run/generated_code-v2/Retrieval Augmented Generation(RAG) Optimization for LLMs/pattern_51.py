import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
import random

def generate_raft_data(num_samples=100, proportion_golden=0.8):
    data = []
    medical_questions = [
        "What are the primary symptoms of type 2 diabetes?",
        "Describe the mechanism of action of ACE inhibitors.",
        "What is the recommended treatment for community-acquired pneumonia in adults?",
        "How does CRISPR-Cas9 work in gene editing?",
        "What are the latest guidelines for hypertension management?"
    ]
    medical_docs = [
        "Type 2 diabetes often presents with increased thirst, frequent urination, fatigue, and blurred vision. It is characterized by insulin resistance and relative insulin deficiency.",
        "ACE inhibitors block the conversion of angiotensin I to angiotensin II, leading to vasodilation, reduced aldosterone secretion, and decreased blood pressure.",
        "For community-acquired pneumonia, empirical antibiotic therapy often involves macrolides or doxycycline. More severe cases may require beta-lactams plus a macrolide.",
        "CRISPR-Cas9 is a gene-editing tool that uses a guide RNA to direct the Cas9 enzyme to a specific DNA sequence, where it creates a double-strand break.",
        "Recent guidelines emphasize lifestyle modifications, with pharmacological treatment initiated for blood pressure >= 130/80 mmHg in high-risk individuals, targeting <130/80 mmHg."
    ]
    distractor_docs = [
        "The capital of France is Paris, a global center for art, fashion, gastronomy and culture.",
        "Quantum computing utilizes quantum-mechanical phenomena such as superposition and entanglement to perform computations.",
        "The history of the internet dates back to the development of packet switching and research commissioned by the United States Department of Defense in the 1960s.",
        "In chess, the queen is the most powerful piece, able to move any number of squares vertically, horizontally, or diagonally.",
        "Photosynthesis is the process used by plants, algae and cyanobacteria to convert light energy into chemical energy, creating glucose and oxygen."
    ]

    for i in range(num_samples):
        question = random.choice(medical_questions)
        golden_doc_candidate = [doc for q, doc in zip(medical_questions, medical_docs) if q == question]
        golden_doc = golden_doc_candidate[0] if golden_doc_candidate else random.choice(medical_docs)
        
        current_docs_for_sample = []
        answer_text = ""

        if random.random() < proportion_golden:
            current_docs_for_sample.append(golden_doc)
            answer_text = f"Based on Document 1: {golden_doc}\nChain-of-Thought: The question asks about {question.lower().replace('?', '')}. Document 1 directly addresses this by stating: '{golden_doc}'. Therefore, the answer is: {golden_doc}"
        else:
            answer_text = f"Based on the provided documents, I cannot find a direct answer to '{question}'."

        num_distractors = random.randint(2, 4)
        available_distractors = [d for d in distractor_docs if d not in current_docs_for_sample]
        selected_distractors = random.sample(available_distractors, min(num_distractors, len(available_distractors)))
        current_docs_for_sample.extend(selected_distractors)

        random.shuffle(current_docs_for_sample)
        
        context_parts = []
        for j, doc in enumerate(current_docs_for_sample):
            context_parts.append(f"Document {j+1}: {doc}")
        context = "\n".join(context_parts)

        data.append({"question": question, "context": context, "answer": answer_text})
    return Dataset.from_list(data)

if __name__ == "__main__":
    model_name = "gpt2"
    output_dir = "./raft_finetuned_model"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    train_dataset = generate_raft_data(num_samples=200, proportion_golden=0.8)

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['question'])):
            text = f"### Question:\n{example['question'][i]}\n\n### Documents:\n{example['context'][i]}\n\n### Answer:\n{example['answer'][i]}{tokenizer.eos_token}"
            output_texts.append(text)
        return output_texts

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        optim="adamw_torch",
        fp16=True,
        report_to="none",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        formatting_func=formatting_prompts_func,
        args=training_args,
        max_seq_length=512,
        peft_config=lora_config,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"RAFT fine-tuned model saved to {output_dir}")


# --- START OF clinical_assistant.py ---

import uvicorn
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import torch

MODEL_PATH = "./raft_finetuned_model"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"

class ClinicalDocumentStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        self.collection = self.client.get_or_create_collection(
            name="medical_docs",
            embedding_function=self.embedding_function
        )
        self._populate_dummy_data()

    def _populate_dummy_data(self):
        dummy_medical_data = [
            {"id": "doc1", "content": "The primary treatment for uncomplicated urinary tract infections is a short course of antibiotics, such as nitrofurantoin or trimethoprim-sulfamethoxazole. Hydration is also important.", "source": "Clinical Guidelines 2023"},
            {"id": "doc2", "content": "Migraine headaches are characterized by throbbing pain, often on one side of the head, accompanied by nausea, vomiting, and sensitivity to light and sound. Triptans are a common acute treatment.", "source": "Neurology Journal Vol. 45"},
            {"id": "doc3", "content": "Hypertension management involves lifestyle modifications (diet, exercise) and pharmacological agents like ACE inhibitors, ARBs, calcium channel blockers, and diuretics. Regular monitoring is crucial.", "source": "Cardiology Review"},
            {"id": "doc4", "content": "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to symptoms like wheezing, shortness of breath, chest tightness, and coughing.", "source": "Pulmonology Text"},
            {"id": "doc5", "content": "The influenza virus is a respiratory pathogen that causes seasonal epidemics. Vaccination is the most effective preventative measure, reducing severity and transmission.", "source": "CDC Report"},
            {"id": "doc6", "content": "Appendicitis is an inflammation of the appendix, typically presenting with abdominal pain that often starts around the navel and shifts to the lower right abdomen. Surgical removal (appendectomy) is the standard treatment.", "source": "Surgical Handbook"},
            {"id": "doc7", "content": "Osteoarthritis is a degenerative joint disease characterized by the breakdown of joint cartilage and underlying bone. Symptoms include joint pain, stiffness, and reduced range of motion.", "source": "Rheumatology Insights"},
            {"id": "doc8", "content": "Depression is a mood disorder causing a persistent feeling of sadness and loss of interest. Treatment often involves psychotherapy, medication (antidepressants), or a combination of both.", "source": "Psychiatry Today"},
        ]

        if self.collection.count() == 0:
            documents = [d["content"] for d in dummy_medical_data]
            metadatas = [{
                "source": d["source"],
                "original_content": d["content"]
            } for d in dummy_medical_data]
            ids = [d["id"] for d in dummy_medical_data]
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print("ChromaDB populated with dummy medical data.")
        else:
            print("ChromaDB already contains data, skipping population.")

    def retrieve_documents(self, query: str, k: int = 3):
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=['documents', 'metadatas']
        )
        retrieved_docs = []
        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                doc_content = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                retrieved_docs.append({"content": doc_content, "source": metadata.get("source", "Unknown")})
        return retrieved_docs

class RetrievalAugmentedClinicalAssistant:
    def __init__(self, model_path: str, tokenizer_path: str, doc_store: ClinicalDocumentStore):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.doc_store = doc_store
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def _construct_prompt(self, question: str, retrieved_docs: list):
        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            context_str += f"Document {i+1} (Source: {doc['source']}): {doc['content']}\n"
        
        prompt = f"### Question:\n{question}\n\n### Documents:\n{context_str}\n\n### Answer:\nBased on the provided documents, answer the question and cite the document number for any facts you use. If the information is not in the documents, state that.\nChain-of-Thought:"
        return prompt

    def generate_answer(self, question: str, k_retrieved_docs: int = 3) -> str:
        retrieved_docs = self.doc_store.retrieve_documents(question, k=k_retrieved_docs)
        prompt = self._construct_prompt(question, retrieved_docs)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        if "### Question" in generated_text:
            generated_text = generated_text.split("### Question")[0].strip()
        if "\n\n### Answer:" in generated_text:
            generated_text = generated_text.split("\n\n### Answer:")[0].strip()

        return generated_text.strip()

app = FastAPI(
    title="Clinical Decision Support Assistant",
    description="AI assistant leveraging RAFT fine-tuned LLM for medical Q&A with document retrieval."
)

chroma_doc_store = ClinicalDocumentStore()
clinical_assistant = RetrievalAugmentedClinicalAssistant(
    model_path=MODEL_PATH,
    tokenizer_path=MODEL_PATH,
    doc_store=chroma_doc_store
)

class QuestionRequest(BaseModel):
    question: str
    k_retrieved_docs: int = 3

@app.post("/ask")
async def ask_medical_question(request: QuestionRequest):
    try:
        answer = clinical_assistant.generate_answer(request.question, request.k_retrieved_docs)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True, "chroma_ready": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
