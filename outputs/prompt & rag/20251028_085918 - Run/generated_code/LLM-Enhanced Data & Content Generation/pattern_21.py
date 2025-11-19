from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import gradio as gr
import os

# --- Configuration --- #
# Set your OpenAI API key here or as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- 1. Unified Retrieval and Reasoning (RAG System) --- #

# 1.1. Data Loading and Preprocessing
def load_and_split_documents(data_path: str = "./medical_data/"):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(data_path, filename))
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return splits

# Create a dummy medical data directory and files if they don't exist
if not os.path.exists("./medical_data/"):
    os.makedirs("./medical_data/")
    with open("./medical_data/cardiac_arrest.txt", "w") as f:
        f.write("Cardiac arrest is the abrupt loss of heart function, breathing, and consciousness. The condition results from an electrical disturbance in the heart that disrupts its pumping action, stopping blood flow to the brain and other organs. It's a medical emergency. Survival depends on immediate CPR and defibrillation. Causes can include coronary artery disease, heart attack, electrocution, or drug overdose. Symptoms include sudden collapse, no pulse, no breathing, and loss of consciousness.")
    with open("./medical_data/diabetes_mellitus.txt", "w") as f:
        f.write("Diabetes mellitus, commonly known as diabetes, is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells for storage or energy. With diabetes, your body either doesn't make enough insulin or can't effectively use the insulin it does make. Untreated high blood sugar can damage your nerves, eyes, kidneys, and other organs. There are several types of diabetes, including type 1, type 2, and gestational diabetes. Management often involves diet, exercise, and medication like insulin.")

medical_splits = load_and_split_documents()

# 1.2. Embedding Model
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 1.3. Vector Store (ChromaDB)
vectorstore = Chroma.from_documents(documents=medical_splits, embedding=embeddings_model)
retriever = vectorstore.as_retriever()

# 1.4. LLM Initialization
# Ensure OPENAI_API_KEY is set in your environment or uncomment the line above
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

# 1.5. RAG Chain with personalized prompt handling
def get_rag_chain(user_profile: dict):
    role = user_profile.get("role", "patient")
    detail_level = user_profile.get("detail_level", "medium")

    system_template = (
        f"You are a highly knowledgeable medical assistant. Provide accurate medical information based on the context provided."
        f"User Role: {role}. Expected Detail Level: {detail_level}."
        f"Explain complex terms if the user role is 'patient'. Be concise if 'doctor' or 'researcher' and detail_level is 'low'."
        f"Always cite the provided context by referring to the information (e.g., 'According to the medical data...', 'The document states...')."
        "If the answer is not in the provided context, state that you cannot answer based on the given information."
        "Context: {context}"
    )

    human_template = "Question: {question}"

    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template),
    ])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )
    return qa_chain

# --- 2. Synthetic QA Data Generation --- #

def generate_synthetic_qa(text_segment: str, num_questions: int = 3):
    qa_generation_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are an expert at generating diverse and relevant question-answer pairs from medical texts."
            "Generate {num_questions} unique question-answer pairs based *only* on the following text."
            "Each question should be followed by its answer, clearly separated. Do not add any introductory or concluding remarks."
            "Format: Q: [Question]\nA: [Answer]\nQ: [Question]\nA: [Answer]...\n"
            f"Text: {text_segment}"
        )),
        HumanMessage(content="Generate question-answer pairs.")
    ])
    
    try:
        response = llm.invoke(qa_generation_prompt.format_prompt(num_questions=num_questions, text_segment=text_segment).to_messages())
        qa_pairs_raw = response.content
        
        # Simple parsing for demonstration
        qa_list = []
        lines = qa_pairs_raw.strip().split('\n')
        q, a = None, None
        for line in lines:
            if line.startswith("Q: "):
                if q and a: # Store previous pair before starting new Q
                    qa_list.append({"question": q.replace("Q: ", "").strip(), "answer": a.replace("A: ", "").strip()})
                q = line
                a = None # Reset answer for new question
            elif line.startswith("A: "):
                a = line
        if q and a: # Store the last pair
            qa_list.append({"question": q.replace("Q: ", "").strip(), "answer": a.replace("A: ", "").strip()})

        return qa_list
    except Exception as e:
        return [{"error": str(e)}]

# --- 3. LLMs as Personalized Content Creator --- #

# Simplified user profile management (in-memory for demonstration)
user_profiles = {
    "default": {"role": "patient", "detail_level": "medium"},
    "doctor_profile": {"role": "doctor", "detail_level": "high"},
    "researcher_profile": {"role": "researcher", "detail_level": "high"},
    "patient_simple": {"role": "patient", "detail_level": "low"},
}

current_user_profile = user_profiles["default"]

# Feedback storage (conceptual, for demonstration)
feedback_log = []
def log_feedback(query: str, answer: str, user_profile: dict, feedback: str):
    feedback_log.append({"query": query, "answer": answer, "user_profile": user_profile, "feedback": feedback})
    print(f"Feedback logged: {feedback}")

# --- Gradio Interface --- #

def medical_query(query: str, user_role: str, detail_level: str):
    global current_user_profile
    current_user_profile = {"role": user_role, "detail_level": detail_level}
    
    qa_chain = get_rag_chain(current_user_profile)
    result = qa_chain.invoke({"query": query})
    
    answer = result["result"]
    sources = "\n".join([doc.metadata.get("source", "N/A") for doc in result["source_documents"]])
    
    # Simulate feedback for potential RLHF later
    # In a real app, this would be collected via a UI element
    # log_feedback(query, answer, current_user_profile, "positive" if "good" in query.lower() else "negative")
    
    return answer, sources

def generate_qa_interface(raw_text_input: str, num_q: int = 3):
    if not raw_text_input.strip():
        return "Please provide text to generate QA pairs from."
    
    qa_pairs = generate_synthetic_qa(raw_text_input, num_q)
    
    output_str = ""
    if qa_pairs and "error" in qa_pairs[0]:
        output_str = f"Error: {qa_pairs[0]['error']}"
    else:
        for i, pair in enumerate(qa_pairs):
            output_str += f"Q{i+1}: {pair['question']}\nA{i+1}: {pair['answer']}\n\n"
    return output_str


with gr.Blocks() as demo:
    gr.Markdown("# Personalized Medical Information & Research Assistant")
    
    with gr.Tab("Medical Q&A"):
        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(label="Your Medical Question", placeholder="e.g., What are the symptoms of diabetes?")
                user_role_dropdown = gr.Dropdown(
                    label="Your Role",
                    choices=["patient", "doctor", "researcher"],
                    value="patient"
                )
                detail_level_dropdown = gr.Dropdown(
                    label="Detail Level",
                    choices=["low", "medium", "high"],
                    value="medium"
                )
                submit_btn = gr.Button("Get Answer")
            with gr.Column(scale=2):
                answer_output = gr.Textbox(label="Answer", interactive=False, lines=10)
                sources_output = gr.Textbox(label="Sources", interactive=False, lines=3)
        
        submit_btn.click(
            medical_query,
            inputs=[query_input, user_role_dropdown, detail_level_dropdown],
            outputs=[answer_output, sources_output]
        )
    
    with gr.Tab("Synthetic QA Data Generation"): 
        gr.Markdown("### Generate Question-Answer Pairs from Raw Text")
        qa_text_input = gr.Textbox(label="Raw Medical Text", lines=10, placeholder="Paste medical text here...")
        num_q_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Number of QA pairs to generate")
        generate_qa_btn = gr.Button("Generate QA Pairs")
        generated_qa_output = gr.Textbox(label="Generated QA Pairs", interactive=False, lines=15)
        
        generate_qa_btn.click(
            generate_qa_interface,
            inputs=[qa_text_input, num_q_slider],
            outputs=generated_qa_output
        )

demo.launch()
