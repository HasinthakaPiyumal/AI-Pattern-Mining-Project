from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Data Layer (Simulated Historical Tickets) ---
historical_tickets_data = [
    {
        "query": "My internet is not working. I have rebooted the router multiple times.",
        "solution": "Please check if the Ethernet cable is securely connected. If still not working, log into your router settings to check connection status. If the issue persists, contact our technical support for a line check."
    },
    {
        "query": "How do I reset my password for my account?",
        "solution": "Go to the login page, click on 'Forgot Password', enter your registered email, and follow the instructions sent to your email to reset it."
    },
    {
        "query": "My recent order #12345 has not shipped yet. What's the status?",
        "solution": "Orders typically ship within 2-3 business days. You can track your order status on your account page under 'My Orders'. For specific details on order #12345, please provide your full name and email for verification."
    },
    {
        "query": "I want to upgrade my current data plan. What are my options?",
        "solution": "You can view available data plan upgrades on our website under 'My Services' or by contacting our sales team directly for personalized recommendations."
    }
]

# --- 2. Embedding Model ---
# Using HuggingFaceEmbeddings for compatibility with Langchain
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# --- 3. Language Models (LLMs) ---
# ExpertCoTGeneratorLLM (Simulated for this example)
def generate_cot_explanation(ticket_query, ticket_solution):
    return (
        f"Thinking Process:\n"  # Simulate Chain-of-Thought
        f"1. Analyze the customer query: '{ticket_query}'\n"
        f"2. Identify key problem areas or user intent (e.g., connectivity issue, password reset, order status, plan upgrade).\n"
        f"3. Formulate a step-by-step solution based on common troubleshooting or service procedures.\n"
        f"4. Include verification steps or alternative contact methods if the initial solution fails.\n"
        f"5. Ensure the solution directly addresses the query."
        f"\nActual Solution: {ticket_solution}"
    )

# PrimarySupportLLM (Using a basic transformers pipeline for text generation)
try:
    primary_llm_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    primary_llm_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    primary_support_llm_pipeline = pipeline(
        "text-generation",
        model=primary_llm_model,
        tokenizer=primary_llm_tokenizer,
        max_new_tokens=200,
        device=-1 # Use -1 for CPU, 0 for first GPU
    )
    # A wrapper for the pipeline to make it behave like an LLM call for demonstration
    class SimpleLLMWrapper:
        def __init__(self, pipeline_instance):
            self.pipeline = pipeline_instance
        def invoke(self, prompt):
            # The pipeline expects a string directly
            response = self.pipeline(prompt, num_return_sequences=1)[0]['generated_text']
            # Clean up potential prompt echoing from the pipeline
            if response.startswith(prompt):
                return response[len(prompt):].strip()
            return response.strip()
    primary_support_llm = SimpleLLMWrapper(primary_support_llm_pipeline)

except Exception as e:
    print(f"Warning: Could not load distilgpt2 for PrimarySupportLLM. Using a dummy LLM. Error: {e}")
    class DummyLLM:
        def invoke(self, prompt):
            return f"[DUMMY LLM RESPONSE] I'm sorry, I cannot provide a detailed solution. The query was: '{prompt}'. Please try again later."
    primary_support_llm = DummyLLM()


# --- 4. & 5. Retrieval Module & Prompt Engineering (Memory Building Phase) ---
cot_texts = []
for ticket in historical_tickets_data:
    cot_texts.append(generate_cot_explanation(ticket["query"], ticket["solution"]))

# Create a dummy list of metadata for FAISS, as it expects it.
# In a real scenario, this could include original query, solution, etc.
metadatas = [{
    "original_query": ticket["query"],
    "original_solution": ticket["solution"]
} for ticket in historical_tickets_data]

# Initialize FAISS vector store with CoT explanations and their embeddings
# We'll use a temporary FAISS instance for this demo
# For a persistent store, save/load methods would be used.
print("Building CoT Memory Store...")
# Langchain's FAISS.from_texts handles embedding internally if an embeddings object is passed.
vectorstore = FAISS.from_texts(cot_texts, embeddings, metadatas=metadatas)
print("CoT Memory Store Built.")

# --- Retrieval Module ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Retrieve top 2 similar CoTs

# --- Prompt Engineering (Test Time Phase) ---
example_template = (
    "Customer Query: {original_query}\n"
    "Chain of Thought: {page_content}\n"
    "Solution: {original_solution}"
)
example_prompt = PromptTemplate(input_variables=["original_query", "page_content", "original_solution"], template=example_template)

# The final prompt template for the LLM
final_prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        "You are an intelligent customer support assistant. Use the following examples to guide your thought process and provide a comprehensive solution to the new customer query.\n\n"
        "{context}\n\n"
        "New Customer Query: {query}\n"
        "Chain of Thought: Let's think step by step to provide the best solution.\n"
        "Solution:"
    )
)

# FewShotPromptTemplate to dynamically select examples
few_shot_prompt = FewShotPromptTemplate(
    example_selector=retriever,
    example_prompt=example_prompt,
    prefix="Here are some examples of customer queries and their solutions with detailed thinking processes:",
    suffix="",
    input_variables=["query"], # This will be passed by the main chain
    example_separator="\n\n---\n\n"
)

# --- Orchestration (Langchain Chain) ---
# This chain will first retrieve examples, then format them, then pass to the final prompt.
chain = (
    {"context": few_shot_prompt, "query": RunnablePassthrough()}
    | final_prompt_template
    | primary_support_llm
    | StrOutputParser()
)


# --- Main Execution (Demonstration) ---
if __name__ == "__main__":
    print("\n--- Starting Customer Support Assistant ---")

    new_customer_query_1 = "I can't log into my account, and I've tried resetting my password, but I'm not getting the email."
    print(f"\nCustomer Query: {new_customer_query_1}")
    response_1 = chain.invoke(new_customer_query_1)
    print(f"Assistant Solution:\n{response_1}")

    print("\n" + "="*80 + "\n")

    new_customer_query_2 = "My internet is extremely slow, and I can barely browse websites. What should I do?"
    print(f"Customer Query: {new_customer_query_2}")
    response_2 = chain.invoke(new_customer_query_2)
    print(f"Assistant Solution:\n{response_2}")

    print("\n" + "="*80 + "\n")

    new_customer_query_3 = "I need to know my order status for a recent purchase. My order number is 67890."
    print(f"Customer Query: {new_customer_query_3}")
    response_3 = chain.invoke(new_customer_query_3)
    print(f"Assistant Solution:\n{response_3}")

    print("\n--- Customer Support Assistant Finished ---")
