import json
import ast
import requests
from typing import List, Dict, Any, Optional

import pydantic
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- 1. API Database Module ---

class APIArgument(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True

class APISchema(BaseModel):
    domain: str
    function_name: str
    description: str
    arguments: List[APIArgument]
    return_type: str
    endpoint: str
    method: str

class APIDatabase:
    def __init__(self, db_path: str = "api_database.jsonl"):
        self.db_path = db_path
        self.api_docs: List[APISchema] = []
        self._load_api_docs()

    def _load_api_docs(self):
        try:
            with open(self.db_path, "r") as f:
                for line in f:
                    self.api_docs.append(APISchema.parse_raw(line))
        except FileNotFoundError:
            self.api_docs = []

    def add_api_doc(self, api_schema: APISchema):
        self.api_docs.append(api_schema)
        with open(self.db_path, "a") as f:
            f.write(api_schema.json() + "\n")

    def get_all_api_descriptions(self) -> List[str]:
        return [doc.description for doc in self.api_docs]

    def get_api_by_description(self, description: str) -> Optional[APISchema]:
        for doc in self.api_docs:
            if doc.description == description:
                return doc
        return None

# Placeholder for API Scrapers/Parsers - in a real scenario, this would populate the database
def populate_mock_api_database(api_db: APIDatabase):
    if not api_db.api_docs:
        print("Populating mock API database...")
        # PubMed Search API
        api_db.add_api_doc(APISchema(
            domain="pubmed",
            function_name="search_articles",
            description="Searches for medical articles on PubMed by keyword.",
            arguments=[
                APIArgument(name="query", type="str", description="Keywords for search"),
                APIArgument(name="max_results", type="int", description="Maximum number of results", required=False)
            ],
            return_type="List[Article]",
            endpoint="https://api.pubmed.gov/search",
            method="GET"
        ))
        # ClinicalTrials.gov Search API
        api_db.add_api_doc(APISchema(
            domain="clinicaltrials",
            function_name="find_trials",
            description="Finds clinical trials by condition or drug.",
            arguments=[
                APIArgument(name="condition", type="str", description="Medical condition"),
                APIArgument(name="drug", type="str", description="Drug name", required=False),
                APIArgument(name="phase", type="str", description="Clinical trial phase (e.g., Phase 3)", required=False)
            ],
            return_type="List[Trial]",
            endpoint="https://api.clinicaltrials.gov/search",
            method="GET"
        ))
        # DrugBank Drug Info API
        api_db.add_api_doc(APISchema(
            domain="drugbank",
            function_name="get_drug_info",
            description="Retrieves detailed information about a specific drug.",
            arguments=[
                APIArgument(name="drug_name", type="str", description="Name of the drug")
            ],
            return_type="Dict[str, Any]",
            endpoint="https://api.drugbank.ca/info",
            method="GET"
        ))
        print("Mock API database populated.")

# --- 2. Synthetic Instruction Generation Module ---

# Placeholder for a powerful LLM to generate instructions
class InstructionLLM:
    def generate_instruction_api_pair(self, api_schema: APISchema, constraints: Optional[str] = None) -> Dict[str, str]:
        natural_language_question = f"Tell me about {api_schema.description.lower().replace('searches for ', '').replace('retrieves detailed information about ', '').replace('finds ', '')}"
        if constraints:
            natural_language_question += f" with constraints: {constraints}"

        mock_args = {}
        for arg in api_schema.arguments:
            if arg.name == "query": mock_args[arg.name] = "diabetes treatment"
            elif arg.name == "condition": mock_args[arg.name] = "type 2 diabetes"
            elif arg.name == "drug": mock_args[arg.name] = "metformin"
            elif arg.name == "drug_name": mock_args[arg.name] = "insulin"
            elif arg.name == "max_results": mock_args[arg.name] = 10
            elif arg.name == "phase": mock_args[arg.name] = "Phase 3"
            else: mock_args[arg.name] = "example_value"

        api_call = f"{api_schema.function_name}(**{mock_args})"

        return {
            "question": natural_language_question.replace('searches for medical articles on pubmed by keyword.', 'medical articles about diabetes treatment'),
            "api_call": api_call,
            "api_description": api_schema.description # For retrieval context
        }

def generate_synthetic_data(api_db: APIDatabase, num_samples: int = 20) -> List[Dict[str, str]]:
    instruction_llm = InstructionLLM()
    synthetic_dataset = []
    for _ in range(num_samples):
        for api_schema in api_db.api_docs:
            synthetic_dataset.append(instruction_llm.generate_instruction_api_pair(api_schema))
            if len(synthetic_dataset) >= num_samples: break
        if len(synthetic_dataset) >= num_samples: break
    return synthetic_dataset

# --- 3. Retriever-Aware Finetuning Module ---

# Placeholder for sentence-transformers
class QueryEncoder:
    def encode(self, texts: List[str]) -> List[List[float]]:
        # In a real scenario, this would use a SentenceTransformer model
        # For simplicity, returning mock embeddings (e.g., one-hot or random)
        # A more realistic mock would be a simple hashing or initial character sum.
        return [[float(ord(c)) / 100 for c in text[:10]] + [0.0] * (10 - len(text[:10])) for text in texts]

# Placeholder for a Base LLM and Finetuning process
class FinetunedLLM:
    def __init__(self, model_name: str = "mock-llama-7b"): # Using a mock name
        self.model_name = model_name

    def generate_api_call_and_explanation(self, query: str, retrieved_api_docs: List[APISchema]) -> Dict[str, str]:
        # This is a highly simplified mock. In reality, the LLM would analyze
        # the query and retrieved docs to generate the API call.
        print(f"LLM received query: '{query}' and retrieved docs: {[d.description for d in retrieved_api_docs]}")

        # Simple heuristic to mock API call generation
        if "pubmed" in query.lower() or "articles" in query.lower():
            api_name = "search_articles"
            args = {"query": query.replace("search pubmed for", "").strip(), "max_results": 5}
            explanation = f"Searching PubMed for articles related to '{args['query']}'."
        elif "clinical trials" in query.lower() or "trials for" in query.lower():
            api_name = "find_trials"
            condition = query.split("trials for")[-1].strip()
            args = {"condition": condition}
            if "phase 3" in query.lower():
                args["phase"] = "Phase 3"
            explanation = f"Looking for clinical trials for '{condition}'."
        elif "drug info" in query.lower() or "about drug" in query.lower():
            api_name = "get_drug_info"
            drug_name = query.split("about drug")[-1].strip()
            args = {"drug_name": drug_name}
            explanation = f"Retrieving detailed information about the drug '{drug_name}'."
        else:
            api_name = "unknown_api"
            args = {}
            explanation = "Could not determine a precise API call for your query."

        return {
            "api_call": f"{api_name}(**{args})" if api_name != "unknown_api" else "",
            "explanation": explanation
        }

def finetune_llm_placeholder(base_llm: FinetunedLLM, dataset: List[Dict[str, str]], query_encoder: QueryEncoder, api_db: APIDatabase):
    print(f"Simulating finetuning of {base_llm.model_name} with {len(dataset)} samples...")
    # In a real setup, this would involve loading models with transformers, applying LoRA with peft,
    # and training with trl/accelerate, incorporating retrieved docs during training.
    print("Finetuning simulation complete.")
    return base_llm # Return the (mock) finetuned LLM

# --- 4. Inference and Execution System ---

# Placeholder for Vector Database (ChromaDB/FAISS)
class VectorDatabase:
    def __init__(self, api_db: APIDatabase, query_encoder: QueryEncoder):
        self.api_db = api_db
        self.query_encoder = query_encoder
        self.api_descriptions = self.api_db.get_all_api_descriptions()
        self.api_embeddings = self.query_encoder.encode(self.api_descriptions)

    def retrieve_top_k(self, query: str, k: int = 3) -> List[APISchema]:
        query_embedding = self.query_encoder.encode([query])[0]
        
        # Simple dot product for similarity (mocking vector similarity)
        similarities = []
        for i, api_emb in enumerate(self.api_embeddings):
            similarity = sum(q_val * a_val for q_val, a_val in zip(query_embedding, api_emb))
            similarities.append((similarity, i))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        top_k_indices = [idx for sim, idx in similarities[:k]]
        
        retrieved_schemas = []
        for idx in top_k_indices:
            schema = self.api_db.get_api_by_description(self.api_descriptions[idx])
            if schema: retrieved_schemas.append(schema)
        return retrieved_schemas

class APIExecutor:
    def execute_api_call(self, api_call_str: str, api_db: APIDatabase) -> Dict[str, Any]:
        try:
            # Parse the function name and arguments
            node = ast.parse(api_call_str).body[0].value
            function_name = node.func.id
            args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}

            # Find the corresponding API schema
            target_api_schema: Optional[APISchema] = None
            for api_schema in api_db.api_docs:
                if api_schema.function_name == function_name:
                    target_api_schema = api_schema
                    break
            
            if not target_api_schema:
                return {"error": f"API function '{function_name}' not found.", "status": 404}

            print(f"Simulating API call to {target_api_schema.endpoint} with args: {args}")
            
            # Mock API response based on function name
            if function_name == "search_articles":
                return {"status": 200, "data": [
                    {"title": f"Article on {args.get('query', 'medical topic')} 1", "author": "Dr. A"},
                    {"title": f"Article on {args.get('query', 'medical topic')} 2", "author": "Dr. B"}
                ]}
            elif function_name == "find_trials":
                return {"status": 200, "data": [
                    {"trial_id": "NCT01234", "condition": args.get('condition', 'unknown'), "phase": args.get('phase', 'N/A')},
                    {"trial_id": "NCT05678", "condition": args.get('condition', 'unknown'), "phase": args.get('phase', 'N/A')}
                ]}
            elif function_name == "get_drug_info":
                return {"status": 200, "data": {"name": args.get('drug_name', 'unknown'), "description": "A widely used medication.", "side_effects": ["nausea"]}}
            
            return {"status": 200, "data": "Simulated API response for " + function_name}

        except Exception as e:
            return {"error": str(e), "status": 500}

# --- FastAPI Backend ---
app = FastAPI()

# Initialize global components (singletons for the backend)
api_database = APIDatabase()
populate_mock_api_database(api_database)
query_encoder = QueryEncoder()
vector_db = VectorDatabase(api_database, query_encoder)
fined_tuned_llm = FinetunedLLM() # Assume it's already finetuned for demo
api_executor = APIExecutor()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def process_user_query(request: QueryRequest):
    user_query = request.query
    print(f"Received user query: {user_query}")

    # 1. Retrieve relevant API documentation
    retrieved_docs = vector_db.retrieve_top_k(user_query, k=3)
    
    # 2. LLM generates API call and explanation
    llm_output = fined_tuned_llm.generate_api_call_and_explanation(user_query, retrieved_docs)
    generated_api_call = llm_output["api_call"]
    explanation = llm_output["explanation"]

    if generated_api_call:
        # 3. Execute the API call
        execution_result = api_executor.execute_api_call(generated_api_call, api_database)
        return {"explanation": explanation, "api_call": generated_api_call, "result": execution_result}
    else:
        return {"explanation": explanation, "api_call": "", "result": {"status": 400, "message": "Could not generate a valid API call."}}

# --- Streamlit Frontend (for local execution) ---

def run_streamlit_frontend():
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Please run `pip install streamlit` to use the frontend.")
        return

    st.title("Medical Research Assistant LLM")
    st.markdown("Ask complex medical questions to retrieve information from various medical APIs.")

    user_input = st.text_area("Your Medical Query:", "Search PubMed for articles about diabetes treatment for patients over 60")
    
    if st.button("Get Information"):
        if user_input:
            st.info("Processing your query...")
            
            # In a real app, this would call the FastAPI backend
            # For this combined script, we'll simulate the backend call directly.
            # You would replace this with: requests.post("http://localhost:8000/query", json={"query": user_input}).json()

            # Simulate backend processing
            req = QueryRequest(query=user_input)
            response = process_user_query(req)

            st.subheader("Explanation")
            st.write(response["explanation"])

            st.subheader("Generated API Call (Simulated)")
            st.code(response["api_call"], language="python")

            st.subheader("API Execution Result (Simulated)")
            st.json(response["result"])
        else:
            st.warning("Please enter a query.")

# --- 5. Evaluation Module (Offline) ---

class ASTMatcher:
    def __init__(self):
        pass

    def parse_api_call_to_ast(self, api_call_str: str) -> Optional[ast.Call]:
        try:
            tree = ast.parse(api_call_str)
            # Expecting a single expression statement which is a Call
            if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Call):
                return tree.body[0].value
            return None
        except SyntaxError:
            return None

    def _compare_args(self, generated_args: List[ast.keyword], ground_truth_args: List[ast.keyword]) -> bool:
        gen_arg_map = {kw.arg: ast.literal_eval(kw.value) for kw in generated_args}
        gt_arg_map = {kw.arg: ast.literal_eval(kw.value) for kw in ground_truth_args}

        if len(gen_arg_map) != len(gt_arg_map): return False

        for arg_name, gen_val in gen_arg_map.items():
            if arg_name not in gt_arg_map or gt_arg_map[arg_name] != gen_val:
                return False
        return True

    def subtree_match(self, generated_call_str: str, ground_truth_call_str: str) -> Dict[str, Any]:
        gen_ast = self.parse_api_call_to_ast(generated_call_str)
        gt_ast = self.parse_api_call_to_ast(ground_truth_call_str)

        if not gen_ast or not gt_ast:
            return {"exact_match": False, "functional_correctness": False, "reason": "Invalid AST parsing"}

        # Compare function name
        func_name_match = (gen_ast.func.id == gt_ast.func.id if isinstance(gen_ast.func, ast.Name) and isinstance(gt_ast.func, ast.Name) else False)
        
        # Compare arguments
        args_match = self._compare_args(gen_ast.keywords, gt_ast.keywords)

        exact_match = func_name_match and args_match
        functional_correctness = func_name_match # Simplified: only function name for functional correctness

        return {"exact_match": exact_match, "functional_correctness": functional_correctness}

def evaluate_model(fined_tuned_llm: FinetunedLLM, api_db: APIDatabase, query_encoder: QueryEncoder, test_dataset: List[Dict[str, str]]):
    ast_matcher = ASTMatcher()
    total_samples = len(test_dataset)
    exact_matches = 0
    functional_correctness_matches = 0
    hallucinations = 0

    print("\n--- Running Evaluation ---")
    for i, sample in enumerate(test_dataset):
        user_query = sample["question"]
        ground_truth_api_call = sample["api_call"]
        
        retrieved_docs = VectorDatabase(api_db, query_encoder).retrieve_top_k(user_query, k=3)
        llm_output = fined_tuned_llm.generate_api_call_and_explanation(user_query, retrieved_docs)
        generated_api_call = llm_output["api_call"]

        if not generated_api_call:
            hallucinations += 1 # Treat empty generation as hallucination/failure
            continue

        match_results = ast_matcher.subtree_match(generated_api_call, ground_truth_api_call)
        if match_results["exact_match"]:
            exact_matches += 1
        if match_results["functional_correctness"]:
            functional_correctness_matches += 1
        else: # Simplified hallucination: if not functionally correct, it's a hallucination
            hallucinations += 1
        
        print(f"Sample {i+1}: GT=\"{ground_truth_api_call}\", Gen=\"{generated_api_call}\", Match={match_results}")

    print("\n--- Evaluation Summary ---")
    print(f"Total Samples: {total_samples}")
    print(f"Exact Matches: {exact_matches}/{total_samples} ({exact_matches/total_samples*100:.2f}%) ")
    print(f"Functional Correctness Matches: {functional_correctness_matches}/{total_samples} ({functional_correctness_matches/total_samples*100:.2f}%) ")
    print(f"Hallucinations/Failures: {hallucinations}/{total_samples} ({hallucinations/total_samples*100:.2f}%) ")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Setup for all modules
    api_db = APIDatabase()
    populate_mock_api_database(api_db)

    # Generate synthetic training data
    print("\n--- Generating Synthetic Data ---")
    synthetic_data = generate_synthetic_data(api_db, num_samples=5)
    print("Synthetic data generated:")
    for item in synthetic_data:
        print(item)

    # Finetune LLM (simulation)
    print("\n--- Finetuning LLM (Simulation) ---")
    query_encoder_for_training = QueryEncoder()
    base_llm = FinetunedLLM()
    finetuned_llm_model = finetune_llm_placeholder(base_llm, synthetic_data, query_encoder_for_training, api_db)

    # --- Demonstrate Inference via FastAPI (Run separately) ---
    print("\n--- FastAPI Backend Setup ---")
    print("To run the FastAPI backend, save this code as a Python file (e.g., `app.py`) and run:")
    print("  uvicorn app:app --reload")
    print("Then navigate to http://localhost:8000/docs for the API interface.")

    # --- Demonstrate Frontend via Streamlit (Run separately) ---
    print("\n--- Streamlit Frontend Setup ---")
    print("To run the Streamlit frontend, save this code as a Python file (e.g., `app.py`) and run:")
    print("  streamlit run app.py")
    print("Note: The Streamlit app in this combined script directly calls the backend logic for demo purposes.")
    
    # Optional: Run a mock evaluation
    # For a real evaluation, you'd have a separate, curated test set.
    print("\n--- Running Mock Evaluation ---")
    evaluate_model(finetuned_llm_model, api_db, query_encoder_for_training, synthetic_data[:2]) # Evaluate on a subset

    # To run the streamlit frontend uncomment the next line and make sure streamlit is installed
    # run_streamlit_frontend()

    # To run the FastAPI server, you would typically put `uvicorn.run(app, host="0.0.0.0", port=8000)`
    # here or run via the command line. For a combined script, this is usually handled externally.
    # For demonstration within __main__, we won't auto-start FastAPI directly here.

