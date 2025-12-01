import json
import ast
import math

# Mocking sentence-transformers and a simple cosine similarity for demonstration
class MockSentenceTransformer:
    def encode(self, texts, convert_to_tensor=False):
        # Very simplistic encoding: sum of ASCII values
        return [sum(ord(char) for char in text) for text in texts]

def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0
    # In a real scenario, these would be actual dense vectors
    # For this mock, we're just checking if they are non-zero and equal for a perfect match
    if vec1 == vec2 and vec1 != 0:
        return 1.0
    return 0.0

mock_model = MockSentenceTransformer()


# 1. API Database Curation (Mock Data)
API_DATABASE = [
    {
        "domain": "product",
        "functionality": "search",
        "description": "Searches for products based on keywords and filters.",
        "endpoint": "/products/search",
        "method": "GET",
        "parameters": [
            {"name": "query", "type": "str", "required": True, "description": "Keywords for product search"},
            {"name": "category", "type": "str", "required": False, "description": "Product category"},
            {"name": "max_price", "type": "float", "required": False, "description": "Maximum price"},
            {"name": "min_reviews", "type": "int", "required": False, "description": "Minimum number of reviews"}
        ],
        "example_response": "{'products': [{'id': '1', 'name': 'Running Shoes', 'price': 89.99}]}",
        "api_call_template": "search_products(query='{query}', category='{category}', max_price={max_price}, min_reviews={min_reviews})"
    },
    {
        "domain": "order",
        "functionality": "get_status",
        "description": "Retrieves the status of a specific order.",
        "endpoint": "/orders/{order_id}/status",
        "method": "GET",
        "parameters": [
            {"name": "order_id", "type": "str", "required": True, "description": "Unique identifier for the order"}
        ],
        "example_response": "{'order_id': '12345', 'status': 'shipped', 'estimated_delivery': '2023-10-27'}",
        "api_call_template": "get_order_status(order_id='{order_id}')"
    },
    {
        "domain": "recommendation",
        "functionality": "get_personalized",
        "description": "Provides personalized product recommendations for a user.",
        "endpoint": "/recommendations/personalized",
        "method": "GET",
        "parameters": [
            {"name": "user_id", "type": "str", "required": True, "description": "User identifier"}
        ],
        "example_response": "{'recommendations': [{'id': '2', 'name': 'Smartwatch'}]}",
        "api_call_template": "get_personalized_recommendations(user_id='{user_id}')"
    },
    {
        "domain": "product",
        "functionality": "get_details",
        "description": "Retrieves detailed information for a specific product.",
        "endpoint": "/products/{product_id}",
        "method": "GET",
        "parameters": [
            {"name": "product_id", "type": "str", "required": True, "description": "Unique identifier for the product"}
        ],
        "example_response": "{'product_id': 'XYZ', 'name': 'XYZ Laptop', 'specs': 'High-end', 'price': 1200.00}",
        "api_call_template": "get_product_details(product_id='{product_id}')"
    }
]

# Pre-compute embeddings for API descriptions for the mock retriever
API_DESCRIPTIONS = [api["description"] for api in API_DATABASE]
API_DESCRIPTION_EMBEDDINGS = mock_model.encode(API_DESCRIPTIONS)


# 2. Synthetic Instruction Generation (Mock)
class SyntheticDataGenerator:
    def generate_data(self):
        # In a real scenario, this would use an LLM to generate diverse pairs.
        # For demonstration, we'll use a few hand-crafted examples.
        synthetic_data = [
            ("Find me running shoes under $100", "search_products(query='running shoes', max_price=100.0)"),
            ("What's the status of my order 12345?", "get_order_status(order_id='12345')"),
            ("Recommend some products for me", "get_personalized_recommendations(user_id='current_user')"),
            ("Tell me about the XYZ laptop", "get_product_details(product_id='XYZ')"),
            ("Show me sneakers under 50 dollars with at least 100 reviews", "search_products(query='sneakers', max_price=50.0, min_reviews=100)")
        ]
        return synthetic_data


# 3. Retriever
class Retriever:
    def __init__(self, api_database, api_description_embeddings, embedding_model):
        self.api_database = api_database
        self.api_description_embeddings = api_description_embeddings
        self.embedding_model = embedding_model

    def retrieve_relevant_api(self, query):
        query_embedding = self.embedding_model.encode([query])[0]
        
        max_similarity = -1
        most_relevant_api = None

        for i, api_doc in enumerate(self.api_database):
            doc_embedding = self.api_description_embeddings[i]
            similarity = cosine_similarity(query_embedding, doc_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_relevant_api = api_doc
        
        return most_relevant_api


# 4. LLM Finetuning Pipeline (Mock)
class FinetunedLLM:
    def __init__(self, synthetic_data):
        self.synthetic_data = synthetic_data

    def generate_api_call_and_explanation(self, prompt):
        # In a real scenario, a finetuned LLM would process the prompt and generate output.
        # Here, we'll do a simple lookup based on synthetic data for demonstration.
        # This part assumes the prompt contains the user query and relevant API docs.
        
        user_query = prompt.split("User Query:")[1].split("Relevant API Docs:")[0].strip()
        
        for nl_query, api_call in self.synthetic_data:
            if user_query.lower() in nl_query.lower():
                explanation = f"Based on your query, I identified the need to call the {api_call.split('(')[0]} function."
                return api_call, explanation
        
        # Fallback if no direct match in synthetic data
        if "running shoes under $100" in user_query.lower():
            return "search_products(query='running shoes', max_price=100.0)", "I found running shoes within your budget."
        if "order status" in user_query.lower() and "12345" in user_query:
            return "get_order_status(order_id='12345')", "Checking the status for your order."
        if "recommend" in user_query.lower():
            return "get_personalized_recommendations(user_id='current_user')", "Here are some personalized recommendations."
        if "specs" in user_query.lower() and "XYZ laptop" in user_query.lower():
            return "get_product_details(product_id='XYZ')", "Retrieving details for the XYZ laptop."
        if "sneakers under 50 dollars with at least 100 reviews" in user_query.lower():
             return "search_products(query='sneakers', max_price=50.0, min_reviews=100)", "Searching for highly-rated sneakers within your price range."

        return "", "I could not generate a precise API call for your request based on my training. Please try rephrasing."


# 5. Inference Engine
class InferenceEngine:
    def __init__(self, retriever, finetuned_llm):
        self.retriever = retriever
        self.finetuned_llm = finetuned_llm

    def process_query(self, user_query):
        relevant_api_doc = self.retriever.retrieve_relevant_api(user_query)
        
        prompt = f"""User Query: {user_query}
Relevant API Docs: {json.dumps(relevant_api_doc, indent=2) if relevant_api_doc else 'None'}
Generate the API call based on the user query and provided API documentation, and provide a brief explanation."""

        api_call_string, explanation = self.finetuned_llm.generate_api_call_and_explanation(prompt)
        return api_call_string, explanation


# 6. AST-based Evaluation
class ASTEvaluator:
    def evaluate_api_call(self, api_call_string):
        if not api_call_string:
            return False, "Empty API call string."

        try:
            parsed_tree = ast.parse(api_call_string)
            
            if not isinstance(parsed_tree.body[0], ast.Expr) or \
               not isinstance(parsed_tree.body[0].value, ast.Call):
                return False, "Not a valid function call expression."

            func_call = parsed_tree.body[0].value
            func_name = func_call.func.id

            # Basic validation: check if function name is among expected ones
            expected_functions = {"search_products", "get_order_status", "get_personalized_recommendations", "get_product_details"}
            if func_name not in expected_functions:
                return False, f"Unknown function name: {func_name}"
            
            # More advanced checks could go here, e.g., argument types, required arguments

            return True, "API call is syntactically valid and uses an expected function."

        except SyntaxError as e:
            return False, f"Syntax Error in API call: {e}"
        except Exception as e:
            return False, f"Evaluation Error: {e}"


# 7. Main Application / User Interface
def main():
    print("Initializing E-commerce AI Assistant...")

    # Initialize components
    synthetic_data_generator = SyntheticDataGenerator()
    synthetic_training_data = synthetic_data_generator.generate_data()
    
    retriever = Retriever(API_DATABASE, API_DESCRIPTION_EMBEDDINGS, mock_model)
    finetuned_llm = FinetunedLLM(synthetic_training_data)
    inference_engine = InferenceEngine(retriever, finetuned_llm)
    ast_evaluator = ASTEvaluator()

    print("AI Assistant ready. Type 'exit' to quit.")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == 'exit':
            break

        api_call, explanation = inference_engine.process_query(user_input)
        print(f"Assistant Explanation: {explanation}")
        print(f"Generated API Call: {api_call}")

        is_valid, eval_message = ast_evaluator.evaluate_api_call(api_call)
        print(f"API Call Evaluation: {'Valid' if is_valid else 'Invalid'} - {eval_message}")
        
        if is_valid and api_call: # Simulate execution if valid and not empty
            print("Simulating API call execution...")
            # In a real system, you would execute the `api_call` here.
            # For example, by parsing the string and calling corresponding Python functions.
            if "search_products" in api_call:
                print("Simulated: Calling product search API...")
                print("Simulated Response: {'products': [{'id': '1', 'name': 'Running Shoes', 'price': 89.99}]}")
            elif "get_order_status" in api_call:
                print("Simulated: Calling order status API...")
                print("Simulated Response: {'order_id': '12345', 'status': 'shipped'}")
            elif "get_personalized_recommendations" in api_call:
                print("Simulated: Calling recommendation API...")
                print("Simulated Response: {'recommendations': [{'id': '2', 'name': 'Smartwatch'}]}")
            elif "get_product_details" in api_call:
                print("Simulated: Calling product details API...")
                print("Simulated Response: {'product_id': 'XYZ', 'name': 'XYZ Laptop', 'price': 1200.00}")
        elif not api_call:
            print("No API call was generated to simulate.")

if __name__ == "__main__":
    main()
