import ast
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel

# --- 1. api_schemas.py content ---

class APIParameter(BaseModel):
    name: str
    type: str
    optional: bool = False

class APIDefinition(BaseModel):
    name: str
    parameters: List[APIParameter]

# Define known API definitions for an e-commerce platform
KNOWN_API_DEFINITIONS: List[APIDefinition] = [
    APIDefinition(
        name="product_search_api.search_item",
        parameters=[
            APIParameter(name="query", type="str"),
            APIParameter(name="category", type="str", optional=True),
            APIParameter(name="min_price", type="float", optional=True),
            APIParameter(name="max_price", type="float", optional=True),
            APIParameter(name="color", type="str", optional=True),
            APIParameter(name="size", type="str", optional=True),
        ],
    ),
    APIDefinition(
        name="order_management_api.create_order",
        parameters=[
            APIParameter(name="user_id", type="str"),
            APIParameter(name="product_ids", type="List[str]"),
            APIParameter(name="quantities", type="List[int]"),
            APIParameter(name="shipping_address", type="str"),
        ],
    ),
    APIDefinition(
        name="customer_service_api.process_return",
        parameters=[
            APIParameter(name="order_id", type="str"),
            APIParameter(name="reason", type="str"),
            APIParameter(name="item_ids", type="List[str]", optional=True),
        ],
    ),
    APIDefinition(
        name="inventory_api.check_availability",
        parameters=[
            APIParameter(name="product_id", type="str"),
            APIParameter(name="quantity", type="int", optional=True),
        ],
    ),
]

# --- 2. api_database.py content (integrated with AST generation) ---

class APIDatabase:
    def __init__(self, api_definitions: List[APIDefinition]):
        self.api_definitions = {api.name: api for api in api_definitions}
        self.known_api_asts: Dict[str, ast.AST] = self._build_known_api_asts()

    def _build_known_api_asts(self) -> Dict[str, ast.AST]:
        asts = {}
        for api_def in self.api_definitions.values():
            # Create a canonical string for each API with all its required parameters
            # For optional parameters, we include them to create a full reference,
            # but the AST matching logic will be flexible.
            args_str = ", ".join(
                f"{param.name}=None" for param in api_def.parameters
            )
            # We wrap it in a function definition to get a proper AST for the call node
            dummy_code = f"def _dummy():\n    {api_def.name}({args_str})"
            try:
                parsed_ast = ast.parse(dummy_code)
                # Extract the Call node from the dummy function body
                for node in ast.walk(parsed_ast):
                    if isinstance(node, ast.Call):
                        asts[api_def.name] = node
                        break
            except SyntaxError as e:
                print(f"Warning: Could not parse reference AST for {api_def.name}: {e}")
        return asts

    def get_api_definition(self, api_name: str) -> Optional[APIDefinition]:
        return self.api_definitions.get(api_name)

    def get_reference_ast(self, api_name: str) -> Optional[ast.AST]:
        return self.known_api_asts.get(api_name)


api_db = APIDatabase(KNOWN_API_DEFINITIONS)


# --- 3. Core AST Validation Logic ---

def parse_code_to_ast(code_string: str) -> Optional[ast.AST]:
    """Safely parses a Python code string into its AST representation, expecting a single call expression."""
    try:
        # We expect a single expression like 'func(arg=val)'
        parsed_ast = ast.parse(code_string.strip(), mode="eval")
        if isinstance(parsed_ast, ast.Expression) and isinstance(parsed_ast.body, ast.Call):
            return parsed_ast.body
        return None
    except SyntaxError:
        return None
    except ValueError:
        return None

def is_ast_subtree(generated_call_ast: ast.Call, reference_api_def: APIDefinition) -> bool:
    """Checks if the generated API call's AST is a valid subtree of the reference API definition.
    This means checking the API name and the presence/validity of arguments.
    """
    # Check function name
    if not isinstance(generated_call_ast.func, ast.Attribute) and not isinstance(generated_call_ast.func, ast.Name):
        return False

    generated_func_name = ''
    if isinstance(generated_call_ast.func, ast.Attribute):
        if isinstance(generated_call_ast.func.value, ast.Name):
            generated_func_name = f"{generated_call_ast.func.value.id}.{generated_call_ast.func.attr}"
    elif isinstance(generated_call_ast.func, ast.Name):
        generated_func_name = generated_call_ast.func.id

    if generated_func_name != reference_api_def.name:
        return False

    # Map reference parameters for quick lookup
    ref_params = {p.name: p for p in reference_api_def.parameters}

    # Check arguments in the generated call
    generated_args = {}
    # Positional arguments - assuming generated calls mostly use keywords for clarity
    # For simplicity, we'll assume LLM primarily generates keyword arguments for API calls.
    # A more robust system would map positional args to ref_params based on order.
    if generated_call_ast.args:
        # If there are positional arguments, it's generally harder to validate without specific order knowledge.
        # For this pattern, we'll make it strict: if positional args are used, they must be exact matches
        # for the *first* required parameters. Or, for simplicity, we might flag it.
        # Let's simplify and primarily validate keyword args.
        return False # Flag as incorrect if positional args are used for this simple validator.

    # Keyword arguments
    for kw in generated_call_ast.keywords:
        arg_name = kw.arg
        if arg_name not in ref_params:
            # Generated argument is not defined in the reference API
            return False
        # You could add type checking here by inspecting kw.value and ref_params[arg_name].type
        # For now, we only check for argument existence.
        generated_args[arg_name] = True

    # Ensure all required parameters in the reference API are present in the generated call
    for ref_param in reference_api_def.parameters:
        if not ref_param.optional and ref_param.name not in generated_args:
            return False # Missing a required argument

    return True

def validate_llm_api_call(generated_code: str) -> Tuple[bool, str, Optional[str]]:
    """Validates an LLM-generated API call string against the known API database.

    Args:
        generated_code: The Python code string representing the LLM-generated API call.

    Returns:
        A tuple: (is_valid, message, detected_api_name)
        is_valid: True if the call is valid and not a hallucination, False otherwise.
        message: A descriptive message about the validation result.
        detected_api_name: The name of the API if a valid one was detected, None otherwise.
    """
    generated_ast = parse_code_to_ast(generated_code)

    if generated_ast is None:
        return False, "Invalid Python syntax or not a single API call expression.", None

    # Attempt to extract the API name from the generated AST
    generated_func_name_parts = []
    current_node = generated_ast.func
    while isinstance(current_node, ast.Attribute):
        generated_func_name_parts.insert(0, current_node.attr)
        current_node = current_node.value
    if isinstance(current_node, ast.Name):
        generated_func_name_parts.insert(0, current_node.id)
    
    extracted_api_name = ".".join(generated_func_name_parts)

    if not extracted_api_name:
        return False, "Could not determine API name from generated call.", None

    # Check if the extracted API name is known
    reference_api_def = api_db.get_api_definition(extracted_api_name)
    if reference_api_def is None:
        return False, f"Hallucination: API '{extracted_api_name}' is not a known API.", None

    # If API name is known, perform detailed argument validation
    if is_ast_subtree(generated_ast, reference_api_def):
        return True, f"Valid call to '{extracted_api_name}'.", extracted_api_name
    else:
        return False, f"Incorrect usage of known API '{extracted_api_name}': arguments do not match required structure.", extracted_api_name


# --- Example Usage ---

if __name__ == "__main__":
    print("--- E-commerce AI Assistant API Validator Examples ---")

    # Valid calls
    valid_call_1 = 'product_search_api.search_item(query="red t-shirt", color="red", category="apparel")'
    valid_call_2 = 'order_management_api.create_order(user_id="user123", product_ids=["prod_a", "prod_b"], quantities=[1, 2], shipping_address="123 Main St")'
    valid_call_3 = 'inventory_api.check_availability(product_id="prod_c")'
    valid_call_4 = 'customer_service_api.process_return(order_id="ord987", reason="wrong size")'

    # Invalid calls (hallucinations - unknown API)
    hallucination_1 = 'unknown_api.do_something(param="value")'
    hallucination_2 = 'product_search_api.imagined_function(query="shoes")' # Invalid function within a known module

    # Invalid calls (incorrect usage of known API - missing required args, unexpected args, or bad structure)
    invalid_usage_1 = 'product_search_api.search_item(color="blue")' # Missing required 'query'
    invalid_usage_2 = 'order_management_api.create_order(user_id="user123", product_ids=["prod_a"], quantities=[1])' # Missing shipping_address
    invalid_usage_3 = 'inventory_api.check_availability(product_id="prod_d", invalid_arg=True)' # Unexpected argument
    invalid_usage_4 = 'customer_service_api.process_return(order_id="ord111")' # Missing required 'reason'
    invalid_usage_5 = 'product_search_api.search_item("t-shirt", "red")' # Using positional args (simplified to fail here)

    test_cases = {
        "Valid Call 1": valid_call_1,
        "Valid Call 2": valid_call_2,
        "Valid Call 3": valid_call_3,
        "Valid Call 4": valid_call_4,
        "Hallucination 1": hallucination_1,
        "Hallucination 2": hallucination_2,
        "Invalid Usage 1 (Missing Required)": invalid_usage_1,
        "Invalid Usage 2 (Missing Required)": invalid_usage_2,
        "Invalid Usage 3 (Unexpected Arg)": invalid_usage_3,
        "Invalid Usage 4 (Missing Required)": invalid_usage_4,
        "Invalid Usage 5 (Positional Args)": invalid_usage_5,
    }

    for name, code in test_cases.items():
        is_valid, message, detected_api = validate_llm_api_call(code)
        print(f"\n--- Test: {name} ---")
        print(f"Code: {code}")
        print(f"Result: {'VALID' if is_valid else 'INVALID'}")
        print(f"Message: {message}")
        if detected_api:
            print(f"Detected API: {detected_api}")

