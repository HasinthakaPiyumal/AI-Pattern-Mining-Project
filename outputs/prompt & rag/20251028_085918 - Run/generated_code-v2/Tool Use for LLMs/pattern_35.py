import ast
from api_database import get_known_api_definition, get_known_api_names

class HallucinationDetector:
    def __init__(self):
        pass

    def _parse_api_call_to_ast(self, api_call_string: str):
        """Parses an API call string into its Abstract Syntax Tree (AST)."""
        try:
            # Wrap the API call in a function call to make it a valid Python expression
            return ast.parse(api_call_string).body[0].value
        except (SyntaxError, AttributeError):
            return None

    def _compare_arguments(self, generated_args, known_args_definition):
        """Compares generated arguments against known argument definitions.
        Returns True if arguments match, False otherwise, along with an explanation.
        """
        generated_arg_names = {kw.arg for kw in generated_args if isinstance(kw, ast.keyword)}
        known_arg_map = {arg["name"]: arg for arg in known_args_definition}

        # Check for unknown arguments
        for gen_arg_name in generated_arg_names:
            if gen_arg_name not in known_arg_map:
                return False, f"Unknown argument '{gen_arg_name}'"

        # Check for missing required arguments
        for known_arg_name, known_arg_def in known_arg_map.items():
            if not known_arg_def["optional"] and known_arg_name not in generated_arg_names:
                return False, f"Missing required argument '{known_arg_name}'"

        # Basic type checking (can be expanded)
        for kw in generated_args:
            if isinstance(kw, ast.keyword):
                arg_name = kw.arg
                if arg_name in known_arg_map:
                    known_type = known_arg_map[arg_name]["type"]
                    if known_type == "str":
                        if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, str):
                            return False, f"Argument '{arg_name}' has incorrect type. Expected string."
                    # Add more type checks as needed (int, bool, etc.)

        return True, "Arguments match known definition."


    def detect_hallucination(self, api_call_string: str) -> dict:
        """Detects hallucinations and functional errors in an LLM-generated API call.
        Returns a dictionary with detection results.
        """
        result = {
            "is_hallucination": False,
            "is_functional_error": False,
            "message": ""
        }

        generated_ast = self._parse_api_call_to_ast(api_call_string)

        if generated_ast is None or not isinstance(generated_ast, ast.Call):
            result["is_hallucination"] = True
            result["message"] = "Could not parse API call string or not a valid function call."
            return result

        api_name = generated_ast.func.id if isinstance(generated_ast.func, ast.Name) else None

        if not api_name:
            result["is_hallucination"] = True
            result["message"] = "Could not extract API name from the generated call."
            return result

        known_api_definition = get_known_api_definition(api_name)

        if known_api_definition is None:
            result["is_hallucination"] = True
            result["message"] = f"Hallucination detected: API '{api_name}' is not a known API."
            return result

        # Compare arguments
        args_match, arg_message = self._compare_arguments(generated_ast.keywords, known_api_definition["arguments"])

        if not args_match:
            result["is_functional_error"] = True
            result["message"] = f"Functional error detected for API '{api_name}': {arg_message}"
            return result

        result["message"] = f"API '{api_name}' is valid and functionally correct."
        return result
