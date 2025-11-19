import io
import contextlib

class ExternalInterpreter:
    def execute_code(self, code_string: str):
        """
        Simulates an external Python interpreter executing the provided code string.
        It captures the 'result' variable set by the executed code.
        
        NOTE: For a real-world application, executing arbitrary code from an LLM
        requires a highly secure sandboxing environment to prevent malicious execution.
        This simple implementation is for demonstration purposes only.
        """
        local_vars = {}
        captured_output = io.StringIO()

        try:
            with contextlib.redirect_stdout(captured_output):
                exec(code_string, {}, local_vars)
            
            if 'result' in local_vars:
                return local_vars['result']
            else:
                print(f"Code executed, but no 'result' variable found. Output: {captured_output.getvalue()}")
                return None
        except Exception as e:
            print(f"Error during code execution: {e}. Output: {captured_output.getvalue()}")
            return None
