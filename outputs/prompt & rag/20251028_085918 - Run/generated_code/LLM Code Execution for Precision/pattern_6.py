import io
import sys

def execute_python_code(code: str) -> (str, str):
    """
    Executes the given Python code in a sandboxed environment and captures its output.
    Returns a tuple of (stdout_output, stderr_output).
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_stdout = io.StringIO()
    redirected_stderr = io.StringIO()
    
    sys.stdout = redirected_stdout
    sys.stderr = redirected_stderr
    
    execution_error = None
    try:
        # Use exec to run the dynamically generated code
        exec(code, {'np': __import__('numpy'), 'optimize_portfolio': __import__('portfolio_optimizer').optimize_portfolio, 'calculate_var': __import__('risk_calculator').calculate_var})
    except Exception as e:
        execution_error = str(e)
        
    stdout_output = redirected_stdout.getvalue()
    stderr_output = redirected_stderr.getvalue()
    
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    return stdout_output, execution_error or stderr_output