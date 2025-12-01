from prompt_templates import PromptTemplates
from code_executor import CodeExecutor

def main():
    print("Welcome to the Financial Advisory and Portfolio Optimization System!")

    # 1. Get User Input (Simulated for this example)
    user_goals = "save for retirement in 20 years, grow wealth"
    user_risk_tolerance = "medium"
    user_investments = "stocks: 10000, bonds: 5000, cash: 2000"

    print(f"\nUser Goals: {user_goals}")
    print(f"User Risk Tolerance: {user_risk_tolerance}")
    print(f"Current Investments: {user_investments}")

    # 2. Generate Prompt for LLM
    llm_prompt = PromptTemplates.financial_analysis_prompt(
        goals=user_goals,
        risk_tolerance=user_risk_tolerance,
        investments=user_investments
    )
    print("\n--- LLM Prompt Generated ---")
    # print(llm_prompt) # Uncomment to see the full prompt sent to the LLM

    # 3. Simulate LLM Generating Code (In a real application, this would be an API call to an LLM)
    # For this demonstration, we'll extract the code block from the prompt itself
    # In a real scenario, the LLM would *respond* with similar code.
    start_code_tag = "# START CODE\n"
    end_code_tag = "\n# END CODE"
    
    if start_code_tag in llm_prompt and end_code_tag in llm_prompt:
        generated_code = llm_prompt.split(start_code_tag)[1].split(end_code_tag)[0].strip()
    else:
        generated_code = "print('Error: Could not extract code from prompt.')"
        
    print("\n--- Simulated LLM Generated Code ---")
    print(generated_code)

    # 4. Execute the Generated Code
    print("\n--- Executing Generated Code ---")
    execution_result = CodeExecutor.execute_python_code(generated_code)

    # 5. Process and Display Results
    if execution_result["error"]:
        print(f"Code Execution Error: {execution_result['error']}")
    else:
        print("\n--- Financial Advice (from Executed Code) ---")
        print(execution_result["output"])
        
        # In a more advanced PAL system, this output might be fed back to the LLM
        # for further natural language refinement before presenting to the user.
        # For this example, we directly present the computational result.

if __name__ == "__main__":
    main()