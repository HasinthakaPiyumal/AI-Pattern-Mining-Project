from llm_service import LLMService
from code_interpreter import CodeInterpreter

def main():
    print("Welcome to your Personal Finance Advisor!\n")
    print("I can help you with financial calculations using AI and code execution.\n")
    
    llm_service = LLMService()
    code_interpreter = CodeInterpreter()

    while True:
        user_question = input("Ask a financial question (e.g., 'Calculate future value of $1000 invested at 5% for 10 years' or 'quit'): ")
        if user_question.lower() == 'quit':
            print("Thank you for using the Personal Finance Advisor. Goodbye!")
            break

        print(f"\nUser: {user_question}")
        print("AI: Thinking...")

        # Step 1: LLM generates code based on the user's question
        generated_code = llm_service.generate_code_for_question(user_question)
        print(f"\nAI (Generated Code):\n```python\n{generated_code}\n```")

        if generated_code:
            # Step 2: Execute the generated code
            execution_result, error = code_interpreter.execute_python_code(generated_code)
            
            if error:
                final_answer = f"I encountered an error during calculation: {error}. Please try rephrasing your question."
                print(f"\nAI: {final_answer}")
            else:
                # Step 3: LLM uses the result to formulate a natural language answer
                # In a real scenario, the LLM would be prompted again with the original question + result
                # For this example, we'll simulate a simple integration.
                if "future value" in user_question.lower() and execution_result:
                    try:
                        # Assuming execution_result contains the final calculated value, possibly prefixed.
                        # We need to extract the numerical part safely.
                        # This is a simplification; a real LLM would parse the output intelligently.
                        value_str = execution_result.split("Result:")[-1].strip()
                        future_value = float(value_str)
                        final_answer = f"Based on my calculations, the future value of your investment is approximately ${future_value:,.2f}."
                    except ValueError:
                        final_answer = f"I calculated a result of: {execution_result}. Let me try to interpret that for you..."
                elif execution_result:
                    final_answer = f"Based on my calculations, here is the result: {execution_result}\nI can now elaborate on this if you'd like."
                else:
                    final_answer = "I executed the code but didn't get a clear numerical result. Can you clarify what you're looking for?"
                
                print(f"\nAI: {final_answer}")
        else:
            print("\nAI: I couldn't generate suitable code for your question. Please try again.")

if __name__ == "__main__":
    main()