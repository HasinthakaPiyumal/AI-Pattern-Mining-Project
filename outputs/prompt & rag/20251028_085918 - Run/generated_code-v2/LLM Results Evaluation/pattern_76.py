from chatbot_simulator import simulate_chatbot_response
from llm_evaluator import evaluate_response, LIKERT_SCALE

def main():
    print("AI-Powered Customer Support Chatbot Evaluation System")
    print("--------------------------------------------------")

    test_queries = [
        "Hello, I have a question about my account balance.",
        "My internet is not working. I need technical support.",
        "How can I request a refund for my last order?",
        "Hi, can you help me?",
        "Thank you for your help!",
        "I need to know my statement details."
    ]

    for i, query in enumerate(test_queries):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Customer Query: {query}")

        chatbot_response = simulate_chatbot_response(query)
        print(f"Chatbot Response: {chatbot_response}")

        evaluation = evaluate_response(query, chatbot_response)
        print(f"LLM Evaluation: {evaluation}")
        print(f"Likert Scale: {', '.join(LIKERT_SCALE)}")

        # Optional: Add a simple interpretation or suggestion based on evaluation
        if evaluation == "Poor":
            print("Suggestion: This response needs significant improvement in clarity, helpfulness, or empathy.")
        elif evaluation == "Acceptable":
            print("Suggestion: The response is adequate but could be more comprehensive or engaging.")
        elif evaluation == "Good":
            print("Suggestion: A solid response, addressing the query effectively.")
        elif evaluation == "Very Good":
            print("Suggestion: Excellent response, clear, helpful, and potentially proactive.")
        elif evaluation == "Incredible":
            print("Suggestion: Outstanding response, highly effective and customer-centric.")

if __name__ == "__main__":
    main()