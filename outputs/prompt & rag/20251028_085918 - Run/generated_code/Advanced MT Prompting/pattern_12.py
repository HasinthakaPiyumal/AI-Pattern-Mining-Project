
from chatbot_core import ChatbotCore
from human_feedback_module import HumanFeedbackModule

def run_chatbot_interaction():
    chatbot = ChatbotCore()
    human_feedback = HumanFeedbackModule()

    print("Welcome to the Multilingual Customer Support Chatbot!")
    print("Type 'exit' to end the conversation.")

    while True:
        user_query = input("\nEnter your query (e.g., 'What is your shipping policy?', '¿Cuál es su política de devoluciones?', 'Lieferzeit?'): ")
        if user_query.lower() == 'exit':
            break

        source_lang = input("Enter the language of your query (e.g., 'en' for English, 'es' for Spanish, 'de' for German, default 'en'): ")
        if not source_lang:
            source_lang = "en"

        # Phase 1: Augmented Prompting & Preprocessing
        processed_query = chatbot.preprocess_query(user_query, source_lang)

        # Phase 2: Strategic Planning & Decomposition, and Initial Response Generation
        draft_response = chatbot.generate_response(processed_query)
        print(f"\nChatbot Draft Response: {draft_response}")

        # Phase 3: Human-in-the-Loop & Iterative Refinement
        feedback = human_feedback.collect_feedback(user_query, draft_response)
        final_response = human_feedback.refine_response(draft_response, feedback)
        print(f"Final Response: {final_response}")

        print("\n--- Session Summary ---")
        print(f"External Translation API Calls: {chatbot.external_translation_api_calls}")
        print(f"Multilingual LM Calls: {chatbot.multilingual_lm_calls}")
        # print("Feedback History:")
        # for i, (q, dr, fb) in enumerate(human_feedback.get_feedback_history()):
        #     print(f"  Interaction {i+1}: Query='{q}', Draft='{dr}', Feedback='{fb['status']}'")

    print("Thank you for using the Multilingual Customer Support Chatbot. Goodbye!")

if __name__ == "__main__":
    run_chatbot_interaction()
