import json

class MockLLM:
    def invoke(self, prompt):
        if "initial draft translation" in prompt:
            # Simulate initial translation
            if "hello" in prompt.lower():
                return {"query_translation": "Hello, how can I help you?", "response_translation": "Hola, ¿cómo puedo ayudarte?"}
            elif "problem with my order" in prompt.lower():
                return {"query_translation": "I have a problem with my order.", "response_translation": "Tengo un problema con mi pedido."}
            else:
                return {"query_translation": "Translated query mock.", "response_translation": "Translated response mock."}
        elif "refine the following translations" in prompt:
            # Simulate refinement
            if "incorrect term" in prompt:
                return {"query_translation": "Refined query: corrected term.", "response_translation": "Respuesta refinada: término corregido."}
            elif "unclear sentiment" in prompt:
                return {"query_translation": "Refined query: clearer tone.", "response_translation": "Respuesta refinada: tono más claro."}
            else:
                return {"query_translation": "Refined query based on feedback.", "response_translation": "Refined response based on feedback."}
        return {"query_translation": "Default translated query.", "response_translation": "Default translated response."}

class LLMTranslator:
    def __init__(self, llm):
        self.llm = llm

    def get_initial_translation(self, customer_query, target_language):
        prompt = f"Please provide an initial draft translation of the customer's query '{customer_query}' into {target_language} and a draft response in the customer's original language." 
        response = self.llm.invoke(prompt)
        return response["query_translation"], response["response_translation"]

    def refine_translation(self, original_query, current_query_translation, current_response_translation, feedback):
        prompt = f"Refine the following translations based on the provided feedback.\nOriginal Query: {original_query}\nCurrent Query Translation: {current_query_translation}\nCurrent Response Translation: {current_response_translation}\nFeedback: {feedback}"
        response = self.llm.invoke(prompt)
        return response["query_translation"], response["response_translation"]

class AutomatedReviewer:
    def __init__(self, knowledge_base, terms_glossary):
        self.knowledge_base = knowledge_base
        self.terms_glossary = terms_glossary

    def check_contextual_relevance(self, translated_query):
        for key_phrase in self.knowledge_base:
            if key_phrase.lower() in translated_query.lower():
                return {"flagged": False, "reason": "Contextually relevant."}
        return {"flagged": True, "reason": "Query context not found in knowledge base."}

    def analyze_tone_sentiment(self, translated_response):
        # Simplified sentiment analysis
        if any(bad_word in translated_response.lower() for bad_word in ["unhappy", "dissatisfied", "bad"]):
            return {"flagged": True, "reason": "Potentially negative sentiment detected."}
        return {"flagged": False, "reason": "Sentiment appears neutral/positive."}

    def check_terminology_consistency(self, translated_response):
        for term, correct_translation in self.terms_glossary.items():
            if term.lower() in translated_response.lower() and correct_translation.lower() not in translated_response.lower():
                return {"flagged": True, "reason": f"Incorrect terminology used for '{term}'. Expected '{correct_translation}'."}
        return {"flagged": False, "reason": "Terminology appears consistent."}

    def perform_all_checks(self, query_translation, response_translation):
        feedback = []
        context_check = self.check_contextual_relevance(query_translation)
        if context_check["flagged"]:
            feedback.append(f"Automated Feedback (Context): {context_check['reason']}")
        
        sentiment_check = self.analyze_tone_sentiment(response_translation)
        if sentiment_check["flagged"]:
            feedback.append(f"Automated Feedback (Sentiment): {sentiment_check['reason']}")
        
        terminology_check = self.check_terminology_consistency(response_translation)
        if terminology_check["flagged"]:
            feedback.append(f"Automated Feedback (Terminology): {terminology_check['reason']}")
        
        return feedback

def get_human_feedback(original_query, query_translation, response_translation, automated_feedback):
    print("\n--- Human Review Required ---")
    print(f"Original Query: {original_query}")
    print(f"Current Query Translation: {query_translation}")
    print(f"Current Response Translation: {response_translation}")
    if automated_feedback:
        print("Automated Feedback:")
        for fb in automated_feedback:
            print(f"  - {fb}")
    
    human_input = input("Please provide feedback for refinement (or press Enter to skip): ")
    return human_input

def main():
    # Mock Data
    knowledge_base = {
        "order status": "Information about order tracking and delivery.",
        "product return": "Policy and process for returning products.",
        "technical issue": "Troubleshooting steps for common technical problems."
    }
    terms_glossary = {
        "shipping": "envío",
        "refund": "reembolso",
        "account": "cuenta"
    }

    llm = MockLLM()
    llm_translator = LLMTranslator(llm)
    automated_reviewer = AutomatedReviewer(knowledge_base, terms_glossary)

    print("Multilingual Customer Support Chatbot")
    customer_query = input("Enter customer query (e.g., 'Hola, tengo un problema con mi pedido'): ")
    target_language = "English"
    original_customer_language = "Spanish" # For more realistic mock responses

    # 1. Initial Draft Translation
    print("\n--- Initial Translation ---")
    draft_query_translation, draft_response_translation = llm_translator.get_initial_translation(customer_query, target_language)
    print(f"Draft Query (to English): {draft_query_translation}")
    print(f"Draft Response (to {original_customer_language}): {draft_response_translation}")

    # Iterative Refinement Loop
    for i in range(2): # Simulate up to 2 refinement iterations
        print(f"\n--- Iteration {i+1}: Review and Refine ---")
        
        # 2. Automated Review
        automated_feedback = automated_reviewer.perform_all_checks(draft_query_translation, draft_response_translation)
        if automated_feedback:
            print("Automated review flagged issues:")
            for fb in automated_feedback:
                print(f"  - {fb}")
        else:
            print("Automated review found no immediate issues.")

        # 3. Human Feedback (Conditional)
        human_feedback = ""
        if automated_feedback or i == 0: # Always ask for human feedback on first iteration or if automated review flags issues
            human_feedback = get_human_feedback(customer_query, draft_query_translation, draft_response_translation, automated_feedback)
        
        combined_feedback = automated_feedback + ([human_feedback] if human_feedback else [])

        if not combined_feedback:
            print("No feedback provided, proceeding with current translation.")
            break

        # 4. Iterative Refinement
        print("\n--- Refining Translation ---")
        draft_query_translation, draft_response_translation = llm_translator.refine_translation(
            customer_query, draft_query_translation, draft_response_translation, ", ".join(combined_feedback)
        )
        print(f"Refined Query (to English): {draft_query_translation}")
        print(f"Refined Response (to {original_customer_language}): {draft_response_translation}")

    # 5. Final Response Delivery
    print("\n--- Final Response ---")
    print(f"Delivering final response to customer in {original_customer_language}: {draft_response_translation}")

if __name__ == "__main__":
    main()