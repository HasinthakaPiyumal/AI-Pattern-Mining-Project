class LLMSimulator:
    def __init__(self):
        self.translation_memory = {
            "hello world": "hola mundo",
            "artificial intelligence": "inteligencia artificial",
            "economic growth": "crecimiento económico",
            "political stability": "estabilidad política",
            "global market": "mercado global"
        }

    def translate(self, text, refinement_context=None):
        if refinement_context:
            print(f"LLM simulating refinement with context: {refinement_context}")
            if "inaccurate" in refinement_context:
                # Simulate correction based on a simple keyword
                if "mercado global" in text and "global market" in refinement_context:
                    text = text.replace("mercado global", "mercado mundial")
                elif "politica" in text and "political stability" in refinement_context:
                    text = text.replace("politica", "estabilidad política")
            if "suggested phrasing" in refinement_context:
                for phrase, suggestion in refinement_context["suggested phrasing"].items():
                    text = text.replace(phrase, suggestion)
            return f"[Refined] {text}"
        else:
            # Simulate initial translation, simple mapping for known phrases
            lower_text = text.lower()
            for phrase, translation in self.translation_memory.items():
                if phrase in lower_text:
                    return translation.capitalize()
            return f"[Draft] {text} (initial translation simulation)"

class AutomatedFeedback:
    def __init__(self):
        self.domain_glossary = {
            "global market": "mercado mundial",  # Preferred term
            "AI": "Inteligencia Artificial",
            "inflation": "inflación",
            "president": "presidente"
        }
        self.inconsistencies_database = {
            "politica": "political stability" # common mistake/oversimplification
        }

    def get_feedback(self, translated_text):
        feedback = []
        for term_en, term_es_preferred in self.domain_glossary.items():
            if term_en in translated_text.lower() and term_es_preferred not in translated_text.lower():
                feedback.append(f"Automated: Consider using '{term_es_preferred}' for '{term_en}'.")
        
        for common_err, correct_phrase_en in self.inconsistencies_database.items():
            if common_err in translated_text.lower() and correct_phrase_en not in translated_text.lower():
                 feedback.append(f"Automated: '{common_err}' might be inaccurate, consider context for '{correct_phrase_en}'.")
        
        if not feedback:
            feedback.append("Automated: No specific issues detected.")
        return feedback

class HumanFeedback:
    def get_feedback(self, translated_text, round_num):
        print(f"\n--- Human Review Round {round_num} ---")
        print(f"Current Translation:\n{translated_text}")
        feedback = input("Human Feedback (e.g., 'correct X to Y', 'phrasing for Z is awkward'): ")
        return feedback if feedback else "No human feedback provided."

class GlobalNewsTranslator:
    def __init__(self):
        self.llm_simulator = LLMSimulator()
        self.automated_feedback = AutomatedFeedback()
        self.human_feedback_module = HumanFeedback()

    def translate_article_iteratively(self, original_article, max_iterations=3, human_review_enabled=False):
        print(f"\n--- Starting translation for: '{original_article}' ---")
        current_translation = self.llm_simulator.translate(original_article)
        print(f"Initial LLM Draft: {current_translation}")

        for i in range(max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            refinement_context = {"automated": [], "human": ""}

            # Automated Feedback
            auto_feedback_list = self.automated_feedback.get_feedback(current_translation)
            print(f"Automated Feedback: {auto_feedback_list}")
            refinement_context["automated"] = auto_feedback_list

            # Human Feedback (if enabled)
            if human_review_enabled:
                human_feedback = self.human_feedback_module.get_feedback(current_translation, i + 1)
                print(f"Human Feedback Received: {human_feedback}")
                refinement_context["human"] = human_feedback
            else:
                print("Human review skipped for this iteration.")

            # Generate refinement prompt for LLM
            llm_refinement_input = {"text": current_translation, "context": refinement_context}
            print(f"LLM will refine with input: {llm_refinement_input}")
            
            # Simulate LLM refinement based on feedback
            # For simplicity, we'll parse very basic keywords from feedback
            simulated_refinement_instruction = {}
            if "Automated: Consider using 'mercado mundial' for 'global market'" in str(auto_feedback_list):
                simulated_refinement_instruction["suggested phrasing"] = {"mercado global": "mercado mundial"}
            if "Automated: 'politica' might be inaccurate, consider context for 'political stability'" in str(auto_feedback_list):
                simulated_refinement_instruction["inaccurate"] = "politica for political stability"
            if "correct" in refinement_context["human"].lower():
                # Very basic human feedback parsing
                if "correct global market to mercado mundial" in refinement_context["human"].lower():
                    simulated_refinement_instruction.setdefault("suggested phrasing", {})["mercado global"] = "mercado mundial"
                elif "correct politica to estabilidad politica" in refinement_context["human"].lower():
                    simulated_refinement_instruction.setdefault("suggested phrasing", {})["politica"] = "estabilidad política"

            if simulated_refinement_instruction:
                current_translation = self.llm_simulator.translate(current_translation, simulated_refinement_instruction)
            else:
                print("No specific refinement instructions from feedback for LLM simulation.")
            
            print(f"Refined Translation: {current_translation}")
            if "No specific issues detected" in str(auto_feedback_list) and not human_review_enabled or (human_review_enabled and "no human feedback" in refinement_context["human"].lower()):
                 print("Translation appears satisfactory or no further feedback. Ending iterations.")
                 break
        
        print(f"\n--- Final Translation for '{original_article}':\n{current_translation} ---")
        return current_translation

if __name__ == "__main__":
    translator = GlobalNewsTranslator()

    # Example 1: Article with automated refinement
    article1 = "The economic growth and global market trends are important."
    final_translation_1 = translator.translate_article_iteratively(article1, max_iterations=2, human_review_enabled=False)

    print("\n" + "="*50 + "\n")

    # Example 2: Article with human feedback (simulated input required)
    article2 = "News about artificial intelligence and politica."
    # For this example, you will be prompted for human feedback in the console
    final_translation_2 = translator.translate_article_iteratively(article2, max_iterations=3, human_review_enabled=True)

    print("\n" + "="*50 + "\n")

    # Example 3: Simple article with no major issues, should converge quickly
    article3 = "Hello world, AI is here."
    final_translation_3 = translator.translate_article_iteratively(article3, max_iterations=1, human_review_enabled=False)
