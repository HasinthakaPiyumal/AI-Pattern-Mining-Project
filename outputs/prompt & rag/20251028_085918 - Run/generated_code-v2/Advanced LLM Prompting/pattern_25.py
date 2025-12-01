import random

class QuizQuestionGenerator:
    def __init__(self):
        self.templates = {}
        self.generated_questions = []
        self._question_id_counter = 0

    def add_template(self, template_name, template_string):
        self.templates[template_name] = template_string

    def get_template(self, template_name):
        return self.templates.get(template_name)

    def _simulate_llm_fill(self, template_string, course_keywords):
        filled_question = template_string
        keywords_iter = iter(course_keywords)

        def get_next_keyword():
            try:
                return next(keywords_iter)
            except StopIteration:
                return "_MISSING_KEYWORD_"

        # Simple placeholder replacement logic
        if "[CONCEPT]" in filled_question:
            filled_question = filled_question.replace("[CONCEPT]", get_next_keyword(), 1)
        if "[DEFINITION]" in filled_question:
            filled_question = filled_question.replace("[DEFINITION]", get_next_keyword(), 1)

        # For options, try to use distinct keywords if available, otherwise just use more keywords or generic terms
        options_placeholders = [p for p in ["[OPTION_1]", "[OPTION_2]", "[OPTION_3]", "[OPTION_4]"] if p in filled_question]
        random.shuffle(course_keywords) # Shuffle keywords to get variety for options
        option_keywords_iter = iter(course_keywords)

        for placeholder in options_placeholders:
            try:
                filled_question = filled_question.replace(placeholder, next(option_keywords_iter), 1)
            except StopIteration:
                # Fallback if not enough keywords for options
                filled_question = filled_question.replace(placeholder, f"_GENERIC_OPTION_{options_placeholders.index(placeholder) + 1}_", 1)

        # Replace any remaining placeholders with a generic fallback or empty string
        import re
        filled_question = re.sub(r"\[[A-Z0-9_]+\]", "...", filled_question) # Replace any remaining brackets with '...'

        return filled_question

    def generate_question(self, template_name, course_content_keywords):
        template_string = self.get_template(template_name)
        if not template_string:
            return None, "Template not found."

        self._question_id_counter += 1
        question_id = self._question_id_counter
        generated_text = self._simulate_llm_fill(template_string, list(course_content_keywords)) # Pass a copy to avoid iterator exhaustion issues elsewhere

        question_data = {
            "id": question_id,
            "question": generated_text,
            "template_name": template_name,
            "is_valid": None  # Will be True/False after validation
        }
        self.generated_questions.append(question_data)
        return question_data, None

    def validate_question(self, question_id, is_valid):
        for q in self.generated_questions:
            if q["id"] == question_id:
                q["is_valid"] = is_valid
                return True
        return False

    def get_generated_questions(self):
        return self.generated_questions


# --- Demo Usage ---
if __name__ == "__main__":
    generator = QuizQuestionGenerator()

    # 1. Educator defines/selects question templates
    generator.add_template("multiple_choice_definition", "What is [CONCEPT]?\nA. [DEFINITION]\nB. [OPTION_1]\nC. [OPTION_2]\nD. [OPTION_3]")
    generator.add_template("true_false_statement", "True or False: [CONCEPT] is [DEFINITION].")
    generator.add_template("fill_in_the_blank", "The process of [CONCEPT] involves [DEFINITION] and ______________.")

    print("--- Templates Defined ---")
    for name, tmpl in generator.templates.items():
        print(f"  {name}: {tmpl}")
    print("\n")

    # 2. Educator provides course content (simplified to keywords)
    course_keywords_set_1 = ["Photosynthesis", "process", "plants", "sunlight", "energy", "glucose", "oxygen", "chlorophyll", "carbon dioxide", "water"]
    course_keywords_set_2 = ["Artificial Intelligence", "machine learning", "data", "algorithms", "neural networks", "deep learning", "automation"]

    # 3. Generate questions
    print("--- Generating Questions ---")
    question1_data, error = generator.generate_question("multiple_choice_definition", course_keywords_set_1)
    if error: print(f"Error: {error}")
    else: print(f"Generated Question 1 (ID: {question1_data['id']}):\n{question1_data['question']}\n")

    question2_data, error = generator.generate_question("true_false_statement", course_keywords_set_1)
    if error: print(f"Error: {error}")
    else: print(f"Generated Question 2 (ID: {question2_data['id']}):\n{question2_data['question']}\n")

    question3_data, error = generator.generate_question("fill_in_the_blank", course_keywords_set_2)
    if error: print(f"Error: {error}")
    else: print(f"Generated Question 3 (ID: {question3_data['id']}):\n{question3_data['question']}\n")

    question4_data, error = generator.generate_question("multiple_choice_definition", course_keywords_set_2)
    if error: print(f"Error: {error}")
    else: print(f"Generated Question 4 (ID: {question4_data['id']}):\n{question4_data['question']}\n")

    # 4. Simulate human validation
    print("--- Simulating Human Validation ---")
    generator.validate_question(1, True) # Question 1 is valid
    generator.validate_question(2, False) # Question 2 is invalid
    generator.validate_question(3, True)
    generator.validate_question(4, True)
    print("Validation complete.\n")

    # 5. Review generated questions with validation status
    print("--- Final Generated Questions ---")
    for q in generator.get_generated_questions():
        print(f"ID: {q['id']}, Valid: {q['is_valid']}, Template: {q['template_name']}\nQuestion: {q['question']}\n")
