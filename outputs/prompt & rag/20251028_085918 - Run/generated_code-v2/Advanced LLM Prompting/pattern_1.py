from pydantic import BaseModel
import json

class Question(BaseModel):
    question_text: str
    question_type: str
    options: list = []
    answer: str
    topic: str

class QuizGenerator:
    def __init__(self, llm_api_key: str):
        self.llm_api_key = llm_api_key
        self.question_templates = {
            "multiple_choice": "Generate a multiple-choice question about {topic} where the main concept is '{concept}'. The options should include '{correct_answer}' and three plausible distractors: '{distractor1}', '{distractor2}', '{distractor3}'. The correct answer is '{correct_answer}'. The question should be: {question_placeholder}",
            "true_false": "Generate a true/false question about {topic} focusing on the statement: '{statement}'. State whether the statement is True or False. The question should be: {question_placeholder}",
            "short_answer": "Generate a short answer question about {topic} asking to explain '{concept}'. The question should be: {question_placeholder}"
        }

    def _simulate_llm_response(self, prompt: str) -> str:
        # In a real application, this would be an actual API call to an LLM like OpenAI GPT-3/4
        # For this example, we'll simulate a response based on the prompt structure
        if "multiple-choice" in prompt:
            parts = prompt.split("The question should be: ")
            question_base = parts[1].split("\n")[0] if len(parts) > 1 else "What is the primary function of a {concept}?"
            
            topic_start = prompt.find("about ") + len("about ")
            topic_end = prompt.find(" where the main concept")
            topic = prompt[topic_start:topic_end]

            concept_start = prompt.find("concept is ") + len("concept is ") + 1
            concept_end = prompt.find("'. The options")
            concept = prompt[concept_start:concept_end]

            correct_answer_start = prompt.find("include '") + len("include '")
            correct_answer_end = prompt.find("' and three plausible")
            correct_answer = prompt[correct_answer_start:correct_answer_end]
            
            distractor1_start = prompt.find("distractors: '") + len("distractors: '")
            distractor1_end = prompt.find("', '", distractor1_start)
            distractor1 = prompt[distractor1_start:distractor1_end]

            distractor2_start = distractor1_end + len("', '")
            distractor2_end = prompt.find("', '", distractor2_start)
            distractor2 = prompt[distractor2_start:distractor2_end]

            distractor3_start = distractor2_end + len("', '")
            distractor3_end = prompt.find("'. The correct answer")
            distractor3 = prompt[distractor3_start:distractor3_end]
            
            return json.dumps({"question_text": f"Which of the following best describes {concept} in {topic}?", "options": [correct_answer, distractor1, distractor2, distractor3], "answer": correct_answer, "question_type": "multiple_choice", "topic": topic})
        elif "true/false" in prompt:
            statement_start = prompt.find("statement: '") + len("statement: '")
            statement_end = prompt.find("'. State whether")
            statement = prompt[statement_start:statement_end]
            
            topic_start = prompt.find("about ") + len("about ")
            topic_end = prompt.find(" focusing on the statement")
            topic = prompt[topic_start:topic_end]

            # Simulate a 50/50 chance for true/false
            answer = "True" if hash(prompt) % 2 == 0 else "False"
            return json.dumps({"question_text": statement, "options": ["True", "False"], "answer": answer, "question_type": "true_false", "topic": topic})
        elif "short answer" in prompt:
            concept_start = prompt.find("explain '") + len("explain '")
            concept_end = prompt.find("'. The question")
            concept = prompt[concept_start:concept_end]
            
            topic_start = prompt.find("about ") + len("about ")
            topic_end = prompt.find(" asking to explain")
            topic = prompt[topic_start:topic_end]

            return json.dumps({"question_text": f"Explain the concept of {concept} in the context of {topic}.", "options": [], "answer": "[LLM-generated explanation expected]", "question_type": "short_answer", "topic": topic})
        return json.dumps({"error": "Could not parse simulated LLM response"})


    def generate_question(self, question_type: str, topic: str, content_data: dict) -> Question:
        template = self.question_templates.get(question_type)
        if not template:
            raise ValueError(f"Unsupported question type: {question_type}")

        # Fill placeholders with content data
        # This step would involve more sophisticated NLP for real content extraction
        filled_template_prompt = template.format(
            topic=topic,
            question_placeholder=f"Generate a question about {topic} based on the following: {content_data.get('summary', '')}",
            **content_data
        )

        llm_output_json = self._simulate_llm_response(filled_template_prompt)
        llm_output = json.loads(llm_output_json)

        if "error" in llm_output:
            raise RuntimeError(f"LLM generation error: {llm_output['error']}")

        return Question(**llm_output)

    def human_validate_template(self, template_name: str, new_template: str) -> bool:
        # In a real scenario, this would involve a UI for human review and approval.
        # For this simulation, we'll just update the template if it's valid.
        print(f"Human validation requested for template '{template_name}'. Review: '{new_template}'")
        is_valid = input("Is this new template valid? (yes/no): ").lower() == 'yes'
        if is_valid:
            self.question_templates[template_name] = new_template
            print(f"Template '{template_name}' updated after human validation.")
        else:
            print(f"Template '{template_name}' not updated.")
        return is_valid

# Example Usage:
if __name__ == "__main__":
    # In a real app, llm_api_key would be loaded securely, e.g., from environment variables
    quiz_gen = QuizGenerator(llm_api_key="YOUR_LLM_API_KEY")

    # Simulate extracted content for a topic
    course_content_data = {
        "summary": "Photosynthesis is the process used by plants, algae, and cyanobacteria to convert light energy into chemical energy, through a process that converts carbon dioxide and water into sugars and oxygen. It occurs in chloroplasts.",
        "concept": "Photosynthesis",
        "correct_answer": "chloroplasts",
        "distractor1": "mitochondria",
        "distractor2": "nucleus",
        "distractor3": "cytoplasm",
        "statement": "Photosynthesis primarily occurs in the mitochondria of plant cells."
    }

    print("\n--- Generating Multiple Choice Question ---")
    try:
        mc_question = quiz_gen.generate_question(
            question_type="multiple_choice",
            topic="Biology",
            content_data=course_content_data
        )
        print(mc_question.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error generating MC question: {e}")

    print("\n--- Generating True/False Question ---")
    try:
        tf_question = quiz_gen.generate_question(
            question_type="true_false",
            topic="Biology",
            content_data=course_content_data
        )
        print(tf_question.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error generating T/F question: {e}")

    print("\n--- Generating Short Answer Question ---")
    try:
        sa_question = quiz_gen.generate_question(
            question_type="short_answer",
            topic="Biology",
            content_data=course_content_data
        )
        print(sa_question.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error generating SA question: {e}")

    print("\n--- Human Validation of Template (Simulated) ---")
    new_mc_template = "For the topic '{topic}', and concept '{concept}', create a multiple-choice question. The correct answer is '{correct_answer}', and distractors are '{distractor1}', '{distractor2}', '{distractor3}'. What is the question?"
    quiz_gen.human_validate_template("multiple_choice", new_mc_template)

    print("\n--- Generating Multiple Choice Question with Updated Template ---")
    try:
        mc_question_updated = quiz_gen.generate_question(
            question_type="multiple_choice",
            topic="Biology",
            content_data=course_content_data
        )
        print(mc_question_updated.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error generating MC question with updated template: {e}")
