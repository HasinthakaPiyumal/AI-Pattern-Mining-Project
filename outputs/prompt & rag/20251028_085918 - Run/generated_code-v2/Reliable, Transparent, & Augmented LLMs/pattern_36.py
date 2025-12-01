import json
import os

# Placeholder for OpenAI client. In a real application, you would initialize it like:
# from openai import OpenAI
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class MockLLMClient:
    """A mock LLM client for demonstration purposes."""
    def chat_completions_create(self, model, messages, response_format=None):
        # Simulate LLM response based on message content
        last_message = messages[-1]['content']
        if "generate a detailed scoring rubric" in last_message.lower():
            return {"choices": [{"message": {"content": json.dumps({
                "rubric_title": "Essay Evaluation Rubric: " + last_message.split("for the essay prompt:")[1].strip(),
                "criteria": [
                    {"name": "Argumentation and Thesis", "weight": 30, "description": "Clarity, strength, and originality of the thesis statement and supporting arguments."},
                    {"name": "Evidence and Support", "weight": 25, "description": "Relevance, sufficiency, and effective integration of evidence."},
                    {"name": "Organization and Structure", "weight": 20, "description": "Logical flow, paragraphing, and coherence."},
                    {"name": "Language and Style", "weight": 15, "description": "Clarity, conciseness, grammar, spelling, and vocabulary."},
                    {"name": "Originality and Critical Thinking", "weight": 10, "description": "Demonstration of independent thought and analytical depth."}
                ],
                "scoring_scale": "0-100"
            })}}]}
        elif "evaluate the following essay" in last_message.lower():
            return {"choices": [{"message": {"content": json.dumps({
                "overall_score": 85,
                "feedback": {
                    "Argumentation and Thesis": "The thesis is clear and well-supported with logical arguments.",
                    "Evidence and Support": "Good use of evidence, though some sections could benefit from deeper analysis.",
                    "Organization and Structure": "Essay is well-organized with clear transitions between paragraphs.",
                    "Language and Style": "Generally strong, with a few minor grammatical errors.",
                    "Originality and Critical Thinking": "Shows good understanding and some original insights."
                },
                "summary": "A strong essay demonstrating good analytical skills and clear communication."
            })}}]}
        return {"choices": [{"message": {"content": "{}"}}]}


class LLMInteractionModule:
    def __init__(self, api_key=None, model="gpt-4"):
        if api_key:
            # from openai import OpenAI
            # self.client = OpenAI(api_key=api_key)
            print("Using OpenAI client (requires valid API key and `openai` package).")
            print("Please uncomment and set up the OpenAI client properly for real use.")
            self.client = MockLLMClient() # Fallback to mock for demonstration
        else:
            print("No API key provided, using mock LLM client.")
            self.client = MockLLMClient()
        self.model = model

    def _call_llm(self, prompt_messages, json_output=False):
        response_format = {"type": "json_object"} if json_output else None
        try:
            response = self.client.chat_completions_create(
                model=self.model,
                messages=prompt_messages,
                response_format=response_format
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return json.dumps({"error": str(e)}) if json_output else f"Error: {e}"


class EssayGrader:
    def __init__(self, api_key=None, llm_model="gpt-4"):
        self.llm_interactor = LLMInteractionModule(api_key=api_key, model=llm_model)

    def generate_guidelines(self, essay_prompt: str) -> dict:
        prompt_messages = [
            {"role": "system", "content": "You are an AI assistant specialized in creating detailed and objective scoring rubrics for academic essays. Provide the rubric in JSON format."},
            {"role": "user", "content": f"Please generate a detailed scoring rubric and evaluation criteria in JSON format for the essay prompt:\n\n{essay_prompt}\n\nThe rubric should include a 'rubric_title', 'criteria' (each with 'name', 'weight', and 'description'), and a 'scoring_scale' (e.g., '0-100')."}
        ]
        guidelines_json_str = self.llm_interactor._call_llm(prompt_messages, json_output=True)
        try:
            return json.loads(guidelines_json_str)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from guideline generation: {guidelines_json_str}")
            return {"error": "Failed to parse guidelines from LLM", "raw_response": guidelines_json_str}

    def evaluate_essay(self, essay: str, original_prompt: str, guidelines: dict) -> dict:
        guidelines_str = json.dumps(guidelines, indent=2)
        prompt_messages = [
            {"role": "system", "content": "You are an AI assistant specialized in evaluating academic essays based on provided guidelines. Provide the evaluation in JSON format including an 'overall_score' (0-100), 'feedback' for each criterion, and a 'summary'."},
            {"role": "user", "content": f"Please evaluate the following essay based on the provided original prompt and evaluation guidelines. Provide an overall score (0-100) and specific feedback for each criterion in the guidelines, along with a summary, all in JSON format.\n\nOriginal Essay Prompt:\n{original_prompt}\n\nEvaluation Guidelines:\n{guidelines_str}\n\nStudent Essay:\n{essay}"}
        ]
        evaluation_json_str = self.llm_interactor._call_llm(prompt_messages, json_output=True)
        try:
            return json.loads(evaluation_json_str)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from essay evaluation: {evaluation_json_str}")
            return {"error": "Failed to parse evaluation from LLM", "raw_response": evaluation_json_str}

    def present_feedback(self, guidelines: dict, evaluation: dict):
        print("\n" + "="*50)
        print("MODEL-GENERATED EVALUATION REPORT")
        print("="*50 + "\n")

        if "error" in guidelines:
            print("Error in Guidelines Generation:", guidelines["error"])
            print("Raw LLM Response:", guidelines.get("raw_response", "N/A"))
            return

        if "error" in evaluation:
            print("Error in Essay Evaluation:", evaluation["error"])
            print("Raw LLM Response:", evaluation.get("raw_response", "N/A"))
            return

        print(f"Rubric Title: {guidelines.get('rubric_title', 'N/A')}")
        print(f"Scoring Scale: {guidelines.get('scoring_scale', 'N/A')}")
        print("\n--- Overall Score ---")
        print(f"Score: {evaluation.get('overall_score', 'N/A')}")
        print(f"Summary: {evaluation.get('summary', 'N/A')}")

        print("\n--- Detailed Feedback ---")
        for criterion in guidelines.get('criteria', []):
            name = criterion.get('name', 'N/A')
            description = criterion.get('description', 'N/A')
            feedback = evaluation.get('feedback', {}).get(name, 'No feedback provided.')
            print(f"\nCriterion: {name} (Weight: {criterion.get('weight', 'N/A')}%) ")
            print(f"  Description: {description}")
            print(f"  Feedback: {feedback}")

        print("\n" + "="*50)
        print("END OF REPORT")
        print("="*50 + "\n")


if __name__ == "__main__":
    # Set your OpenAI API key here or as an environment variable
    # For demonstration, we'll use a mock client if no key is provided.
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
    OPENAI_API_KEY = None # Set to your actual key if you want to use OpenAI

    grader = EssayGrader(api_key=OPENAI_API_KEY)

    # --- Step 1: Instructor provides an essay prompt ---
    essay_prompt_example = (
        "Discuss the impact of artificial intelligence on the future of work. "
        "Consider both potential benefits and challenges, providing specific examples "
        "to support your arguments. (Minimum 750 words)"
    )
    print("\n--- Generating Guidelines for Essay Prompt ---")
    print(f"Prompt: {essay_prompt_example[:100]}...")
    generated_guidelines = grader.generate_guidelines(essay_prompt_example)
    # print("\nGenerated Guidelines:", json.dumps(generated_guidelines, indent=2))

    # --- Step 2: Student submits an essay ---
    student_essay_example = (
        "Artificial intelligence is poised to revolutionize the global workforce, bringing both transformative benefits and significant challenges. "
        "One primary benefit is increased productivity and efficiency. AI-powered automation can streamline repetitive tasks, allowing human workers to focus on more complex, creative, and strategic endeavors. For instance, in manufacturing, robotic automation has already optimized production lines, leading to higher output and reduced costs. In customer service, AI chatbots handle routine inquiries, freeing up human agents for more nuanced interactions. This shift can lead to overall economic growth and the creation of new, higher-skilled jobs requiring human oversight, AI development, and data analysis. However, a major challenge is job displacement. As AI becomes more sophisticated, it will increasingly automate tasks traditionally performed by humans, not just in manual labor but also in white-collar professions like data entry, accounting, and even certain aspects of legal research. The World Economic Forum predicts significant job shifts due to AI, necessitating massive reskilling and upskilling initiatives to prevent widespread unemployment. Another concern is algorithmic bias. If AI systems are trained on biased data, they can perpetuate and even amplify existing societal inequalities in hiring, lending, and other critical areas, creating a less equitable future of work. Addressing this requires careful attention to data diversity and fairness in algorithm design. Furthermore, there's the ethical dilemma of decision-making by AI. As AI takes on more autonomous roles, questions arise about accountability and transparency when errors occur. In conclusion, while AI offers immense potential to enhance productivity and create new opportunities, its integration into the workforce demands proactive strategies to manage job displacement, mitigate bias, and establish clear ethical frameworks to ensure a just and prosperous future for all workers."
    )
    print("\n--- Evaluating Student Essay ---")
    print(f"Student Essay (excerpt): {student_essay_example[:100]}...")
    evaluated_feedback = grader.evaluate_essay(student_essay_example, essay_prompt_example, generated_guidelines)
    # print("\nEvaluated Feedback:", json.dumps(evaluated_feedback, indent=2))

    # --- Step 3: Present Feedback ---
    grader.present_feedback(generated_guidelines, evaluated_feedback)