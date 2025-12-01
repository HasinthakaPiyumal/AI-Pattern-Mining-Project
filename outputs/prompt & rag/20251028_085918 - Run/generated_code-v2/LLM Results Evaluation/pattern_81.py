from pydantic import BaseModel
import json
from openai import OpenAI

class EvaluationResult(BaseModel):
    score: int
    explanation: str
    accuracy: bool
    completeness: bool
    empathy: bool
    conciseness: bool
    grammar_and_spelling: bool

class CustomerSupportEvaluator:
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.evaluation_criteria_map = {
            "accuracy": "Is the response factually correct and relevant to the query?",
            "completeness": "Does the response address all parts of the customer's query?",
            "empathy": "Is the tone appropriate and empathetic?",
            "conciseness": "Is the response clear and to the point without unnecessary verbosity?",
            "grammar_and_spelling": "Is the response free of grammatical errors and typos?"
        }

    def _generate_prompt(self, query: str, response: str) -> str:
        criteria_bullet_points = ""
        for key, value in self.evaluation_criteria_map.items():
            criteria_bullet_points += f"- {key.replace('_', ' ').capitalize()}: {value}\n"

        return f"""
        You are an AI assistant designed to evaluate the quality of customer support responses.
        Your task is to assess an LLM-generated response to a customer query based on the following criteria:

        Criteria:
        {criteria_bullet_points}

        Customer Query:
        {query}

        LLM-Generated Response:
        {response}

        Please provide a numerical score from 1 to 5 (1 being very poor, 5 being excellent) for the overall quality,
        a detailed explanation for your score, and a boolean value for each specified criterion indicating if it was met (True/False).
        Your output MUST be a JSON object matching the following Pydantic schema:
        {{
            "score": int,
            "explanation": str,
            "accuracy": bool,
            "completeness": bool,
            "empathy": bool,
            "conciseness": bool,
            "grammar_and_spelling": bool
        }}
        """

    def evaluate_response(self, customer_query: str, llm_response: str) -> EvaluationResult:
        prompt = self._generate_prompt(customer_query, llm_response)

        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            llm_output = chat_completion.choices[0].message.content
            parsed_output = json.loads(llm_output)
            evaluation_result = EvaluationResult(**parsed_output)
            return evaluation_result
        except Exception as e:
            print(f"Error during LLM evaluation: {e}")
            return EvaluationResult(
                score=1,
                explanation=f"Evaluation failed due to an error: {e}",
                accuracy=False,
                completeness=False,
                empathy=False,
                conciseness=False,
                grammar_and_spelling=False
            )