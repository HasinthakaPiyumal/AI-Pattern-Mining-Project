import os
import json
from openai import OpenAI

class LLMEvaluator:
    """A framework for unified, multidimensional automatic evaluation of open-domain conversations using an LLM."""

    def __init__(self, evaluation_criteria: dict, score_range: tuple = (1, 5), model_name: str = "gpt-4"):
        """
        Initializes the LLMEvaluator with evaluation criteria and scoring parameters.

        Args:
            evaluation_criteria (dict): A dictionary where keys are criterion names (e.g., 'relevance')
                                        and values are their descriptions (e.g., 'How relevant is the response?').
            score_range (tuple): A tuple (min_score, max_score) defining the allowed score range.
            model_name (str): The name of the OpenAI model to use for evaluation.
        """
        self.evaluation_criteria = evaluation_criteria
        self.min_score, self.max_score = score_range
        self.model_name = model_name

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key)

    def _generate_prompt(self, conversation_history: list) -> str:
        """
        Generates the evaluation prompt for the LLM based on conversation history and criteria.

        Args:
            conversation_history (list): A list of dictionaries representing the conversation,
                                         e.g., [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there!'}]

        Returns:
            str: The formatted prompt string.
        """
        criteria_description = "\n".join(
            [f"- {name}: {description}" for name, description in self.evaluation_criteria.items()]
        )

        conversation_text = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history]
        )

        prompt = f"""
        You are an AI assistant designed to evaluate customer support chatbot conversations.
        Your task is to evaluate the following conversation based on the criteria provided below.

        For each criterion, assign a score between {self.min_score} and {self.max_score}.
        Provide your output as a JSON object where keys are the criterion names and values are the scores.

        Evaluation Criteria:
        {criteria_description}

        Conversation History:
        {conversation_text}

        Please provide your evaluation in JSON format:
        """
        return prompt

    def evaluate_conversation(self, conversation_history: list) -> dict:
        """
        Evaluates a given conversation using the configured LLM.

        Args:
            conversation_history (list): The list of dictionaries representing the conversation.

        Returns:
            dict: A dictionary containing the evaluation scores for each criterion.
                  Returns an empty dictionary if evaluation fails or parsing error occurs.
        """
        prompt = self._generate_prompt(conversation_history)

        messages = [
            {"role": "system", "content": "You are an expert evaluator."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0 # For consistent evaluation
            )
            evaluation_str = response.choices[0].message.content
            evaluation_results = json.loads(evaluation_str)
            return evaluation_results
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response: {e}")
            print(f"Raw LLM response: {evaluation_str}")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred during LLM evaluation: {e}")
            return {}

# Example usage would typically be in a separate script or main function.