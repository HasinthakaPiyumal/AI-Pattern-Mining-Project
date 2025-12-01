import json
from typing import List, Dict, Any, Optional
from collections import defaultdict

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)
        def dict(self):
            return self.__dict__
        def json(self, **kwargs):
            return json.dumps(self.dict(), **kwargs)

    class Field:
        def __init__(self, *args, **kwargs):
            pass
        def __get__(self, instance, owner):
            return self
        def __call__(self, *args, **kwargs):
            return self

class EvaluationCriteria(BaseModel):
    name: str = Field(description="Name of the evaluation criterion (e.g., 'grammar', 'relevance')")
    description: str = Field(description="Detailed description of what to evaluate for this criterion")
    min_score: int = Field(default=1, description="Minimum possible score for this criterion")
    max_score: int = Field(default=5, description="Maximum possible score for this criterion")

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, EvaluationCriteria):
            return NotImplemented
        return self.name == other.name


DEFAULT_EVAL_CRITERIA: List[EvaluationCriteria] = [
    EvaluationCriteria(name="grammar", description="Grammatical correctness and fluency of the chatbot's responses.", min_score=1, max_score=5),
    EvaluationCriteria(name="relevance", description="How relevant are the chatbot's responses to the user's queries?", min_score=1, max_score=5),
    EvaluationCriteria(name="helpfulness", description="Does the chatbot provide useful information or solutions to the user's problem?", min_score=1, max_score=5),
    EvaluationCriteria(name="empathy", description="Does the chatbot show understanding and appropriate emotional intelligence?", min_score=1, max_score=5),
    EvaluationCriteria(name="conciseness", description="Are the chatbot's responses brief and to the point without unnecessary verbosity?", min_score=1, max_score=5),
]

class EvaluationScore(BaseModel):
    score: int = Field(description="The score for this criterion within the defined range")
    reasoning: str = Field(description="Brief explanation for the given score")

class LLMEvaluationOutput(BaseModel):
    evaluation: Dict[str, EvaluationScore] = Field(description="Dictionary of evaluation scores and reasoning for each criterion")

class PromptGenerator:
    def __init__(self, criteria: List[EvaluationCriteria]):
        self.criteria = criteria

    def _generate_criteria_schema(self) -> str:
        schema_parts = []
        for criterion in self.criteria:
            schema_parts.append(
                f'"{criterion.name}": {{ "score": [int, {criterion.min_score}-{criterion.max_score}], "reasoning": "string" }}'
            )
        return ",\n        ".join(schema_parts)

    def generate_prompt(self, conversation_log: List[Dict[str, str]]) -> str:
        conversation_text = ""
        for turn in conversation_log:
            conversation_text += f"{turn['speaker']}: {turn['text']}\n"

        criteria_schema = self._generate_criteria_schema()
        output_format_example = {
            "evaluation": {
                self.criteria[0].name: {"score": 3, "reasoning": "Example reasoning for grammar."},
                self.criteria[1].name: {"score": 4, "reasoning": "Example reasoning for relevance."}
            }
        }
        
        example_eval = {}
        for crit in self.criteria:
            example_eval[crit.name] = {"score": crit.min_score + (crit.max_score - crit.min_score) // 2, "reasoning": f"Example reasoning for {crit.name}."}
        output_format_example = {"evaluation": example_eval}


        prompt = f"""
You are an expert chatbot evaluator. Your task is to analyze the following conversation between a user and a chatbot and provide a multidimensional evaluation based on the given criteria.

**Conversation Log:**
```
{conversation_text.strip()}
```

**Evaluation Criteria:**
{chr(10).join([f"- {c.name.capitalize()} (Score {c.min_score}-{c.max_score}): {c.description}" for c in self.criteria])}

**Instructions:**
1.  For each criterion, assign a score between {self.criteria[0].min_score} and {self.criteria[0].max_score}.
2.  Provide a brief, concise reasoning for each score.
3.  Output your evaluation in a JSON object strictly following the schema below. Do not include any other text before or after the JSON.

**Output JSON Schema:**
```json
{{
    "evaluation": {{
        {criteria_schema}
    }}
}}
```

**Example Output:**
```json
{json.dumps(output_format_example, indent=4)}
```
"""
        return prompt.strip()

class LLMClient:
    def __init__(self, mock_response_data: Optional[Dict[str, Any]] = None):
        self.mock_response_data = mock_response_data

    def get_evaluation(self, prompt: str) -> str:
        if self.mock_response_data:
            return json.dumps(self.mock_response_data)

        mock_eval = {}
        for criterion in DEFAULT_EVAL_CRITERIA:
            mock_eval[criterion.name] = {
                "score": 3,
                "reasoning": f"The chatbot's {criterion.name} was generally acceptable."
            }
        mock_response = {"evaluation": mock_eval}
        return json.dumps(mock_response)

class ResponseParser:
    def parse_response(self, json_string: str, criteria: List[EvaluationCriteria]) -> Optional[LLMEvaluationOutput]:
        try:
            data = json.loads(json_string)
            if "evaluation" not in data:
                return None
            
            parsed_evaluation = {}
            for crit in criteria:
                if crit.name in data["evaluation"]:
                    score_data = data["evaluation"][crit.name]
                    if "score" in score_data and "reasoning" in score_data:
                        score = int(score_data["score"])
                        if not (crit.min_score <= score <= crit.max_score):
                            print(f"Warning: Score for '{crit.name}' is out of range ({score}). Expected {crit.min_score}-{crit.max_score}.")
                            score = max(crit.min_score, min(crit.max_score, score))
                        
                        parsed_evaluation[crit.name] = EvaluationScore(
                            score=score,
                            reasoning=score_data["reasoning"]
                        )
            return LLMEvaluationOutput(evaluation=parsed_evaluation)
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response JSON: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during parsing: {e}")
            return None

def load_conversation_logs(filepath: str) -> List[List[Dict[str, str]]]:
    if filepath == "mock_logs.json":
        return [
            [
                {"speaker": "User", "text": "Hi, I have a problem with my order #12345."},
                {"speaker": "Chatbot", "text": "Hello! I can assist you with that. Could you please provide your full name and the email address associated with the order?"},
                {"speaker": "User", "text": "My name is John Doe and my email is john.doe@example.com."},
                {"speaker": "Chatbot", "text": "Thank you, John. Let me check the status of order #12345 for you."},
                {"speaker": "User", "text": "Great, thanks!"},
                {"speaker": "Chatbot", "text": "I see that your order #12345 was shipped yesterday and is expected to arrive within 3-5 business days. Is there anything else I can help you with today?"},
                {"speaker": "User", "text": "No, that's all. Thanks for your help!"}
            ],
            [
                {"speaker": "User", "text": "I can't log into my account. I forgot my password."},
                {"speaker": "Chatbot", "text": "I understand. To reset your password, please visit our website and click on the 'Forgot Password' link on the login page."},
                {"speaker": "User", "text": "I already tried that, but I didn't receive the email."},
                {"speaker": "Chatbot", "text": "Oh, I apologize for the inconvenience. Please check your spam folder. If it's not there, I can help you manually reset it. For security, I'll need to ask a few verification questions. Could you please tell me the last 4 digits of the phone number associated with your account?"},
                {"speaker": "User", "text": "It's 1234. I'm feeling very frustrated right now."},
                {"speaker": "Chatbot", "text": "I understand your frustration, and I'm truly sorry for the trouble this has caused. Let me just verify your details quickly, and we'll get this sorted for you."},
                {"speaker": "User", "text": "Okay, thanks."
                }
            ]
        ]
    return []

def generate_report(evaluations: List[LLMEvaluationOutput], criteria: List[EvaluationCriteria]):
    print("\n--- Evaluation Report ---")
    if not evaluations:
        print("No evaluations to report.")
        return

    aggregated_scores = defaultdict(lambda: defaultdict(int))
    total_conversations = len(evaluations)

    for eval_output in evaluations:
        for crit_name, eval_score in eval_output.evaluation.items():
            aggregated_scores[crit_name]["sum_scores"] += eval_score.score
            aggregated_scores[crit_name]["count"] += 1

    print(f"Total Conversations Evaluated: {total_conversations}\n")

    for criterion in criteria:
        crit_name = criterion.name
        if crit_name in aggregated_scores and aggregated_scores[crit_name]["count"] > 0:
            avg_score = aggregated_scores[crit_name]["sum_scores"] / aggregated_scores[crit_name]["count"]
            print(f"{crit_name.capitalize()} (Avg Score: {avg_score:.2f}/{criterion.max_score})")
            print(f"  Description: {criterion.description}")
        else:
            print(f"{crit_name.capitalize()}: No evaluations found.")
    
    print("\n--- Detailed Evaluation ---")
    for i, eval_output in enumerate(evaluations):
        print(f"\nConversation {i+1} Evaluation:")
        for crit_name, eval_score in eval_output.evaluation.items():
            print(f"  {crit_name.capitalize()}: Score {eval_score.score}, Reason: {eval_score.reasoning}")

def main():
    print("Starting Chatbot Evaluation System...")

    conversation_logs = load_conversation_logs("mock_logs.json")
    if not conversation_logs:
        print("No conversation logs found. Exiting.")
        return

    print(f"Loaded {len(conversation_logs)} conversation logs.")

    criteria = DEFAULT_EVAL_CRITERIA

    prompt_generator = PromptGenerator(criteria=criteria)
    response_parser = ResponseParser()

    all_evaluations: List[LLMEvaluationOutput] = []

    for i, convo in enumerate(conversation_logs):
        print(f"\nProcessing Conversation {i+1}...")
        prompt = prompt_generator.generate_prompt(conversation_log=convo)

        if i == 0:
            mock_llm_response_data = {
                "evaluation": {
                    "grammar": {"score": 5, "reasoning": "Chatbot's grammar was perfect."},
                    "relevance": {"score": 4, "reasoning": "Responses were relevant to user's queries."},
                    "helpfulness": {"score": 5, "reasoning": "Successfully resolved the user's order inquiry."},
                    "empathy": {"score": 3, "reasoning": "Neutral tone, could be slightly more empathetic."},
                    "conciseness": {"score": 4, "reasoning": "Responses were generally concise, no unnecessary verbosity."
                    }
                }
            }
        else:
             mock_llm_response_data = {
                "evaluation": {
                    "grammar": {"score": 4, "reasoning": "Minor grammatical error in one turn."},
                    "relevance": {"score": 3, "reasoning": "Initially struggled to provide a direct solution, needed redirection."},
                    "helpfulness": {"score": 3, "reasoning": "Eventually offered a manual reset, but user had to prompt for it."},
                    "empathy": {"score": 4, "reasoning": "Acknowledged user frustration and apologized."},
                    "conciseness": {"score": 3, "reasoning": "Responses were a bit wordy, could be more direct."
                    }
                }
            }
        
        llm_client = LLMClient(mock_response_data=mock_llm_response_data)
        llm_raw_response = llm_client.get_evaluation(prompt)
        
        parsed_evaluation = response_parser.parse_response(llm_raw_response, criteria)

        if parsed_evaluation:
            all_evaluations.append(parsed_evaluation)
            print("Evaluation successful.")
        else:
            print("Evaluation failed for this conversation.")

    generate_report(all_evaluations, criteria)
    print("\nChatbot Evaluation System Finished.")

if __name__ == "__main__":
    main()