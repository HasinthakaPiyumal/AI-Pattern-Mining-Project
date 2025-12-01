import os
import json
from dotenv import load_dotenv
import openai

class LLMEvaluator:
    def __init__(self, llm_client: openai.OpenAI, evaluation_schema: dict):
        self.llm_client = llm_client
        self.evaluation_schema = evaluation_schema

    def _generate_prompt(self, conversation_history: list) -> str:
        prompt_parts = [
            "You are an expert evaluator of customer support conversations. Your task is to assess the quality of the following conversation based on specific criteria.",
            "Output your evaluation as a JSON object, where keys are the evaluation criteria and values are integer scores within the specified range.",
            "\nEvaluation Criteria and Scoring Ranges:"
        ]

        for criterion, details in self.evaluation_schema.items():
            prompt_parts.append(f"- {criterion.capitalize()}: {details['description']} (Score range: {details['range']['min']}-{details['range']['max']})")
        
        prompt_parts.append("\nConversation History:")
        for message in conversation_history:
            prompt_parts.append(f"{message['role'].capitalize()}: {message['content']}")
        
        prompt_parts.append("\nNow, provide your evaluation as a JSON object:")
        return "\n".join(prompt_parts)

    def evaluate_conversation(self, conversation_history: list) -> dict:
        prompt = self._generate_prompt(conversation_history)
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."}, 
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            evaluation_output = response.choices[0].message.content
            scores = json.loads(evaluation_output)
            return scores
        except openai.APIError as e:
            print(f"OpenAI API error: {e}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}\nRaw LLM output: {evaluation_output}")
            return {"error": f"Failed to parse LLM output: {e}", "raw_output": evaluation_output}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    llm_client = openai.OpenAI(api_key=openai_api_key)

    evaluation_schema = {
        "grammar": {"description": "Is the language grammatically correct and fluent?", "range": {"min": 1, "max": 5}},
        "relevance": {"description": "Is the chatbot's response directly addressing the user's query?", "range": {"min": 1, "max": 5}},
        "helpfulness": {"description": "Does the chatbot's response provide useful and actionable information?", "range": {"min": 1, "max": 5}},
        "empathy": {"description": "Does the chatbot show understanding and appropriate tone?", "range": {"min": 1, "max": 5}},
        "completeness": {"description": "Does the chatbot fully answer the user's request without needing further prompts?", "range": {"min": 1, "max": 5}}
    }

    evaluator = LLMEvaluator(llm_client=llm_client, evaluation_schema=evaluation_schema)

    sample_conversations = [
        [
            {"role": "user", "content": "Hi, my internet is not working."}, 
            {"role": "assistant", "content": "I understand you're having internet issues. Can you please restart your router?"}
        ],
        [
            {"role": "user", "content": "What's the weather like today in London?"},
            {"role": "assistant", "content": "The current stock price of Apple is $170."} 
        ],
        [
            {"role": "user", "content": "I'm feeling really frustrated with this product."}, 
            {"role": "assistant", "content": "Oh, that's not good. I'm sorry to hear that. Can you tell me more about what's going on?"}
        ]
    ]

    print("\n--- Evaluating Sample Conversations ---")
    for i, conversation in enumerate(sample_conversations):
        print(f"\nConversation {i + 1}:")
        for msg in conversation:
            print(f"  {msg['role'].capitalize()}: {msg['content']}")
        
        scores = evaluator.evaluate_conversation(conversation)
        print(f"  Evaluation Scores: {json.dumps(scores, indent=2)}")
        print("-" * 30)
