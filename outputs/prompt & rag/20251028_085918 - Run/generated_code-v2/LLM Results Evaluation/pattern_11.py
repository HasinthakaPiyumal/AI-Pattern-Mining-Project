import json
from pydantic import BaseModel, ValidationError

class EvaluationResult(BaseModel):
    overall_score: int
    clarity_rating: int
    resolution_status: str  # e.g., "Resolved", "Unresolved", "Partial Resolution"
    feedback_summary: str
    areas_for_improvement: list[str]

def generate_llm_response_mock(prompt: str) -> str:
    # Simulate an LLM generating a JSON response based on the prompt.
    # In a real application, this would involve an actual API call to an LLM.
    if "transcript about a frustrated customer" in prompt:
        return json.dumps({
            "overall_score": 3,
            "clarity_rating": 3,
            "resolution_status": "Partial Resolution",
            "feedback_summary": "Agent attempted to resolve the issue but missed some nuances. Customer frustration was not fully addressed.",
            "areas_for_improvement": ["Empathy and active listening", "Probing deeper into customer needs", "Providing clearer next steps"]
        })
    else:
        return json.dumps({
            "overall_score": 5,
            "clarity_rating": 5,
            "resolution_status": "Resolved",
            "feedback_summary": "The agent provided excellent support, clearly explained the solution, and resolved the issue efficiently.",
            "areas_for_improvement": []
        })

def create_evaluation_prompt(transcript: str) -> str:
    return f"""Evaluate the following customer support transcript for quality, clarity, and resolution status. Provide your judgment in a JSON format with the following keys: 'overall_score' (1-5), 'clarity_rating' (1-5), 'resolution_status' (e.g., "Resolved", "Unresolved", "Partial Resolution"), 'feedback_summary', and 'areas_for_improvement' (a list of strings).\

Transcript: {transcript}

JSON Evaluation:"""

def evaluate_transcript(transcript: str) -> dict | None:
    prompt = create_evaluation_prompt(transcript)
    llm_raw_response = generate_llm_response_mock(prompt)

    try:
        evaluation_data = json.loads(llm_raw_response)
        parsed_evaluation = EvaluationResult(**evaluation_data)
        print(f"Successfully parsed LLM evaluation for transcript:\n---\n{transcript[:100]}...\n---")
        return parsed_evaluation.dict()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        print(f"Raw LLM response: {llm_raw_response}")
        return None
    except ValidationError as e:
        print(f"Error validating LLM response against schema: {e}")
        print(f"Raw LLM response: {llm_raw_response}")
        return None

if __name__ == "__main__":
    sample_transcript_good = (
        "Customer: My internet is not working. Agent: I understand, let\'s troubleshoot this. Can you restart your router? "
        "Customer: Yes, it\'s restarting now. Agent: Great, once it\'s back online, please check if you can access websites. "
        "Customer: It\'s working! Thank you! Agent: You\'re welcome! Is there anything else I can assist you with today? "
        "Customer: No, that\'s all. Agent: Have a great day!"
    )

    sample_transcript_bad = (
        "Customer: I\'m so frustrated, my new account setup is completely messed up and I can\'t log in. Agent: Did you follow the instructions? "
        "Customer: Yes, I\'ve tried everything. Agent: What\'s the error code? Customer: I don\'t see one, it just says \'invalid credentials\'. "
        "Agent: Hmm, try resetting your password. Customer: I did that three times! Agent: Okay, I\'ll open a ticket. "
        "Customer: How long will that take? Agent: I don\'t know. Customer: This is unacceptable!"
    )

    print("\n--- Evaluating Good Transcript ---")
    evaluation_good = evaluate_transcript(sample_transcript_good)
    if evaluation_good:
        print(json.dumps(evaluation_good, indent=2))

    print("\n--- Evaluating Bad Transcript ---")
    evaluation_bad = evaluate_transcript(sample_transcript_bad)
    if evaluation_bad:
        print(json.dumps(evaluation_bad, indent=2))

    print("\n--- Demonstrating Schema Validation Failure (Simulated) ---")
    # Simulate an LLM response that doesn't conform to the schema (e.g., wrong type for score)
    invalid_llm_response_mock = lambda p: json.dumps({
        "overall_score": "five", # Should be int
        "clarity_rating": 4,
        "resolution_status": "Resolved",
        "feedback_summary": "Okay",
        "areas_for_improvement": []
    })
    
    original_llm_mock = globals()['generate_llm_response_mock']
    globals()['generate_llm_response_mock'] = invalid_llm_response_mock # Temporarily override
    
    print("Attempting to evaluate with invalid LLM response:")
    evaluate_transcript("Customer: Hello. Agent: Hi.")

    globals()['generate_llm_response_mock'] = original_llm_mock # Restore original mock