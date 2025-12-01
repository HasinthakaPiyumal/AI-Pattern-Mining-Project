import random
import time

def mock_llm_batch_process(reviews: list[str], prompt_template: str, task: str) -> list[dict]:
    time.sleep(0.1) # Simulate LLM inference time
    results = []
    for review in reviews:
        response = {"review": review}
        if task == "moderation":
            status = random.choice(["spam", "not_spam"])
            response["moderation_status"] = status
        elif task == "categorization":
            category = random.choice(["positive", "negative", "feature_request"])
            response["category"] = category
        results.append(response)
    return results