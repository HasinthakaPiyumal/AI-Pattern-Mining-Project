from fastapi import FastAPI
from pydantic import BaseModel
import random
from collections import Counter
import uvicorn

app = FastAPI()

# --- 1. Data Simulation Module ---

def simulate_content_data(num_content_items: int = 100):
    content_data = []
    demographics = ["Male", "Female", "Non-binary"]
    topics = ["Politics", "Sports", "Tech", "Art", "Science"]
    sentiments = ["Positive", "Neutral", "Negative"]

    for i in range(num_content_items):
        content_data.append({
            "content_id": f"content_{i}",
            "creator_demographic": random.choice(demographics),
            "topic": random.choice(topics),
            "sentiment": random.choice(sentiments),
            "text": f"This is some interesting content about {random.choice(topics)} by a {random.choice(demographics)} creator with a {random.choice(sentiments)} sentiment."
        })
    return content_data

# --- 2. Fairness Analyzer & Demonstration Selector ---

def select_balanced_demonstrations(
    content_pool: list,
    num_demonstrations: int,
    fairness_targets: dict
) -> list:
    selected_demonstrations = []
    available_pool = list(content_pool)

    for _ in range(num_demonstrations):
        if not available_pool:
            break

        best_item = None
        best_score = -1

        current_demographics = Counter([d["creator_demographic"] for d in selected_demonstrations])
        current_topics = Counter([d["topic"] for d in selected_demonstrations])
        current_sentiments = Counter([d["sentiment"] for d in selected_demonstrations])

        for item in available_pool:
            temp_demographics = current_demographics + Counter([item["creator_demographic"]])
            temp_topics = current_topics + Counter([item["topic"]])
            temp_sentiments = current_sentiments + Counter([item["sentiment"]])

            score = 0
            # Simple scoring: reward items that bring distributions closer to targets
            # This is a basic heuristic and can be made more sophisticated
            for demographic, target_ratio in fairness_targets.get("creator_demographic", {}).items():
                if demographic in temp_demographics:
                    # Reward if it helps reach the target ratio relative to selected demonstrations count
                    score -= abs((temp_demographics[demographic] / (len(selected_demonstrations) + 1)) - target_ratio)

            for topic, target_ratio in fairness_targets.get("topic", {}).items():
                if topic in temp_topics:
                    score -= abs((temp_topics[topic] / (len(selected_demonstrations) + 1)) - target_ratio)

            for sentiment, target_ratio in fairness_targets.get("sentiment", {}).items():
                if sentiment in temp_sentiments:
                    score -= abs((temp_sentiments[sentiment] / (len(selected_demonstrations) + 1)) - target_ratio)

            if score > best_score:
                best_score = score
                best_item = item

        if best_item:
            selected_demonstrations.append(best_item)
            available_pool.remove(best_item)
        else:
            # If no item improves the score, pick a random one to avoid infinite loop
            if available_pool:
                selected_demonstrations.append(random.choice(available_pool))
                available_pool.remove(selected_demonstrations[-1])

    return selected_demonstrations

# --- 3. LLM Prompting & Recommendation Generation Module ---

def mock_llm_recommendation(prompt: str) -> str:
    # Simplified mock LLM response
    # In a real scenario, this would call an actual LLM API
    if "travel" in prompt.lower():
        return "Recommended: content_42, content_15, content_78 (related to travel)"
    if "tech" in prompt.lower():
        return "Recommended: content_05, content_22, content_91 (latest tech trends)"
    return "Recommended: content_01, content_02, content_03 (general trending)"

def generate_recommendations(
    user_preferences: dict,
    balanced_demonstrations: list,
    content_pool: list
) -> list:
    demonstration_texts = "\n".join([
        f"Example Content (Creator: {d['creator_demographic']}, Topic: {d['topic']}, Sentiment: {d['sentiment']}): {d['text']}"
        for d in balanced_demonstrations
    ])

    user_query = f"User preferences: {user_preferences.get('interests', 'general')}, sentiment: {user_preferences.get('preferred_sentiment', 'any')}."

    prompt = (
        f"Given the following content examples, recommend new content for a user.\n\n"
        f"{demonstration_texts}\n\n"
        f"Based on the above and the {user_query} \n"
        f"Please recommend 3 relevant content IDs. Example Format: Recommended: content_ID1, content_ID2, content_ID3"
    )

    llm_output = mock_llm_recommendation(prompt)

    # Parse LLM output (very basic parsing)
    recommended_ids = []
    if "Recommended:" in llm_output:
        parts = llm_output.split("Recommended:")[1].strip().split(", ")
        for part in parts:
            cleaned_id = part.split(" ")[0].replace("(", "").replace(")", "").strip()
            if cleaned_id.startswith("content_"):
                recommended_ids.append(cleaned_id)

    # Map IDs back to full content items
    recommendations = []
    content_map = {item["content_id"]: item for item in content_pool}
    for rec_id in recommended_ids:
        if rec_id in content_map:
            recommendations.append(content_map[rec_id])

    return recommendations

# --- 4. API Endpoint (FastAPI) ---

class UserPreferences(BaseModel):
    interests: str = "general"
    preferred_sentiment: str = "any"

@app.post("/recommend")
async def get_recommendations(user_prefs: UserPreferences):
    num_content_items = 200 # Simulate a larger pool
    content_pool = simulate_content_data(num_content_items)

    num_demonstrations = 5 # Number of few-shot examples

    fairness_targets = {
        "creator_demographic": {"Male": 0.33, "Female": 0.33, "Non-binary": 0.34},
        "topic": {"Politics": 0.2, "Sports": 0.2, "Tech": 0.2, "Art": 0.2, "Science": 0.2},
        "sentiment": {"Positive": 0.33, "Neutral": 0.33, "Negative": 0.34}
    }

    balanced_demos = select_balanced_demonstrations(
        content_pool=content_pool,
        num_demonstrations=num_demonstrations,
        fairness_targets=fairness_targets
    )

    recommendations = generate_recommendations(
        user_preferences=user_prefs.dict(),
        balanced_demonstrations=balanced_demos,
        content_pool=content_pool
    )

    return {"user_preferences": user_prefs.dict(), "recommendations": recommendations, "selected_demonstrations_count": len(balanced_demos)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)