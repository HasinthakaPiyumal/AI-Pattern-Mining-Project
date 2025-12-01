import gradio as gr
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

openai.api_key = OPENAI_API_KEY

def call_llm_for_comparison(text_1: str, text_2: str) -> dict:
    """
    Calls the LLM to compare two texts and return a structured response.
    """
    prompt = f"""Compare the following two product descriptions based on quality, persuasiveness, clarity, and SEO-friendliness.
    
    Description 1:
    {text_1}
    
    Description 2:
    {text_2}
    
    Based on these criteria, which description is superior? Provide a clear judgment, a concise rationale, and individual scores for each description.
    
    Judgment: [e.g., "Description 1 is better", "Description 2 is better", "They are similar"]
    Rationale: [Explain your reasoning]
    Score Description 1 (1-10): [e.g., 8]
    Score Description 2 (1-10): [e.g., 7]
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", # Consider "gpt-4" for higher quality if available
            messages=[
                {"role": "system", "content": "You are a helpful assistant for evaluating product descriptions. Provide concise and clear responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, # Lower temperature for more consistent results
            max_tokens=500
        )
        
        raw_output = response.choices[0].message.content
        
        judgment = "N/A"
        rationale = "N/A"
        score_1 = "N/A"
        score_2 = "N/A"
        
        lines = raw_output.splitlines()
        for line in lines:
            if line.startswith("Judgment:"):
                judgment = line.replace("Judgment:", "").strip()
            elif line.startswith("Rationale:"):
                rationale = line.replace("Rationale:", "").strip()
            elif line.startswith("Score Description 1 (1-10):"):
                score_1 = line.replace("Score Description 1 (1-10):", "").strip()
            elif line.startswith("Score Description 2 (1-10):"):
                score_2 = line.replace("Score Description 2 (1-10):", "").strip()
                
        return {
            "judgment": judgment,
            "rationale": rationale,
            "score_1": score_1,
            "score_2": score_2,
            "raw_output": raw_output
        }
        
    except Exception as e:
        return {
            "judgment": f"Error during LLM call: {str(e)}",
            "rationale": "Could not complete LLM evaluation.",
            "score_1": "N/A",
            "score_2": "N/A",
            "raw_output": ""
        }

def get_judgment_score(judgment_string: str) -> int:
    """Assigns a numerical score to the LLM's judgment for internal use."""
    lower_judgment = judgment_string.lower()
    if "description 1 is better" in lower_judgment:
        return 1
    elif "description 2 is better" in lower_judgment:
        return -1
    elif "similar" in lower_judgment or "comparable" in lower_judgment or "equal" in lower_judgment:
        return 0
    return 0 # Default to similar if unclear

def ab_test_product_descriptions(desc_a: str, desc_b: str):
    """
    Performs A/B testing on two product descriptions using pairwise LLM evaluation with bias mitigation.
    """
    
    # Evaluation 1: A vs B
    eval_ab_result = call_llm_for_comparison(desc_a, desc_b)
    
    # Evaluation 2: B vs A (to mitigate order bias)
    eval_ba_result = call_llm_for_comparison(desc_b, desc_a)
    
    # Aggregate judgment scores
    judgment_score_ab = get_judgment_score(eval_ab_result["judgment"])
    judgment_score_ba_relative_to_A = -get_judgment_score(eval_ba_result["judgment"]) # Invert score for B vs A perspective

    total_judgment_score = judgment_score_ab + judgment_score_ba_relative_to_A

    final_judgment = ""
    final_rationale = ""

    if total_judgment_score > 0:
        final_judgment = "Description A is superior."
    elif total_judgment_score < 0:
        final_judgment = "Description B is superior."
    else:
        # total_judgment_score == 0 implies either both similar or conflicting/balanced
        # Check if actual judgments were conflicting (e.g., A better in first eval, B better in second)
        if (judgment_score_ab == 1 and get_judgment_score(eval_ba_result["judgment"]) == 1) or \
           (judgment_score_ab == -1 and get_judgment_score(eval_ba_result["judgment"]) == -1):
            final_judgment = "Conflicting judgments detected. Descriptions are likely similar, or the difference is subtle. Consider them comparable."
        else:
            final_judgment = "Both descriptions are evaluated as similar in quality, or the preference is balanced after order bias mitigation."

    final_rationale = f"Evaluation (A vs B): {eval_ab_result['judgment']}\nRationale: {eval_ab_result['rationale']}\n\n" \
                      f"Evaluation (B vs A): {eval_ba_result['judgment']}\nRationale: {eval_ba_result['rationale']}\n\n" \
                      f"Final Composite Judgment Score (A vs B): {total_judgment_score} (Positive means A better, Negative means B better, Zero means Similar/Conflicting)"
    
    # Aggregate individual scores
    score_a_ab = float(eval_ab_result["score_1"]) if eval_ab_result["score_1"].isdigit() else None
    score_b_ab = float(eval_ab_result["score_2"]) if eval_ab_result["score_2"].isdigit() else None
    
    # Note: For eval_ba_result, score_1 is for desc_b, score_2 is for desc_a
    score_a_ba = float(eval_ba_result["score_2"]) if eval_ba_result["score_2"].isdigit() else None
    score_b_ba = float(eval_ba_result["score_1"]) if eval_ba_result["score_1"].isdigit() else None

    final_score_a = "N/A"
    final_score_b = "N/A"

    scores_for_a = [s for s in [score_a_ab, score_a_ba] if s is not None]
    if scores_for_a:
        final_score_a = str(round(sum(scores_for_a) / len(scores_for_a), 1))

    scores_for_b = [s for s in [score_b_ab, score_b_ba] if s is not None]
    if scores_for_b:
        final_score_b = str(round(sum(scores_for_b) / len(scores_for_b), 1))
    
    return final_judgment, final_rationale, final_score_a, final_score_b

# Gradio Interface
iface = gr.Interface(
    fn=ab_test_product_descriptions,
    inputs=[
        gr.Textbox(lines=10, label="Product Description A", placeholder="Enter the first product description here...", value="Experience ultimate comfort with our ergonomic office chair. Designed for long hours of work, it provides superior lumbar support and breathable mesh material to keep you cool. Easy to assemble and adjust to your perfect height."),
        gr.Textbox(lines=10, label="Product Description B", placeholder="Enter the second product description here...", value="Our new office chair is a game-changer for your productivity. Featuring a sleek design and advanced back support, this chair is perfect for any modern workspace. Crafted with premium materials for durability and style. Enhance your work setup today!")
    ],
    outputs=[
        gr.Textbox(label="Final Judgment"),
        gr.Textbox(label="Overall Rationale"),
        gr.Textbox(label="Average Score for Description A (1-10)"),
        gr.Textbox(label="Average Score for Description B (1-10)")
    ],
    title="E-commerce Product Description A/B Testing Platform",
    description="Compare two product descriptions using an LLM for quality, persuasiveness, clarity, and SEO-friendliness. Order bias is mitigated by evaluating both A vs B and B vs A. Scores are averaged across evaluations."
)

if __name__ == "__main__":
    iface.launch()