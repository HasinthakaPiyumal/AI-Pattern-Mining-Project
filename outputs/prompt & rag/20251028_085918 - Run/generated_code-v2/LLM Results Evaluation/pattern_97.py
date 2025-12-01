import gradio as gr

def simulate_llm_evaluation(customer_query: str, agent_response: str, criteria: list) -> list:
    results = []

    for criterion in criteria:
        judgment = "Good"
        explanation = "The agent's response generally meets this criterion based on basic keyword analysis."

        if "accurate" in criterion.lower() or "correct" in criterion.lower():
            if any(word.lower() in agent_response.lower() for word in customer_query.split() if len(word) > 3):
                judgment = "Excellent"
                explanation = "The response appears to be accurate and addresses key elements from the customer query."
            elif len(agent_response) < len(customer_query) / 2:
                judgment = "Needs Improvement"
                explanation = "The response might lack sufficient detail to be fully accurate or comprehensive."
            else:
                judgment = "Good"
                explanation = "The accuracy seems reasonable, though a deeper semantic check would be ideal."
        elif "empathetic" in criterion.lower() or "tone" in criterion.lower():
            empathy_keywords = ["apologize", "understand", "sorry", "regret", "concern", "appreciate"]
            if any(keyword in agent_response.lower() for keyword in empathy_keywords):
                judgment = "Excellent"
                explanation = "The response uses empathetic language and shows understanding of the customer's situation."
            elif "rudely" in agent_response.lower() or "impolite" in agent_response.lower(): # Basic negative tone detection
                judgment = "Needs Improvement"
                explanation = "The tone of the response may not be sufficiently empathetic or could be perceived negatively."
            else:
                judgment = "Good"
                explanation = "The tone appears neutral and professional, exhibiting a reasonable level of empathy."
        elif "resolve" in criterion.lower() or "issue" in criterion.lower():
            resolution_keywords = ["resolved", "fixed", "solution", "addressed", "completed", "taken care of"]
            if any(keyword in agent_response.lower() for keyword in resolution_keywords):
                judgment = "Excellent"
                explanation = "The response clearly indicates a resolution to the customer's issue."
            elif "unable to" in agent_response.lower() or "can't fix" in agent_response.lower():
                judgment = "Needs Improvement"
                explanation = "The response does not seem to fully resolve the customer's issue or indicates an inability to do so."
            else:
                judgment = "Good"
                explanation = "The response contributes towards resolving the issue, but a clearer resolution might be beneficial."
        elif "guidelines" in criterion.lower() or "policy" in criterion.lower():
            judgment = "Good"
            explanation = "The response is assumed to adhere to company guidelines, as no explicit violations are detected in this simulated check."

        results.append({"criterion": criterion, "judgment": judgment, "explanation": explanation})
    return results

def evaluate_agent_response(customer_query: str, agent_response: str, evaluation_criteria_input: str):
    if not customer_query or not agent_response or not evaluation_criteria_input:
        return "", "Please provide all inputs (Customer Query, Agent Response, Evaluation Criteria).", ""

    criteria = [c.strip() for c in evaluation_criteria_input.split(',') if c.strip()]
    if not criteria:
        return "", "Please provide at least one evaluation criterion.", ""

    detailed_evaluations = simulate_llm_evaluation(customer_query, agent_response, criteria)

    overall_judgment_scores = {"Excellent": 3, "Good": 2, "Needs Improvement": 1}
    min_score = 3
    overall_explanation_parts = []
    detailed_markdown = "### Detailed Criteria Evaluation\n\n"

    for eval_item in detailed_evaluations:
        judgment = eval_item["judgment"]
        explanation = eval_item["explanation"]
        criterion = eval_item["criterion"]

        if overall_judgment_scores.get(judgment, 0) < min_score:
            min_score = overall_judgment_scores.get(judgment, 0)

        overall_explanation_parts.append(f"- {criterion}: {judgment}. {explanation}")
        detailed_markdown += f"**{criterion}:** {judgment}\n  *Explanation:* {explanation}\n\n"

    if min_score == 3:
        overall_judgment = "Excellent"
        overall_summary = "The agent's response is outstanding across all evaluated criteria."
    elif min_score == 2:
        overall_judgment = "Good"
        overall_summary = "The agent's response is generally good, meeting most criteria effectively."
    else:
        overall_judgment = "Needs Improvement"
        overall_summary = "The agent's response requires improvement in one or more key areas."

    return overall_judgment, overall_summary, detailed_markdown

interface = gr.Interface(
    fn=evaluate_agent_response,
    inputs=[
        gr.Textbox(label="Customer Query", lines=3, placeholder="e.g., My internet is not working. I need help."),
        gr.Textbox(label="Agent Response", lines=5, placeholder="e.g., I apologize for the inconvenience. I've initiated a diagnostic. Please restart your router and I'll check the status."),
        gr.Textbox(label="Evaluation Criteria (comma-separated)", placeholder="e.g., Is the response accurate?, Is the tone empathetic?, Does it resolve the customer's issue?, Does it adhere to company guidelines?")
    ],
    outputs=[
        gr.Textbox(label="Overall Judgment"),
        gr.Textbox(label="Overall Explanation"),
        gr.Markdown(label="Detailed Criteria Evaluation")
    ],
    title="Automated Customer Support Response Quality Evaluator",
    description="Evaluate customer support agent responses using a simulated LLM autorater. Enter the customer's query, the agent's response, and specific evaluation criteria."
)

if __name__ == "__main__":
    interface.launch()