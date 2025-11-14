from typing import Dict, Any

class SynthesisModule:
    def __init__(self):
        pass

    def synthesize_evaluations(self, evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        total_score = 0
        feedback_report = """Comprehensive Essay Feedback Report
===================================

"""
        num_agents = len(evaluations)

        for persona, result in evaluations.items():
            score = result.get("score", 0)
            feedback = result.get("feedback", "No feedback provided.")

            total_score += score
            feedback_report += f"""--- {persona} Perspective ---
Score: {score}/100
Feedback: {feedback}

"""

        consolidated_grade = total_score / num_agents if num_agents > 0 else 0
        feedback_report += f"""--- Overall Assessment ---
Consolidated Grade: {consolidated_grade:.2f}/100

Summary: This grade is an average across all evaluative perspectives. The detailed feedback above provides insights from each specialized agent to help you understand your strengths and areas for improvement.
"""

        return {
            "consolidated_grade": consolidated_grade,
            "feedback_report": feedback_report,
            "individual_evaluations": evaluations
        }