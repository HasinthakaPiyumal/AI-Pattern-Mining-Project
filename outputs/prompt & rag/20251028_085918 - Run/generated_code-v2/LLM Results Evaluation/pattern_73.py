import json

class MockLLM:
    def generate(self, prompt):
        if "Empathetic Customer" in prompt:
            score = 5 if "empathy" in prompt and "polite" in prompt else 3
            feedback = "The response was somewhat empathetic and polite." if score == 3 else "The response demonstrated strong empathy and politeness."
        elif "Technical Expert" in prompt:
            score = 4 if "accuracy" in prompt and "complete" in prompt else 2
            feedback = "The technical information provided was partially accurate." if score == 2 else "The technical details were accurate and comprehensive."
        elif "Business Goal Aligner" in prompt:
            score = 3 if "policy" in prompt and "retention" in prompt else 1
            feedback = "The response somewhat aligned with business goals." if score == 1 else "The response strongly aligned with business policies and customer retention."
        elif "Clarity and Conciseness Checker" in prompt:
            score = 5 if "clear" in prompt and "concise" in prompt and "grammatical errors" not in prompt else 4
            feedback = "The response was mostly clear and concise." if score == 4 else "The response was extremely clear, concise, and error-free."
        else:
            score = 3
            feedback = "General evaluation."

        return {"score": score, "feedback": feedback}

class Agent:
    def __init__(self, name, role, evaluation_criteria, llm):
        self.name = name
        self.role = role
        self.evaluation_criteria = evaluation_criteria
        self.llm = llm

    def evaluate(self, customer_response, debate_history):
        prompt = f"As a {self.role}, evaluate the following customer support response based on these criteria: {self.evaluation_criteria}.\nCustomer Response: {customer_response}\n\nDebate History so far:\n"
        for entry in debate_history:
            prompt += f"- Agent {entry['agent_name']} (Score: {entry['evaluation']['score']}): {entry['evaluation']['feedback']}\n"
        
        evaluation_result = self.llm.generate(prompt)
        return evaluation_result

class ChatEvalFramework:
    def __init__(self, agents):
        self.agents = agents

    def run_debate(self, customer_support_response, num_rounds=1):
        debate_history = []
        all_individual_evaluations = []

        for _ in range(num_rounds):
            for agent in self.agents:
                evaluation = agent.evaluate(customer_support_response, debate_history)
                debate_entry = {
                    "agent_name": agent.name,
                    "role": agent.role,
                    "evaluation": evaluation
                }
                debate_history.append(debate_entry)
                all_individual_evaluations.append(debate_entry)

        return self._summarize_results(all_individual_evaluations)

    def _summarize_results(self, evaluations):
        total_scores = [e['evaluation']['score'] for e in evaluations]
        
        score_distribution = {
            "average_score": sum(total_scores) / len(total_scores) if total_scores else 0,
            "min_score": min(total_scores) if total_scores else 0,
            "max_score": max(total_scores) if total_scores else 0
        }

        detailed_feedback = {}
        for agent_name in set([e['agent_name'] for e in evaluations]):
            agent_feedback = [e['evaluation']['feedback'] for e in evaluations if e['agent_name'] == agent_name]
            detailed_feedback[agent_name] = agent_feedback

        overall_conclusion = "The overall quality of the response is satisfactory based on the multi-agent debate." # Simplified conclusion
        if score_distribution['average_score'] < 3:
            overall_conclusion = "The overall quality of the response is poor and requires significant improvement."
        elif score_distribution['average_score'] >= 4:
            overall_conclusion = "The overall quality of the response is excellent."

        return {
            "overall_conclusion": overall_conclusion,
            "total_evaluations": len(evaluations),
            "score_distribution": score_distribution,
            "detailed_feedback_by_agent": detailed_feedback,
            "raw_individual_evaluations": evaluations
        }


if __name__ == "__main__":
    mock_llm = MockLLM()

    agent_empathetic = Agent(
        name="Agent Persona 1",
        role="Empathetic Customer",
        evaluation_criteria="Does the response address the customer's emotional state, show understanding, and use polite language? (Score 1-5)",
        llm=mock_llm
    )

    agent_technical = Agent(
        name="Agent Persona 2",
        role="Technical Expert",
        evaluation_criteria="Assess the accuracy and completeness of the technical information provided in the response. (Score 1-5)",
        llm=mock_llm
    )

    agent_business = Agent(
        name="Agent Persona 3",
        role="Business Goal Aligner",
        evaluation_criteria="Check if the response adheres to company policies, promotes customer retention, or guides the customer towards a desired action. (Score 1-5)",
        llm=mock_llm
    )

    agent_clarity = Agent(
        name="Agent Persona 4",
        role="Clarity and Conciseness Checker",
        evaluation_criteria="Review the response for grammatical errors, clarity of language, and conciseness, ensuring it's easy to understand. (Score 1-5)",
        llm=mock_llm
    )

    agents = [agent_empathetic, agent_technical, agent_business, agent_clarity]
    chateval_framework = ChatEvalFramework(agents)

    customer_response_good = "Thank you for reaching out! I understand this issue is frustrating. To resolve your internet connectivity, please ensure your router is plugged in and restart it. This usually fixes the problem. If not, visit our support page for more advanced troubleshooting. We appreciate your patience."
    customer_response_bad = "Router broken. Restart it. Go to website if no fix."

    print("\n--- Evaluating a GOOD customer support response ---")
    evaluation_results_good = chateval_framework.run_debate(customer_response_good, num_rounds=2)
    print(json.dumps(evaluation_results_good, indent=2))

    print("\n--- Evaluating a BAD customer support response ---")
    evaluation_results_bad = chateval_framework.run_debate(customer_response_bad, num_rounds=2)
    print(json.dumps(evaluation_results_bad, indent=2))