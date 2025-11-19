import json

class MockLLM:
    def __init__(self, name="MockLLM"):
        self.name = name

    def invoke(self, prompt):
        if "initial plan" in prompt.lower():
            return "Proposed initial plan: Adjust blood pressure medication dosage, monitor glucose levels, recommend low-sodium diet."
        elif "refine plan" in prompt.lower() and "good" in prompt.lower():
            return "Refined plan: Continue blood pressure medication adjustment, introduce personalized exercise regimen, reinforce dietary changes."
        elif "refine plan" in prompt.lower() and "bad" in prompt.lower():
            return "Refined plan: Re-evaluate blood pressure medication, investigate alternative glucose monitoring, consult nutritionist for diet plan revision."
        elif "reflect" in prompt.lower():
            return "Reflection: The plan was somewhat effective but needs further personalization based on patient response."
        return "Default LLM response based on prompt."

def check_drug_interaction(drug1: str, drug2: str) -> dict:
    if "insulin" in drug1.lower() and "beta-blocker" in drug2.lower():
        return {"tool_name": "DrugInteractionChecker", "result": f"Potential interaction between {drug1} and {drug2}: increased risk of hypoglycemia."}
    return {"tool_name": "DrugInteractionChecker", "result": f"No significant interaction found between {drug1} and {drug2}."}

def get_medical_guidelines(disease: str) -> dict:
    if "hypertension" in disease.lower():
        return {"tool_name": "MedicalGuidelineRetriever", "result": "Hypertension guidelines: DASH diet, regular exercise, medication titration based on BP readings."}
    return {"tool_name": "MedicalGuidelineRetriever", "result": f"No specific guidelines found for {disease} in mock database."}

def simulate_patient_feedback(current_plan: str, iteration: int) -> dict:
    if iteration == 1:
        return {"source": "Patient", "feedback": "Blood pressure improved slightly, but feeling tired. Diet is hard to follow.", "sentiment": "mixed"}
    elif iteration == 2:
        return {"source": "Doctor", "feedback": "Glucose levels stable, but BP is still high for target. Patient compliance with diet is low.", "sentiment": "bad"}
    return {"source": "System", "feedback": "No new feedback.", "sentiment": "neutral"}

class AdaptiveAgent:
    def __init__(self, llm):
        self.llm = llm
        self.current_plan = None
        self.feedback_history = []
        self.iteration = 0
        self.tools = {
            "check_drug_interaction": check_drug_interaction,
            "get_medical_guidelines": get_medical_guidelines,
            "simulate_patient_feedback": simulate_patient_feedback,
        }

    def run_tool(self, tool_name: str, **kwargs):
        if tool_name in self.tools:
            print(f"Executing tool: {tool_name} with args {kwargs}")
            return self.tools[tool_name](**kwargs)
        return {"error": f"Tool '{tool_name}' not found."}

    def generate_plan(self, prompt_context: str):
        prompt = f"Given the following context: {prompt_context}, please generate a detailed treatment plan."
        response = self.llm.invoke(prompt)
        self.current_plan = response
        return self.current_plan

    def process_feedback(self):
        feedback = self.run_tool("simulate_patient_feedback", current_plan=self.current_plan, iteration=self.iteration)
        self.feedback_history.append(feedback)
        print(f"Received feedback: {feedback}")
        return feedback

    def reflect_and_refine(self):
        reflection_prompt = f"Reflect on the current plan '{self.current_plan}' and the feedback: {json.dumps(self.feedback_history[-1])}. What worked, what didn't, and how can the plan be improved?"
        reflection = self.llm.invoke(reflection_prompt)
        print(f"Agent Reflection: {reflection}")

        refine_prompt = f"Refine the treatment plan based on this reflection and the latest feedback: {json.dumps(self.feedback_history[-1])}. Current plan: '{self.current_plan}'. Refined plan should address issues found."
        refined_plan = self.llm.invoke(refine_prompt)
        self.current_plan = refined_plan
        print(f"Refined plan: {self.current_plan}")
        return refined_plan

    def optimize_treatment_plan(self, max_iterations=3):
        print("--- Starting Treatment Plan Optimization ---")
        initial_context = "Patient has chronic hypertension and pre-diabetes. Needs a personalized treatment plan."
        self.current_plan = self.generate_plan(initial_context)
        print(f"Initial Plan: {self.current_plan}")

        for i in range(max_iterations):
            self.iteration = i + 1
            print(f"\n--- Iteration {self.iteration} ---")

            # Simulate tool usage for information gathering
            guidelines = self.run_tool("get_medical_guidelines", disease="hypertension")
            print(f"Tool Output: {guidelines}")

            drug_check = self.run_tool("check_drug_interaction", drug1="insulin", drug2="beta-blocker")
            print(f"Tool Output: {drug_check}")

            # Process Feedback
            feedback = self.process_feedback()
            if feedback["sentiment"] == "neutral" and self.iteration > 1:
                print("No new significant feedback, potentially converging or needs external input.")
                break # Termination condition based on feedback

            # Reflect and Refine
            self.reflect_and_refine()

            # Example termination condition (simplified)
            if "stable" in self.current_plan.lower() and "optimized" in self.current_plan.lower():
                print("Plan seems optimized. Terminating iterative process.")
                break

        print("\n--- Optimization Complete ---")
        print(f"Final Optimized Plan: {self.current_plan}")

if __name__ == "__main__":
    mock_llm = MockLLM()
    agent = AdaptiveAgent(mock_llm)
    agent.optimize_treatment_plan(max_iterations=3)
