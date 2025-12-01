import time

class SimulatedLLMAgent:
    def __init__(self):
        pass

    def process_patient_data(self, symptoms, history, initial_results, previous_lab_results=None):
        rationale = ""
        proposed_tests = []
        diagnosis = ""
        confident = False

        if "fever" in symptoms.lower() and "cough" in symptoms.lower():
            rationale = "Considering fever and cough, common respiratory infections like flu or a cold are possibilities. To differentiate, a viral panel might be helpful."
            proposed_tests = ["viral_panel_test"]
        elif "abdominal pain" in symptoms.lower() and "nausea" in symptoms.lower():
            rationale = "Abdominal pain and nausea could indicate various gastrointestinal issues. An ultrasound or blood work to check inflammatory markers would be prudent."
            proposed_tests = ["abdominal_ultrasound", "cbc_inflammatory_markers"]
        elif previous_lab_results and "positive for influenza A" in previous_lab_results.lower():
            rationale = "Given the positive influenza A result from the viral panel, the patient likely has the flu. Symptomatic treatment and monitoring are recommended."
            diagnosis = "Influenza A"
            confident = True
        elif previous_lab_results and "elevated crp" in previous_lab_results.lower() and "normal ultrasound" in previous_lab_results.lower():
            rationale = "Elevated CRP suggests inflammation. With a normal abdominal ultrasound, further investigation into inflammatory bowel disease or other systemic inflammation causes might be needed. A stool test could be useful."
            proposed_tests = ["stool_calprotectin_test"]
        elif previous_lab_results and "stool calprotectin elevated" in previous_lab_results.lower():
            rationale = "Elevated stool calprotectin strongly suggests intestinal inflammation, possibly inflammatory bowel disease. Gastroenterology consultation is recommended."
            diagnosis = "Possible Inflammatory Bowel Disease"
            confident = True
        else:
            rationale = "Initial assessment suggests a general illness. Further information or broad screening might be beneficial."
            proposed_tests = ["basic_blood_work"]

        return rationale, proposed_tests, diagnosis, confident

class SimulatedExternalLabTool:
    def __init__(self):
        self.test_database = {
            "viral_panel_test": "Results: Positive for Influenza A",
            "abdominal_ultrasound": "Results: Normal abdominal organs, no acute findings.",
            "cbc_inflammatory_markers": "Results: White blood cell count slightly elevated (12.5 x 10^9/L), CRP elevated (35 mg/L).",
            "basic_blood_work": "Results: All parameters within normal limits.",
            "stool_calprotectin_test": "Results: Stool calprotectin elevated (250 ug/g), suggesting intestinal inflammation."
        }

    def execute_test(self, test_program):
        print(f"Simulating lab test: {test_program}...")
        time.sleep(1)
        return self.test_database.get(test_program.lower(), "Results: Test not found or inconclusive.")

class DiagnosticOrchestrationLoop:
    def __init__(self, llm_agent, lab_tool, max_turns=5):
        self.llm_agent = llm_agent
        self.lab_tool = lab_tool
        self.max_turns = max_turns

    def run_diagnosis(self, patient_symptoms, patient_history, patient_initial_results):
        print("\n--- Starting Medical Diagnostic Assistant ---")
        current_lab_results = None
        turn = 0
        final_diagnosis = ""

        while turn < self.max_turns:
            print(f"\n--- Turn {turn + 1} ---")
            print(f"Patient Symptoms: {patient_symptoms}")
            print(f"Patient History: {patient_history}")
            print(f"Initial Results: {patient_initial_results}")
            if current_lab_results:
                print(f"Previous Lab Results: {current_lab_results}")

            rationale, proposed_tests, diagnosis, confident = self.llm_agent.process_patient_data(
                patient_symptoms, patient_history, patient_initial_results, current_lab_results
            )

            print(f"LLM Rationale: {rationale}")

            if confident:
                final_diagnosis = diagnosis
                print(f"LLM confidently diagnosed: {final_diagnosis}")
                break

            if not proposed_tests:
                print("LLM did not propose any further tests and could not reach a confident diagnosis.")
                print("Final diagnosis (tentative): " + (diagnosis if diagnosis else "Undetermined"))
                break

            print(f"LLM Proposed Tests: {', '.join(proposed_tests)}")

            all_test_results_this_turn = []
            for test in proposed_tests:
                test_output = self.lab_tool.execute_test(test)
                all_test_results_this_turn.append(test_output)

            current_lab_results = " ".join(all_test_results_this_turn)
            print(f"Combined Lab Tool Output: {current_lab_results}")

            turn += 1

        if not final_diagnosis and turn == self.max_turns:
            print("\n--- Max turns reached without a confident diagnosis. ---")
            print("Please consult with a human specialist.")
        elif final_diagnosis:
            print("\n--- Diagnosis Complete ---")
            print(f"Final Confident Diagnosis: {final_diagnosis}")

        return final_diagnosis


if __name__ == "__main__":
    llm = SimulatedLLMAgent()
    lab = SimulatedExternalLabTool()
    orchestrator = DiagnosticOrchestrationLoop(llm, lab)

    patient_case_1 = {
        "symptoms": "High fever, persistent cough, body aches",
        "history": "No significant medical history, unvaccinated for flu this season",
        "initial_results": "Normal chest X-ray"
    }

    patient_case_2 = {
        "symptoms": "Severe abdominal pain, nausea, loss of appetite",
        "history": "History of occasional indigestion",
        "initial_results": "Blood pressure normal, slight tenderness on palpation of abdomen"
    }

    patient_case_3 = {
        "symptoms": "Fatigue, mild joint pain, recurrent diarrhea",
        "history": "Family history of autoimmune conditions",
        "initial_results": "Routine blood work normal"
    }

    print("\n--- Running Patient Case 1 ---")
    orchestrator.run_diagnosis(
        patient_case_1["symptoms"],
        patient_case_1["history"],
        patient_case_1["initial_results"]
    )

    print("\n\n--- Running Patient Case 2 ---")
    orchestrator.run_diagnosis(
        patient_case_2["symptoms"],
        patient_case_2["history"],
        patient_case_2["initial_results"]
    )

    print("\n\n--- Running Patient Case 3 ---")
    orchestrator.run_diagnosis(
        patient_case_3["symptoms"],
        patient_case_3["history"],
        patient_case_3["initial_results"]
    )
