import json
from llm_model import LLMModel
from uncertainty_calculator import UncertaintyCalculator
from data_manager import DataManager
from human_annotator_interface import HumanAnnotatorInterface

def run_active_prompting_loop(num_iterations=3):
    data_manager = DataManager("medical_exemplars.json")
    llm = LLMModel()
    uncertainty_calculator = UncertaintyCalculator()
    human_interface = HumanAnnotatorInterface()

    # Initialize with some dummy exemplars
    initial_exemplars = [
        {"id": 1, "patient_case": "25-year-old male with persistent cough and fever.", "cot_reasoning": "Influenza is common, consider antibiotics if bacterial infection suspected.", "diagnosis": "Flu"},
        {"id": 2, "patient_case": "60-year-old female with sudden severe headache and stiff neck.", "cot_reasoning": "Could be migraine, but stiff neck is concerning.", "diagnosis": "Migraine"}
    ]
    data_manager.save_exemplars(initial_exemplars)

    print("--- Starting Active Prompting Loop ---")

    for i in range(num_iterations):
        print(f"\nIteration {i + 1}:")
        exemplars = data_manager.load_exemplars()
        
        if not exemplars:
            print("No exemplars found. Exiting.")
            break

        # Step 1: LLM solves exemplars and generates CoT reasoning
        llm_outputs = []
        for exemplar in exemplars:
            prompt = f"Patient Case: {exemplar['patient_case']}\nTask: Provide Chain-of-Thought reasoning and a diagnosis for complex rare diseases. Focus on a CoT style output.\n"
            simulated_cot, simulated_diagnosis = llm.generate_cot_and_diagnosis(prompt)
            llm_outputs.append({
                "id": exemplar["id"],
                "original_case": exemplar["patient_case"],
                "llm_cot": simulated_cot,
                "llm_diagnosis": simulated_diagnosis
            })
        
        print(f"LLM processed {len(llm_outputs)} exemplars.")

        # Step 2: Calculate uncertainty for LLM outputs
        uncertainty_scores = uncertainty_calculator.calculate_uncertainty(llm_outputs)
        print(f"Calculated uncertainty for {len(uncertainty_scores)} exemplars.")

        # Step 3: Identify top N uncertain exemplars for human review
        num_to_review = min(2, len(uncertainty_scores)) # Review up to 2 per iteration for demo
        sorted_uncertainty = sorted(uncertainty_scores.items(), key=lambda item: item[1], reverse=True)
        exemplars_for_review_ids = [item[0] for item in sorted_uncertainty[:num_to_review]]

        print(f"Identified exemplars {exemplars_for_review_ids} for human review (top {num_to_review} most uncertain).")

        exemplars_to_be_refined = []
        for output in llm_outputs:
            if output["id"] in exemplars_for_review_ids:
                exemplars_to_be_refined.append(output)

        # Step 4: Human annotators rewrite/refine exemplars
        if exemplars_to_be_refined:
            refined_exemplars = human_interface.review_and_refine(exemplars_to_be_refined)
            
            # Update original exemplars with refined ones
            updated_exemplars = []
            refined_ids = {r["id"] for r in refined_exemplars}
            for original_exemplar in exemplars:
                if original_exemplar["id"] in refined_ids:
                    # Find the refined version
                    for refined in refined_exemplars:
                        if refined["id"] == original_exemplar["id"]:
                            updated_exemplars.append({
                                "id": refined["id"],
                                "patient_case": refined["original_case"], # Keep original case description
                                "cot_reasoning": refined["human_refined_cot"], # Use human refined CoT
                                "diagnosis": refined["human_refined_diagnosis"] # Use human refined diagnosis
                            })
                            break
                else:
                    updated_exemplars.append(original_exemplar)
            
            data_manager.save_exemplars(updated_exemplars)
            print(f"Human experts refined {len(refined_exemplars)} exemplars. Updated exemplars saved.")
        else:
            print("No exemplars to refine in this iteration.")

    print("--- Active Prompting Loop Finished ---")
    print("Final exemplars after refinement:")
    print(json.dumps(data_manager.load_exemplars(), indent=2))

if __name__ == "__main__":
    run_active_prompting_loop()