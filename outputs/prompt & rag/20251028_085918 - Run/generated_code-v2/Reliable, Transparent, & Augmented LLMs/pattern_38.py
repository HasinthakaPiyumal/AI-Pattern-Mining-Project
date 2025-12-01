import pandas as pd
import numpy as np
import random
from collections import defaultdict

def generate_synthetic_resume_data(num_resumes=1000):
    genders = ['Male', 'Female', 'Non-binary']
    ethnicities = ['Caucasian', 'African American', 'Asian', 'Hispanic']
    education_levels = ['High School', 'Bachelors', 'Masters', 'PhD']

    data = {
        'resume_id': range(num_resumes),
        'text': [f"Resume {i} skilled in {random.choice(['Python', 'Java', 'Data Science'])} and {random.choice(['Leadership', 'Teamwork'])}" for i in range(num_resumes)],
        'gender': random.choices(genders, weights=[0.45, 0.45, 0.10], k=num_resumes),
        'ethnicity': random.choices(ethnicities, weights=[0.4, 0.25, 0.2, 0.15], k=num_resumes),
        'education_level': random.choices(education_levels, weights=[0.1, 0.4, 0.3, 0.2], k=num_resumes)
    }
    df = pd.DataFrame(data)

    # Simulate a biased hiring decision: e.g., slightly higher hiring rate for Male/Caucasian
    # This bias is what we aim to mitigate in the LLM's few-shot learning.
    df['hired'] = 0 # Default to not hired

    # Bias towards Male and Caucasian for higher skills
    for i in range(num_resumes):
        if 'Python' in df.loc[i, 'text'] and df.loc[i, 'gender'] == 'Male' and random.random() < 0.7:
            df.loc[i, 'hired'] = 1
        elif 'Data Science' in df.loc[i, 'text'] and df.loc[i, 'ethnicity'] == 'Caucasian' and random.random() < 0.6:
            df.loc[i, 'hired'] = 1
        elif random.random() < 0.3: # Base hiring rate for others
            df.loc[i, 'hired'] = 1

    return df

def select_balanced_demonstrations(data, num_demos_per_class_and_attr=2, sensitive_attributes=['gender', 'ethnicity']):
    demonstrations = []
    # Separate data by hiring outcome
    hired_data = data[data['hired'] == 1].copy()
    not_hired_data = data[data['hired'] == 0].copy()

    for outcome_data in [hired_data, not_hired_data]:
        outcome_value = outcome_data['hired'].iloc[0] if not outcome_data.empty else None
        if outcome_value is None: # Handle case where there are no hired/not_hired examples
            continue

        # Group by sensitive attributes and try to select balanced demos
        groups = defaultdict(list)
        for _, row in outcome_data.iterrows():
            key = tuple(row[attr] for attr in sensitive_attributes)
            groups[key].append(row.to_dict())

        for group_key, group_demos in groups.items():
            selected_from_group = random.sample(group_demos, min(num_demos_per_class_and_attr, len(group_demos)))
            demonstrations.extend(selected_from_group)
            
    # Shuffle to mix positive and negative examples
    random.shuffle(demonstrations)
    return demonstrations

def generate_few_shot_prompt(demonstrations, candidate_resume):
    prompt = "Evaluate the following candidate resume based on the provided examples.\n\n"

    for demo in demonstrations:
        outcome = "Hired" if demo['hired'] == 1 else "Not Hired"
        prompt += f"Example Resume:\nText: {demo['text']}\nOutcome: {outcome}\n---\n"
    
    prompt += f"\nCandidate Resume:\nText: {candidate_resume['text']}\nOutcome:"
    return prompt

def simulate_llm_prediction(prompt, candidate_gender, demo_bias_factor=0.2):
    # A very simple simulation: if the candidate resume has 'Python' or 'Data Science', lean towards hired.
    # However, introduce a bias based on gender, which can be mitigated by balanced demos.
    predicted_hired = 0 # Default to Not Hired

    # Check for positive keywords in the candidate part of the prompt
    candidate_text_start = prompt.rfind("Candidate Resume:\nText: ") + len("Candidate Resume:\nText: ")
    candidate_text_end = prompt.rfind("\nOutcome:")
    candidate_text = prompt[candidate_text_start:candidate_text_end]
    
    if 'Python' in candidate_text or 'Data Science' in candidate_text:
        predicted_hired = 1
    
    # Simulate LLM's sensitivity to demonstration bias:
    # Analyze demonstrations within the prompt to infer a bias.
    demo_hired_genders = []
    for line in prompt.split('\n'):
        if "Example Resume:" in line: # Start of an example
            current_demo_outcome = None
            current_demo_gender = None
        elif "Outcome:" in line:
            if "Hired" in line: current_demo_outcome = 1
            else: current_demo_outcome = 0
            # This simulation assumes gender is not explicitly in prompt, but we know it from demos
            # For a real LLM, you'd need to extract gender from demonstration text if embedded.
            # Here, we'll infer it from the *intended* demonstration set.
        # This simplistic simulation requires direct access to demo info for bias.
        # A more robust simulation would parse demo text for gender.
        # For this demo, let's assume `candidate_gender` is available and we simulate based on *demonstration-level* bias.

    # Let's make the 'LLM' infer a bias directly from the prompt's examples count for 'Hired'
    # This is a highly simplified way to make the LLM 'learn' bias from demos
    male_hired_demos = prompt.count("Gender: Male\nOutcome: Hired") # Simplified extraction assuming gender in demo text
    female_hired_demos = prompt.count("Gender: Female\nOutcome: Hired")

    # If the demonstrations are heavily skewed towards one gender being hired, amplify the prediction for that gender
    if male_hired_demos > female_hired_demos + 1 and candidate_gender == 'Male':
        if random.random() < demo_bias_factor: # Small chance to flip to hired if male and demo is biased male
            predicted_hired = 1
    elif female_hired_demos > male_hired_demos + 1 and candidate_gender == 'Female':
        if random.random() < demo_bias_factor: # Small chance to flip to hired if female and demo is biased female
            predicted_hired = 1

    return predicted_hired

def evaluate_bias(predictions_df, sensitive_attribute):
    # Calculate Disparate Impact Ratio for the sensitive_attribute
    # This is defined as (P(hired|Group A) / P(hired|Group B))
    # We'll consider the ratio of hiring rates for different groups.

    if predictions_df.empty:
        return {sensitive_attribute: {'ratio': 1.0, 'details': 'No predictions to evaluate'}}

    attribute_values = predictions_df[sensitive_attribute].unique()
    if len(attribute_values) < 2:
        return {sensitive_attribute: {'ratio': 1.0, 'details': f'Only one group for {sensitive_attribute}'}}

    hiring_rates = {}
    for attr_val in attribute_values:
        group_df = predictions_df[predictions_df[sensitive_attribute] == attr_val]
        if not group_df.empty:
            hiring_rate = group_df['predicted_hired'].mean()
            hiring_rates[attr_val] = hiring_rate
        else:
            hiring_rates[attr_val] = 0.0

    # Find the maximum and minimum hiring rates to calculate the ratio
    if not hiring_rates:
        return {sensitive_attribute: {'ratio': 1.0, 'details': 'No hiring rates to compare'}}

    max_rate = 0.0
    min_rate = float('inf')
    max_group = None
    min_group = None

    for group, rate in hiring_rates.items():
        if rate > max_rate:
            max_rate = rate
            max_group = group
        if rate < min_rate:
            min_rate = rate
            min_group = group
    
    if min_rate == 0 and max_rate == 0: # Avoid division by zero if no one was hired
        disparate_impact_ratio = 1.0
    elif min_rate == 0: # If minimum rate is 0, and max rate is > 0, ratio is inf (highly biased)
        disparate_impact_ratio = float('inf')
    else:
        disparate_impact_ratio = max_rate / min_rate
    
    details = {
        'hiring_rates': hiring_rates,
        'max_hiring_group': max_group,
        'min_hiring_group': min_group
    }
    return {sensitive_attribute: {'ratio': disparate_impact_ratio, 'details': details}}

# --- Main Execution --- #
if __name__ == "__main__":
    print("Starting Fair Resume Screening Assistant simulation...")

    # 1. Generate Synthetic Data
    print("Generating synthetic resume data...")
    synthetic_data = generate_synthetic_resume_data(num_resumes=1000)
    # print("\nSynthetic Data Sample:")
    # print(synthetic_data.head())
    # print("\nHiring distribution in synthetic data:")
    # print(synthetic_data.groupby(['gender', 'hired']).size().unstack(fill_value=0))

    # Split data into demonstration pool and test set
    demonstration_pool, test_data = synthetic_data[:800], synthetic_data[800:].copy()
    print(f"Generated {len(demonstration_pool)} demonstration examples and {len(test_data)} test examples.")

    sensitive_attributes_to_balance = ['gender', 'ethnicity']
    num_demonstrations_per_scenario = 6 # Total demonstrations for each LLM call

    # --- Scenario 1: Unbalanced Demonstrations (Random Selection) ---
    print("\n--- Scenario 1: Using Unbalanced (Randomly Selected) Demonstrations ---")
    unbalanced_demonstrations = random.sample(demonstration_pool.to_dict('records'), num_demonstrations_per_scenario)

    unbalanced_predictions = []
    for _, candidate in test_data.iterrows():
        prompt = generate_few_shot_prompt(unbalanced_demonstrations, candidate)
        # For simulation, pass candidate gender to simulate_llm_prediction to show effect of demonstration bias
        predicted_hired = simulate_llm_prediction(prompt, candidate['gender'], demo_bias_factor=0.3) # Higher bias factor
        unbalanced_predictions.append({
            'resume_id': candidate['resume_id'],
            'gender': candidate['gender'],
            'ethnicity': candidate['ethnicity'],
            'true_hired': candidate['hired'],
            'predicted_hired': predicted_hired
        })
    unbalanced_predictions_df = pd.DataFrame(unbalanced_predictions)
    # print("\nUnbalanced Predictions Sample:")
    # print(unbalanced_predictions_df.head())

    print("Evaluating bias with unbalanced demonstrations...")
    unbalanced_bias_results = {}
    for attr in sensitive_attributes_to_balance:
        unbalanced_bias_results.update(evaluate_bias(unbalanced_predictions_df, attr))
    
    for attr, result in unbalanced_bias_results.items():
        print(f"  Unbalanced Bias (Disparate Impact Ratio for {attr}): {result['ratio']:.2f}")
        # print(f"    Details: {result['details']}")


    # --- Scenario 2: Balanced Demonstrations ---
    print("\n--- Scenario 2: Using Balanced Demonstrations ---")
    # Need to adjust `num_demos_per_class_and_attr` to get `num_demonstrations_per_scenario` total
    # If 2 classes (hired/not hired) and 2 sensitive attributes, 2*2 groups. 
    # So, num_demos_per_class_and_attr = total_demos / (num_outcomes * num_gender_groups * num_ethnicity_groups)
    # This is getting complex, let's simplify for the demo.

    # Let's target getting `num_demonstrations_per_scenario` in total.
    # The `select_balanced_demonstrations` will try to balance based on available groups.
    # For a simple demo, we'll try to get 1 demo per combination (hired/not_hired x gender x ethnicity)
    # if num_demonstrations_per_scenario is 6, and we have (2 outcomes * 2 genders * 2 ethnicities = 8 ideal groups)
    # this is tricky to guarantee a precise total.

    # Let's simplify select_balanced_demonstrations to pick 'num_demos_per_group' for each group (e.g. 'Male Hired', 'Female Not Hired', etc.)
    # and then limit the overall count for the prompt.

    # Re-calling `select_balanced_demonstrations` with a refined strategy:
    # We'll aim for a specific number of demonstrations that are balanced. 
    # For this simulation, we'll ensure a minimum count per sensitive group in *hired* and *not_hired* categories.
    
    # Simplified: Get all balanced examples possible and then sample from them if too many.
    balanced_demonstrations_full_set = select_balanced_demonstrations(demonstration_pool, num_demos_per_class_and_attr=1, sensitive_attributes=sensitive_attributes_to_balance)
    
    # Now, if we have more than `num_demonstrations_per_scenario`, we sample randomly from the balanced set.
    if len(balanced_demonstrations_full_set) > num_demonstrations_per_scenario:
        balanced_demonstrations = random.sample(balanced_demonstrations_full_set, num_demonstrations_per_scenario)
    else:
        balanced_demonstrations = balanced_demonstrations_full_set
        # If not enough balanced demos, fill with random ones, but prioritize balanced.
        remaining_needed = num_demonstrations_per_scenario - len(balanced_demonstrations)
        if remaining_needed > 0:
            extra_demos = random.sample(demonstration_pool.to_dict('records'), remaining_needed)
            balanced_demonstrations.extend(extra_demos)

    print(f"Selected {len(balanced_demonstrations)} balanced demonstrations.")
    # print("\nBalanced Demonstrations Sample:")
    # for demo in balanced_demonstrations[:3]: print(demo)

    balanced_predictions = []
    for _, candidate in test_data.iterrows():
        prompt = generate_few_shot_prompt(balanced_demonstrations, candidate)
        # For simulation, pass candidate gender to simulate_llm_prediction to show effect of demonstration bias
        predicted_hired = simulate_llm_prediction(prompt, candidate['gender'], demo_bias_factor=0.05) # Lower bias factor
        balanced_predictions.append({
            'resume_id': candidate['resume_id'],
            'gender': candidate['gender'],
            'ethnicity': candidate['ethnicity'],
            'true_hired': candidate['hired'],
            'predicted_hired': predicted_hired
        })
    balanced_predictions_df = pd.DataFrame(balanced_predictions)
    # print("\nBalanced Predictions Sample:")
    # print(balanced_predictions_df.head())

    print("Evaluating bias with balanced demonstrations...")
    balanced_bias_results = {}
    for attr in sensitive_attributes_to_balance:
        balanced_bias_results.update(evaluate_bias(balanced_predictions_df, attr))

    for attr, result in balanced_bias_results.items():
        print(f"  Balanced Bias (Disparate Impact Ratio for {attr}): {result['ratio']:.2f}")
        # print(f"    Details: {result['details']}")

    print("\n--- Bias Mitigation Comparison ---")
    for attr in sensitive_attributes_to_balance:
        unbalanced_ratio = unbalanced_bias_results.get(attr, {}).get('ratio', 1.0)
        balanced_ratio = balanced_bias_results.get(attr, {}).get('ratio', 1.0)
        print(f"For '{attr}':")
        print(f"  Unbalanced Demonstrations DI Ratio: {unbalanced_ratio:.2f}")
        print(f"  Balanced Demonstrations DI Ratio: {balanced_ratio:.2f}")
        if unbalanced_ratio > balanced_ratio:
            print(f"  Bias for {attr} was reduced with balanced demonstrations. (Lower ratio is better towards 1.0)")
        elif unbalanced_ratio < balanced_ratio:
            print(f"  Bias for {attr} increased with balanced demonstrations (this might indicate a need for more sophisticated balancing or simulation).")
        else:
            print(f"  Bias for {attr} remained similar.")

    print("Simulation complete.")
