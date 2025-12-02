
# -*- coding: utf-8 -*-

class KnowledgeGraph:
    def __init__(self):
        # A simplified knowledge graph represented as a dictionary of triples.
        # Each key is a subject, and its value is a list of (predicate, object) tuples.
        self.graph = {
            "fever": [
                ("is_symptom_of", "influenza"),
                ("is_symptom_of", "common_cold"),
                ("can_be_caused_by", "bacterial_infection")
            ],
            "cough": [
                ("is_symptom_of", "influenza"),
                ("is_symptom_of", "common_cold"),
                ("associated_with", "bronchitis")
            ],
            "sore_throat": [
                ("is_symptom_of", "common_cold"),
                ("is_symptom_of", "strep_throat")
            ],
            "influenza": [
                ("has_symptom", "fever"),
                ("has_symptom", "cough"),
                ("treated_by", "antivirals"),
                ("prevented_by", "flu_vaccine")
            ],
            "common_cold": [
                ("has_symptom", "fever"),
                ("has_symptom", "cough"),
                ("has_symptom", "sore_throat"),
                ("treated_by", "rest"),
                ("treated_by", "fluids")
            ],
            "strep_throat": [
                ("has_symptom", "sore_throat"),
                ("treated_by", "antibiotics")
            ],
            "antibiotics": [
                ("treats", "strep_throat"),
                ("treats", "bacterial_infection")
            ],
            "bacterial_infection": [
                ("can_cause", "fever"),
                ("treated_by", "antibiotics")
            ]
        }

    def get_paths_from_symptoms(self, symptoms, max_hops=2):
        """
        Simulates retrieving multi-hop reasoning paths from the KG based on symptoms.
        This is a simplified path retrieval for demonstration purposes.
        """
        reasoning_paths = []
        for symptom in symptoms:
            if symptom in self.graph:
                # First hop
                for pred1, obj1 in self.graph[symptom]:
                    path = [(symptom, pred1, obj1)]
                    reasoning_paths.append(path)

                    # Second hop (e.g., condition to treatment or other symptoms)
                    if max_hops > 1 and obj1 in self.graph:
                        for pred2, obj2 in self.graph[obj1]:
                            full_path = path + [(obj1, pred2, obj2)]
                            reasoning_paths.append(full_path)

        return reasoning_paths

def format_paths_as_triples(reasoning_paths):
    """
    Formats a list of reasoning paths into a triple-based string representation
    suitable for an LLM prompt.
    Example: 'symptom_X associated_with condition_Y; condition_Y treated_by treatment_Z'
    """
    formatted_prompts = []
    for path in reasoning_paths:
        triple_strings = []
        for subject, predicate, obj in path:
            triple_strings.append(f"{subject} {predicate} {obj}")
        formatted_prompts.append("; ".join(triple_strings))
    return formatted_prompts

def simulate_llm_reasoning(triple_based_prompt):
    """
    Simulates a Large Language Model's reasoning process based on the
    triple-based prompt. In a real application, this would be an API call to an LLM.
    """
    print(f"\n--- Simulating LLM Reasoning ---")
    print(f"Received structured knowledge: {triple_based_prompt}")
    # Simple heuristic to generate a mock diagnostic insight
    if "influenza treated_by antivirals" in triple_based_prompt and "fever is_symptom_of influenza" in triple_based_prompt:
        return "Based on symptoms and knowledge graph paths, influenza is a strong possibility. Antivirals may be considered, along with rest and fluids."
    elif "strep_throat treated_by antibiotics" in triple_based_prompt and "sore_throat is_symptom_of strep_throat" in triple_based_prompt:
        return "Given the sore throat and related paths, strep throat is a consideration. Antibiotics are typically prescribed."
    elif "common_cold treated_by rest" in triple_based_prompt:
        return "Symptoms suggest a common cold. Recommendations include rest and fluids."
    else:
        return "Further investigation may be needed. Consider general supportive care."


def main():
    print("Initializing Medical Diagnostic Assistant...")
    kg = KnowledgeGraph()

    patient_symptoms = ["fever", "cough"]
    print(f"\nPatient symptoms: {', '.join(patient_symptoms)}")

    # Step 1: Retrieve multi-hop reasoning paths from the Knowledge Graph
    print("\nRetrieving reasoning paths from Knowledge Graph...")
    reasoning_paths = kg.get_paths_from_symptoms(patient_symptoms, max_hops=2)
    # print("Raw reasoning paths:", reasoning_paths)

    if not reasoning_paths:
        print("No relevant reasoning paths found for the given symptoms.")
        return

    # Step 2: Format the retrieved paths into a triple-based representation
    print("\nFormatting paths into triple-based prompts for LLM...")
    triple_based_prompts = format_paths_as_triples(reasoning_paths)
    for i, prompt in enumerate(triple_based_prompts):
        print(f"  Path {i+1}: {prompt}")

    # Step 3: Send the structured prompts to a simulated LLM for reasoning
    print("\nSending structured prompts to simulated LLM for diagnostic insights...")
    diagnostic_insights = []
    for prompt in triple_based_prompts:
        insight = simulate_llm_reasoning(prompt)
        diagnostic_insights.append(insight)

    # Step 4: Present the LLM's diagnostic insights
    print("\n--- Final Diagnostic Insights from LLM ---")
    for insight in diagnostic_insights:
        print(f"- {insight}")
    print("-------------------------------------------")


if __name__ == "__main__":
    main()
