import json
from typing import List, Dict, Any, Tuple

# --- 1. Simulated Knowledge Graph (KG) ---
class MedicalKnowledgeGraph:
    def __init__(self):
        # A simple KG: {entity: {relation: [connected_entity, ...]}}
        self.graph = {
            "fever": {"is_symptom_of": ["influenza", "common_cold", "bacterial_infection"]},
            "cough": {"is_symptom_of": ["influenza", "common_cold", "bronchitis"]},
            "sore_throat": {"is_symptom_of": ["common_cold", "strep_throat"]},
            "fatigue": {"is_symptom_of": ["influenza", "mononucleosis"]},
            "influenza": {
                "has_symptom": ["fever", "cough", "fatigue"],
                "has_treatment": ["antivirals", "rest"],
                "is_type_of": ["viral_infection"]
            },
            "common_cold": {
                "has_symptom": ["fever", "cough", "sore_throat"],
                "has_treatment": ["symptomatic_relief", "rest"],
                "is_type_of": ["viral_infection"]
            },
            "bacterial_infection": {
                "has_symptom": ["fever", "body_ache"], # Adding a new symptom
                "has_treatment": ["antibiotics"]
            },
            "strep_throat": {
                "has_symptom": ["sore_throat", "fever", "swollen_lymph_nodes"],
                "has_treatment": ["antibiotics"]
            },
            "mononucleosis": {
                "has_symptom": ["fatigue", "sore_throat", "swollen_lymph_nodes"],
                "has_treatment": ["rest"]
            },
            "antibiotics": {"treats": ["bacterial_infection", "strep_throat"]},
            "antivirals": {"treats": ["influenza"]},
            "rest": {"treats": ["influenza", "common_cold", "mononucleosis"]},
            "symptomatic_relief": {"treats": ["common_cold"]},
            "viral_infection": {"causes": ["influenza", "common_cold", "mononucleosis"]},
            "swollen_lymph_nodes": {"is_symptom_of": ["strep_throat", "mononucleosis"]}
        }

    def get_neighbors(self, entity: str) -> Dict[str, List[str]]:
        """Returns immediate neighbors and their relations for a given entity."""
        return self.graph.get(entity, {})

    def get_all_relations(self) -> List[str]:
        """Returns all unique relation types in the KG."""
        relations = set()
        for entity_data in self.graph.values():
            for relation in entity_data.keys():
                relations.add(relation)
        return list(relations)

    def get_entities_connected_by_relation(self, entity: str, relation: str) -> List[str]:
        """Returns entities connected to a given entity by a specific relation."""
        return self.graph.get(entity, {}).get(relation, [])

# --- 2. LLM Agent Simulation ---
class LLMAgent:
    def __init__(self, model_name="simulated-llm"):
        self.model_name = model_name
        # In a real scenario, this would initialize an actual LLM client (e.g., OpenAI, HuggingFace)

    def _simulate_llm_response(self, prompt: str) -> str:
        """
        Simulates an LLM's response based on the prompt type.
        This is a placeholder for actual LLM API calls.
        """
        print(f"\n--- LLM Input Prompt (Simulated) ---\n{prompt}\n----------------------------------")

        if "Relation Prune Prompt" in prompt:
            # Simulate selecting relevant relations based on common sense
            if "fever" in prompt and "cough" in prompt:
                return json.dumps({"selected_relations": ["is_symptom_of", "has_symptom"]})
            if "sore_throat" in prompt:
                return json.dumps({"selected_relations": ["is_symptom_of"]})
            return json.dumps({"selected_relations": ["is_symptom_of"]})
        elif "Entity Prune Prompt" in prompt:
            # Simulate selecting relevant entities
            if "influenza" in prompt and "common_cold" in prompt:
                return json.dumps({"selected_entities": ["influenza", "common_cold"]})
            if "strep_throat" in prompt and "mononucleosis" in prompt:
                 return json.dumps({"selected_entities": ["strep_throat"]})
            return json.dumps({"selected_entities": ["influenza"]})
        elif "Reasoning Prompt" in prompt:
            # Simulate reasoning sufficiency
            if "Diagnosis for Patient" in prompt and (("influenza" in prompt or "common_cold" in prompt) and "treatment" in prompt) or ("strep_throat" in prompt and "antibiotics" in prompt):
                return json.dumps({"sufficiency": "sufficient", "reasoning_confidence": 0.8})
            return json.dumps({"sufficiency": "insufficient", "reasoning_confidence": 0.3})
        elif "Generate Prompt" in prompt:
            # Simulate generating the final diagnosis
            if "influenza" in prompt and "fever" in prompt:
                return json.dumps({"diagnosis": "Influenza", "treatment_recommendation": "Antivirals and rest."})
            elif "common_cold" in prompt:
                return json.dumps({"diagnosis": "Common Cold", "treatment_recommendation": "Symptomatic relief and rest."})
            elif "strep_throat" in prompt:
                return json.dumps({"diagnosis": "Strep Throat", "treatment_recommendation": "Antibiotics."})
            return json.dumps({"diagnosis": "Undetermined", "treatment_recommendation": "Further investigation needed."})
        else:
            return json.dumps({"error": "Unknown prompt type."})

    def send_prompt(self, prompt_template: str, **kwargs) -> Dict[str, Any]:
        """
        Sends a templated prompt to the LLM and parses its JSON response.
        """
        filled_prompt = prompt_template.format(**kwargs)
        llm_output_str = self._simulate_llm_response(filled_prompt)
        try:
            return json.loads(llm_output_str)
        except json.JSONDecodeError:
            print(f"Error parsing LLM response: {llm_output_str}")
            return {"error": "Invalid JSON response from LLM."}

    # --- Prompt Definitions ---

    def relation_prune_prompt(self, patient_symptoms: List[str], current_entities: List[str], candidate_relations: List[str], reasoning_path_summary: str) -> Dict[str, Any]:
        """
        Guides the LLM to identify and score relevant relations from a candidate set.
        """
        prompt_template = """
        --- Relation Prune Prompt ---
        Patient Symptoms: {patient_symptoms}
        Currently considered entities: {current_entities}
        Summary of current reasoning paths: {reasoning_path_summary}

        Candidate relations to consider for further exploration: {candidate_relations}

        Based on the patient's symptoms and the current entities under consideration,
        identify and score the most relevant relations from the candidate list that
        are likely to lead to a correct diagnosis. Provide a list of selected relations.

        Expected JSON output format: {{"selected_relations": ["relation_1", "relation_2"]}}
        """
        return self.send_prompt(
            prompt_template,
            patient_symptoms=patient_symptoms,
            current_entities=current_entities,
            candidate_relations=candidate_relations,
            reasoning_path_summary=reasoning_path_summary
        )

    def entity_prune_prompt(self, patient_symptoms: List[str], current_entities: List[str], candidate_entities: List[str], reasoning_path_summary: str) -> Dict[str, Any]:
        """
        Directs the LLM to score the contribution of candidate entities.
        """
        prompt_template = """
        --- Entity Prune Prompt ---
        Patient Symptoms: {patient_symptoms}
        Currently considered entities: {current_entities}
        Summary of current reasoning paths: {reasoning_path_summary}

        Candidate entities found via graph exploration: {candidate_entities}

        Based on the patient's symptoms and the current reasoning context,
        score the contribution of the candidate entities to a potential diagnosis.
        Select the most promising entities for further investigation.

        Expected JSON output format: {{"selected_entities": ["entity_1", "entity_2"]}}
        """
        return self.send_prompt(
            prompt_template,
            patient_symptoms=patient_symptoms,
            current_entities=current_entities,
            candidate_entities=candidate_entities,
            reasoning_path_summary=reasoning_path_summary
        )

    def reasoning_prompt(self, patient_symptoms: List[str], explored_entities: List[str], reasoning_path_summary: str, question: str) -> Dict[str, Any]:
        """
        Asks the LLM to evaluate the sufficiency of the current reasoning paths for answering the question.
        """
        prompt_template = """
        --- Reasoning Prompt ---
        Patient Symptoms: {patient_symptoms}
        Explored entities in Knowledge Graph: {explored_entities}
        Summary of current reasoning paths: {reasoning_path_summary}
        Primary Question: {question}

        Evaluate whether the accumulated knowledge and reasoning paths are sufficient
        to confidently answer the primary question. If sufficient, indicate readiness
        for final answer generation.

        Expected JSON output format: {{"sufficiency": "sufficient" or "insufficient", "reasoning_confidence": 0.0-1.0}}
        """
        return self.send_prompt(
            prompt_template,
            patient_symptoms=patient_symptoms,
            explored_entities=explored_entities,
            reasoning_path_summary=reasoning_path_summary,
            question=question
        )

    def generate_prompt(self, patient_symptoms: List[str], relevant_entities: List[str], final_reasoning_paths: str, question: str) -> Dict[str, Any]:
        """
        Instructs the LLM to synthesize the final answer based on the accumulated knowledge and reasoning paths.
        """
        prompt_template = """
        --- Generate Prompt ---
        Patient Symptoms: {patient_symptoms}
        Relevant Entities identified: {relevant_entities}
        Final Reasoning Paths: {final_reasoning_paths}
        Primary Question: {question}

        Synthesize a comprehensive answer to the primary question based on the
        provided patient symptoms, relevant entities, and reasoning paths.
        Include a diagnosis and recommended treatment if applicable.

        Expected JSON output format: {{"diagnosis": "Diagnosis Name", "treatment_recommendation": "Treatment details."}}
        """
        return self.send_prompt(
            prompt_template,
            patient_symptoms=patient_symptoms,
            relevant_entities=relevant_entities,
            final_reasoning_paths=final_reasoning_paths,
            question=question
        )


# --- 3. Diagnostic Assistant Orchestration ---
class MedicalDiagnosticAssistant:
    def __init__(self, kg: MedicalKnowledgeGraph, llm_agent: LLMAgent):
        self.kg = kg
        self.llm_agent = llm_agent
        self.max_iterations = 5

    def diagnose_patient(self, patient_case: Dict[str, Any]) -> Dict[str, Any]:
        patient_symptoms = patient_case.get("symptoms", [])
        patient_history = patient_case.get("history", "No significant history.") # Not directly used in current flow but good for context
        primary_question = patient_case.get("question", "What is the most likely diagnosis and recommended treatment?")

        print(f"\n--- Starting Diagnosis for Patient: {patient_case.get('id', 'N/A')} ---")
        print(f"Initial Symptoms: {patient_symptoms}")

        current_explored_entities: set = set(patient_symptoms)
        reasoning_paths: List[str] = [f"Initial symptoms: {', '.join(patient_symptoms)}"]
        final_diagnosis_data: Dict[str, Any] = {"diagnosis": "Undetermined", "treatment_recommendation": "Pending."}

        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1}/{self.max_iterations} ---")
            current_reasoning_summary = "\n".join(reasoning_paths)

            # Step 1: Relation Prune Prompt
            print("Step 1: Relation Pruning...")
            all_kg_relations = self.kg.get_all_relations()
            relation_prune_response = self.llm_agent.relation_prune_prompt(
                patient_symptoms=patient_symptoms,
                current_entities=list(current_explored_entities),
                candidate_relations=all_kg_relations,
                reasoning_path_summary=current_reasoning_summary
            )
            selected_relations = relation_prune_response.get("selected_relations", [])
            print(f"LLM selected relations: {selected_relations}")
            if not selected_relations:
                print("No relations selected by LLM. Ending exploration early.")
                break
            reasoning_paths.append(f"LLM selected relations for exploration: {', '.join(selected_relations)}")

            # Step 2: Entity Prune Prompt (Explore and then Prune)
            print("Step 2: Entity Pruning...")
            candidate_entities_for_pruning: set = set()
            for entity in list(current_explored_entities):
                neighbors_data = self.kg.get_neighbors(entity)
                for rel in selected_relations:
                    if rel in neighbors_data:
                        candidate_entities_for_pruning.update(neighbors_data[rel])

            if not candidate_entities_for_pruning:
                print("No new candidate entities found. Ending exploration.")
                # If no new entities found, but LLM might still consider the existing path sufficient
                # We will rely on Reasoning Prompt to decide sufficiency

            entity_prune_response = self.llm_agent.entity_prune_prompt(
                patient_symptoms=patient_symptoms,
                current_entities=list(current_explored_entities),
                candidate_entities=list(candidate_entities_for_pruning),
                reasoning_path_summary=current_reasoning_summary
            )
            selected_entities = entity_prune_response.get("selected_entities", [])
            print(f"LLM selected entities: {selected_entities}")
            if not selected_entities and not candidate_entities_for_pruning:
                print("No entities selected by LLM and no candidates. Ending exploration early.")
                break
            elif not selected_entities and candidate_entities_for_pruning:
                 print("LLM did not select any from candidates, relying on existing explored.")

            # Add newly selected entities to explored set
            # Convert to sets for efficient difference and union operations
            current_explored_set = set(current_explored_entities)
            selected_set = set(selected_entities)
            newly_explored = selected_set - current_explored_set

            if newly_explored:
                current_explored_entities.update(newly_explored)
                reasoning_paths.append(f"New entities explored: {', '.join(sorted(list(newly_explored)))}")
            else:
                print("No entirely new entities added to exploration in this iteration.")

            # Step 3: Reasoning Prompt
            print("Step 3: Evaluating Reasoning Sufficiency...")
            reasoning_response = self.llm_agent.reasoning_prompt(
                patient_symptoms=patient_symptoms,
                explored_entities=list(current_explored_entities),
                reasoning_path_summary="\n".join(reasoning_paths),
                question=primary_question
            )
            sufficiency = reasoning_response.get("sufficiency")
            confidence = reasoning_response.get("reasoning_confidence")
            print(f"LLM assessed sufficiency: {sufficiency} (Confidence: {confidence})")

            if sufficiency == "sufficient" and confidence and confidence > 0.7: # Threshold for confidence
                print("LLM indicates sufficient information for diagnosis.")
                # Proceed to Generate Prompt
                break
            elif i == self.max_iterations - 1:
                print("Max iterations reached without sufficient reasoning.")
                # Fallback to generate with what's available
            else:
                print("LLM indicates insufficient information. Continuing exploration.")

        # Step 4: Generate Prompt (after loop or when sufficient)
        print("\n--- Generating Final Diagnosis ---")
        final_reasoning_summary = "\n".join(reasoning_paths)
        generate_response = self.llm_agent.generate_prompt(
            patient_symptoms=patient_symptoms,
            relevant_entities=list(current_explored_entities),
            final_reasoning_paths=final_reasoning_summary,
            question=primary_question
        )
        final_diagnosis_data = {
            "diagnosis": generate_response.get("diagnosis", "Undetermined"),
            "treatment_recommendation": generate_response.get("treatment_recommendation", "Consult a specialist.")
        }
        print(f"Final Diagnosis: {final_diagnosis_data['diagnosis']}")
        print(f"Treatment Recommendation: {final_diagnosis_data['treatment_recommendation']}")

        return final_diagnosis_data

# Example Usage:
if __name__ == "__main__":
    kg = MedicalKnowledgeGraph()
    llm_agent = LLMAgent()
    assistant = MedicalDiagnosticAssistant(kg, llm_agent)

    patient_case_1 = {
        "id": "P001",
        "symptoms": ["fever", "cough", "fatigue"],
        "history": "No known allergies.",
        "question": "What is the most likely diagnosis and treatment?"
    }

    patient_case_2 = {
        "id": "P002",
        "symptoms": ["sore_throat", "fever"],
        "history": "Recently visited a daycare.",
        "question": "Given the symptoms, what could be the condition?"
    }

    patient_case_3 = {
        "id": "P003",
        "symptoms": ["fatigue", "sore_throat"],
        "history": "Has been feeling unusually tired for weeks.",
        "question": "Identify the potential viral infection and treatment."
    }

    diagnosis_1 = assistant.diagnose_patient(patient_case_1)
    print(f"\n--- Final Result for P001: ---\nDiagnosis: {diagnosis_1['diagnosis']}\nTreatment: {diagnosis_1['treatment_recommendation']}")

    diagnosis_2 = assistant.diagnose_patient(patient_case_2)
    print(f"\n--- Final Result for P002: ---\nDiagnosis: {diagnosis_2['diagnosis']}\nTreatment: {diagnosis_2['treatment_recommendation']}")

    diagnosis_3 = assistant.diagnose_patient(patient_case_3)
    print(f"\n--- Final Result for P003: ---\nDiagnosis: {diagnosis_3['diagnosis']}\nTreatment: {diagnosis_3['treatment_recommendation']}")
