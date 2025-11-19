class KnowledgeGraph:
    def __init__(self):
        self.graph = {}  # {entity: {relation: [connected_entities]}}
        self.facts = []  # List of (entity1, relation, entity2) triples

    def add_triple(self, entity1, relation, entity2):
        self.facts.append((entity1, relation, entity2))

        if entity1 not in self.graph:
            self.graph[entity1] = {}
        if relation not in self.graph[entity1]:
            self.graph[entity1][relation] = []
        self.graph[entity1][relation].append(entity2)

        inverse_relation = f"has_inverse_{relation}"
        if entity2 not in self.graph:
            self.graph[entity2] = {}
        if inverse_relation not in self.graph[entity2]:
            self.graph[entity2][inverse_relation] = []
        self.graph[entity2][inverse_relation].append(entity1)

    def get_neighbors(self, entity, relation=None):
        if entity not in self.graph:
            return []
        if relation:
            return self.graph[entity].get(relation, [])
        all_neighbors = []
        for rel in self.graph[entity]:
            all_neighbors.extend(self.graph[entity][rel])
        return list(set(all_neighbors))

    def get_facts_about_entity(self, entity):
        relevant_facts = []
        for fact in self.facts:
            if entity == fact[0] or entity == fact[2]:
                relevant_facts.append(fact)
        return relevant_facts

def create_medical_kg():
    kg = KnowledgeGraph()
    kg.add_triple("Patient_A", "has_symptom", "Fever")
    kg.add_triple("Patient_A", "has_symptom", "Cough")
    kg.add_triple("Patient_A", "has_lab_result", "High_CRP")
    kg.add_triple("Fever", "is_symptom_of", "Flu")
    kg.add_triple("Fever", "is_symptom_of", "Pneumonia")
    kg.add_triple("Cough", "is_symptom_of", "Flu")
    kg.add_triple("Cough", "is_symptom_of", "Pneumonia")
    kg.add_triple("High_CRP", "indicates", "Inflammation")
    kg.add_triple("Inflammation", "is_associated_with", "Bacterial_Infection")
    kg.add_triple("Inflammation", "is_associated_with", "Viral_Infection")
    kg.add_triple("Flu", "is_a_type_of", "Viral_Infection")
    kg.add_triple("Pneumonia", "can_be_caused_by", "Bacterial_Infection")
    kg.add_triple("Pneumonia", "can_be_caused_by", "Viral_Infection")
    kg.add_triple("Flu", "has_treatment", "Antivirals")
    kg.add_triple("Pneumonia", "has_treatment", "Antibiotics")
    kg.add_triple("Pneumonia", "has_treatment", "Antivirals")
    kg.add_triple("Headache", "is_symptom_of", "Flu")
    kg.add_triple("Headache", "is_symptom_of", "Migraine")
    kg.add_triple("Migraine", "has_treatment", "Triptans")
    return kg

class LLMSimulator:
    def __init__(self):
        pass

    def generate_response(self, prompt, max_tokens=100):
        if "next reasoning step" in prompt.lower() and "symptoms" in prompt.lower():
            return "Based on the symptoms, consider common diseases. Explore 'is_symptom_of' relations from the given symptoms in the KG."
        elif "next reasoning step" in prompt.lower() and "diseases" in prompt.lower():
            return "Given potential diseases, retrieve their causes, associated conditions, and treatments from the KG. Look for 'can_be_caused_by', 'is_associated_with', 'has_treatment' relations."
        elif "diagnostic hypothesis" in prompt.lower():
            if "Flu" in prompt and "Fever" in prompt and "Cough" in prompt:
                return "Diagnostic Hypothesis: Flu (Viral Infection). Explanation: Patient exhibits Fever and Cough, which are common symptoms of Flu. Flu is a Viral Infection. Treatment Recommendation: Antivirals, rest, hydration."
            elif "Pneumonia" in prompt and "Cough" in prompt and "High_CRP" in prompt:
                return "Diagnostic Hypothesis: Pneumonia (Bacterial/Viral Infection). Explanation: Patient presents with Cough and High CRP, indicating inflammation, which is associated with Pneumonia. Further tests needed to distinguish bacterial vs. viral. Treatment Recommendation: Antibiotics (if bacterial), antivirals (if viral), supportive care."
            elif "Migraine" in prompt and "Headache" in prompt:
                return "Diagnostic Hypothesis: Migraine. Explanation: Patient reported Headache. Migraine is a common cause of headaches. Treatment Recommendation: Triptans, pain relievers, rest in a dark room."
            else:
                return "Diagnostic Hypothesis: Requires more information or further investigation. Explanation: The available information suggests several possibilities or is insufficient for a definitive diagnosis. Treatment Recommendation: Symptomatic relief and further diagnostic tests."
        elif "entities and relations" in prompt.lower():
            if "fever and cough" in prompt.lower():
                return '{"entities": ["Fever", "Cough"], "relations": ["is_symptom_of"]}'
            elif "patient a medical history" in prompt.lower():
                return '{"entities": ["Patient_A"], "relations": ["has_symptom", "has_lab_result"]}'
            elif "headache" in prompt.lower():
                return '{"entities": ["Headache"], "relations": ["is_symptom_of"]}'
            else:
                return '{"entities": [], "relations": []}'
        elif "relevant paths" in prompt.lower() and "fever" in prompt.lower() and "flu" in prompt.lower():
            return '["(Fever, is_symptom_of, Flu)"]'
        elif "relevant paths" in prompt.lower() and "cough" in prompt.lower() and "pneumonia" in prompt.lower():
            return '["(Cough, is_symptom_of, Pneumonia)"]'
        elif "treatment recommendations" in prompt.lower() and "flu" in prompt.lower():
            return "Antivirals, rest, hydration."
        elif "treatment recommendations" in prompt.lower() and "pneumonia" in prompt.lower():
            return "Antibiotics (if bacterial), antivirals (if viral), supportive care."
        elif "most relevant next entities and relations to explore" in prompt.lower():
            if "fever" in prompt.lower() or "cough" in prompt.lower():
                return '["(is_symptom_of, Flu)", "(is_symptom_of, Pneumonia)", "(indicates, Inflammation)"]'
            elif "inflammation" in prompt.lower():
                return '["(is_associated_with, Bacterial_Infection)", "(is_associated_with, Viral_Infection)"]'
            elif "flu" in prompt.lower() or "pneumonia" in prompt.lower():
                return '["(has_treatment, Antivirals)", "(has_treatment, Antibiotics)", "(is_a_type_of, Viral_Infection)", "(can_be_caused_by, Bacterial_Infection)"]'
            elif "headache" in prompt.lower():
                return '["(is_symptom_of, Flu)", "(is_symptom_of, Migraine)"]'
            else:
                return '[]'
        else:
            return "Simulated LLM response: I'm processing your request with the knowledge graph. Please provide more context or ask a specific question."

    def parse_json_response(self, text):
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
            parts = text.split(",")
            result = {}
            for part in parts:
                if ":" in part:
                    key, value = part.split(":", 1)
                    key = key.strip().strip('"\'')
                    value = value.strip()
                    if value.startswith("[") and value.endswith("]"):
                        items = value[1:-1].split(",")
                        result[key] = [item.strip().strip('"\'') for item in items if item.strip()]
                    else:
                        result[key] = value.strip().strip('"\'')
            return result
        elif text.startswith("[") and text.endswith("]"):
            items = text[1:-1].split("),")
            parsed_list = []
            for item in items:
                item = item.strip()
                if item.startswith("(") and item.endswith(")"):
                    triple_str = item[1:-1]
                    parts = triple_str.split(",", 2)
                    if len(parts) == 3:
                        parsed_list.append(tuple(p.strip().strip("'\"") for p in parts))
                elif item.startswith("(") and not item.endswith(")"): # Handle cases where last ) is missing due to split
                    item += ")" # add back for consistency for parsing
                    triple_str = item[1:-1]
                    parts = triple_str.split(",", 2)
                    if len(parts) == 3:
                        parsed_list.append(tuple(p.strip().strip("'\"") for p in parts))
                elif item.startswith("(") and item.endswith(")"): # Also handle simpler (rel, entity) tuples
                    parts = item[1:-1].split(",", 1)
                    if len(parts) == 2:
                        parsed_list.append(tuple(p.strip().strip("'\"") for p in parts))
            return parsed_list
        return {}

class SemanticParser:
    def __init__(self):
        self.keywords = {
            "symptom": ["symptom", "symptoms", "suffering from", "experiencing"],
            "lab_result": ["lab result", "test result", "blood work", "CRP"],
            "diagnosis": ["diagnose", "diagnosis", "what is it"],
            "cause": ["cause", "caused by", "etiology"],
            "treatment": ["treatment", "cure", "medication", "therapy"],
            "patient": ["patient"],
            "history": ["history", "medical history"]
        }
        self.medical_entities = ["Fever", "Cough", "High_CRP", "Inflammation", "Flu", "Pneumonia", "Headache", "Migraine", "Antivirals", "Antibiotics", "Triptans", "Patient_A"]

    def parse_natural_language(self, query):
        query_lower = query.lower()
        extracted_entities = []
        extracted_query_type = "unknown"

        for entity in self.medical_entities:
            if entity.lower() in query_lower:
                extracted_entities.append(entity)

        if any(kw in query_lower for kw in self.keywords["diagnosis"]):
            extracted_query_type = "diagnosis"
        elif any(kw in query_lower for kw in self.keywords["symptom"]):
            extracted_query_type = "symptom_query"
        elif any(kw in query_lower for kw in self.keywords["lab_result"]):
            extracted_query_type = "lab_result_query"
        elif any(kw in query_lower for kw in self.keywords["treatment"]):
            extracted_query_type = "treatment_query"
        elif any(kw in query_lower for kw in self.keywords["cause"]):
            extracted_query_type = "cause_query"
        elif any(kw in query_lower for kw in self.keywords["patient"]) or any(kw in query_lower for kw in self.keywords["history"]):
            extracted_query_type = "patient_profile"

        return {
            "query_type": extracted_query_type,
            "entities": list(set(extracted_entities)),
            "original_query": query
        }

class KGReasoningAgent:
    def __init__(self, kg, llm_simulator):
        self.kg = kg
        self.llm_simulator = llm_simulator

    def _format_triple_path_for_prompt(self, path):
        if not path:
            return "No path found."
        formatted_path = []
        for e1, r, e2 in path:
            formatted_path.append(f"({e1} --{r}--> {e2})")
        return " -> ".join(formatted_path)

    def _extract_topic_entities_from_llm(self, text):
        prompt = f"Given the text: \"{text}\", identify key medical entities and relevant relations for a knowledge graph query. Respond in JSON format like: {{\"entities\": [\"Entity1\", \"Entity2\"], \"relations\": [\"relation1\"]}}."
        llm_response_str = self.llm_simulator.generate_response(prompt)
        llm_response = self.llm_simulator.parse_json_response(llm_response_str)
        return llm_response.get("entities", []), llm_response.get("relations", [])

    def explore_kg_with_llm_guidance(self, initial_entities, max_steps=3, beam_width=2):
        if not initial_entities:
            return []

        beams = []
        for entity in initial_entities:
            beams.append((entity, [], 0))

        final_paths = []
        all_visited_entities = set(initial_entities)

        for step in range(max_steps):
            new_beams = []
            if not beams:
                break

            for current_entity, current_path_triples, current_score in beams:
                current_path_str = self._format_triple_path_for_prompt(current_path_triples)
                prompt = (f"Considering the current path in the knowledge graph: {current_path_str} "
                          f"and the current focus entity: {current_entity}, what are the most relevant "
                          f"next entities and relations to explore to find diagnostic information? "
                          f"Provide a list of suggested (relation, next_entity) pairs related to '{current_entity}'. "
                          f"Example: ['(is_symptom_of, Disease)', '(indicates, Condition)']")
                
                llm_guidance_str = self.llm_simulator.generate_response(prompt)
                parsed_guidance = self.llm_simulator.parse_json_response(llm_guidance_str)
                
                next_possibilities = []

                if isinstance(parsed_guidance, list):
                    for suggested_rel, suggested_next_entity in parsed_guidance:
                        if suggested_next_entity not in all_visited_entities:
                            neighbors = self.kg.get_neighbors(current_entity, relation=suggested_rel)
                            if suggested_next_entity in neighbors:
                                next_possibilities.append((suggested_rel, suggested_next_entity))
                
                if not next_possibilities and current_entity in self.kg.graph:
                    for rel, next_entities in self.kg.graph[current_entity].items():
                        for next_e in next_entities:
                            if next_e not in all_visited_entities:
                                next_possibilities.append((rel, next_e))

                for relation, next_entity in next_possibilities:
                    new_path_triples = current_path_triples + [(current_entity, relation, next_entity)]
                    new_score = current_score + 1
                    new_beams.append((next_entity, new_path_triples, new_score))
                    all_visited_entities.add(next_entity)

            beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]
            
            for _, path_triples, _ in beams:
                final_paths.append(path_triples)

        unique_paths = []
        seen_path_tuples = set()
        for path in final_paths:
            path_as_tuple = tuple(tuple(t) for t in path)
            if path_as_tuple not in seen_path_tuples:
                unique_paths.append(path)
                seen_path_tuples.add(path_as_tuple)
        
        return unique_paths

    def retrieve_and_reason_with_rag_kdcot(self, patient_input, query_entities, explored_paths):
        initial_facts = []
        for entity in query_entities:
            initial_facts.extend(self.kg.get_facts_about_entity(entity))

        formatted_initial_facts = "\n".join([f"  - ({e1} --{r}--> {e2})" for e1, r, e2 in initial_facts])
        if not formatted_initial_facts:
            formatted_initial_facts = "No direct facts found for initial entities."

        initial_rag_prompt = (
            f"Patient Input: {patient_input}\n\n"
            f"Relevant Knowledge Graph Facts (initial retrieval):\n{formatted_initial_facts}\n\n"
            f"Considering this information, what is the first step in diagnosing the patient? "
            f"Think step-by-step and identify key symptoms and potential diseases to investigate."
        )

        llm_reasoning_step_1 = self.llm_simulator.generate_response(initial_rag_prompt)
        chain_of_thought_steps = [f"Initial thought: {llm_reasoning_step_1}"]
        current_context_facts = list(initial_facts)
        
        if explored_paths:
            chain_of_thought_steps.append("\nExplored Knowledge Paths:")
            for path in explored_paths:
                path_str = self._format_triple_path_for_prompt(path)
                chain_of_thought_steps.append(f"  - {path_str}")
                current_context_facts.extend(path)

        for i in range(2):
            current_reasoning_state = "\n".join(chain_of_thought_steps)
            
            next_reasoning_prompt = (
                f"Current Reasoning State:\n{current_reasoning_state}\n\n"
                f"Considering the patient input and the retrieved KG facts, what is the next logical step in your diagnostic reasoning process? "
                f"Refine your understanding and propose a sub-goal or further investigation."
            )
            llm_next_step = self.llm_simulator.generate_response(next_reasoning_prompt)
            chain_of_thought_steps.append(f"Step {i+2}: {llm_next_step}")

        final_context_facts_str = "\n".join([f"  - ({e1} --{r}--> {e2})" for e1, r, e2 in current_context_facts])
        final_diagnosis_prompt = (
            f"Patient Input: {patient_input}\n\n"
            f"Comprehensive Knowledge Graph Facts:\n{final_context_facts_str}\n\n"
            f"Chain of Thought:\n" + "\n".join(chain_of_thought_steps) + "\n\n"
            f"Based on all the information, provide a primary diagnostic hypothesis and a detailed, faithful explanation. "
            f"Also, suggest initial treatment recommendations."
        )

        final_llm_response = self.llm_simulator.generate_response(final_diagnosis_prompt)
        return final_llm_response, "\n".join(chain_of_thought_steps)


class ClinicalDiagnosticAssistant:
    def __init__(self):
        self.kg = create_medical_kg()
        self.llm_simulator = LLMSimulator()
        self.semantic_parser = SemanticParser()
        self.kg_reasoning_agent = KGReasoningAgent(self.kg, self.llm_simulator)

    def diagnose_patient(self, patient_description):
        print(f"\n--- Diagnosing Patient: {patient_description} ---")

        parsed_query = self.semantic_parser.parse_natural_language(patient_description)
        initial_entities = parsed_query["entities"]
        print(f"Parsed initial entities: {initial_entities}")

        if not initial_entities:
            print("No significant medical entities extracted. Cannot proceed with KG reasoning.")
            return "Diagnosis: Undetermined. No specific medical entities found in your input.", "No reasoning path."

        print("\n--- KG Exploration with LLM Guidance (Simulated Beam Search) ---")
        explored_paths = self.kg_reasoning_agent.explore_kg_with_llm_guidance(initial_entities, max_steps=3, beam_width=2)
        print(f"Explored paths ({len(explored_paths)} found):")
        for i, path in enumerate(explored_paths):
            print(f"  Path {i+1}: {self.kg_reasoning_agent._format_triple_path_for_prompt(path)}")
        
        if not explored_paths:
            print("No relevant paths found in KG exploration.")

        print("\n--- RAG and KDCoT Reasoning with LLM (Simulated) ---")
        final_diagnosis_output, chain_of_thought = self.kg_reasoning_agent.retrieve_and_reason_with_rag_kdcot(
            patient_description,
            initial_entities,
            explored_paths
        )
        print("\n--- Final Diagnostic Output ---")
        print(final_diagnosis_output)
        print("\n--- Chain of Thought ---")
        print(chain_of_thought)

        return final_diagnosis_output, chain_of_thought

if __name__ == "__main__":
    assistant = ClinicalDiagnosticAssistant()

    patient_input_1 = "Patient A presents with a high fever, persistent cough, and general fatigue. Lab results show elevated CRP."
    diagnosis_1, cot_1 = assistant.diagnose_patient(patient_input_1)
    print("\n-------------------------------------------------\n")

    patient_input_2 = "A 35-year-old patient complains of severe headache and sensitivity to light. No fever or cough."
    diagnosis_2, cot_2 = assistant.diagnose_patient(patient_input_2)
    print("\n-------------------------------------------------\n")

    patient_input_3 = "Patient B feels unwell with some body aches but no specific fever or cough."
    diagnosis_3, cot_3 = assistant.diagnose_patient(patient_input_3)
    print("\n-------------------------------------------------\n")