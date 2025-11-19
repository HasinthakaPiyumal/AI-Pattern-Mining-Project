import networkx as nx
import gradio as gr
import json

class MediGraphKG:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity_id, entity_type, attributes=None):
        if not self.graph.has_node(entity_id):
            self.graph.add_node(entity_id, type=entity_type, **(attributes if attributes else {}))

    def add_relationship(self, source_id, target_id, rel_type, attributes=None):
        if self.graph.has_node(source_id) and self.graph.has_node(target_id):
            self.graph.add_edge(source_id, target_id, type=rel_type, **(attributes if attributes else {}))
        else:
            print(f"Warning: One or both entities '{source_id}', '{target_id}' not found. Relationship not added.")

    def query_facts(self, entity_id=None, rel_type=None, target_entity_id=None, depth=1):
        results = []
        if entity_id and self.graph.has_node(entity_id):
            # Find outgoing relationships
            for neighbor in self.graph.neighbors(entity_id):
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                if rel_type is None or edge_data.get("type") == rel_type:
                    if target_entity_id is None or neighbor == target_entity_id:
                        results.append((entity_id, edge_data.get("type", "UNKNOWN_REL"), neighbor))
            # Find incoming relationships
            for source, target in self.graph.in_edges(entity_id):
                edge_data = self.graph.get_edge_data(source, target)
                if rel_type is None or edge_data.get("type") == rel_type:
                     if target_entity_id is None or source == target_entity_id:
                        results.append((source, edge_data.get("type", "UNKNOWN_REL"), target))

            if depth > 1:
                for _, _, next_entity in list(results):
                    if next_entity != entity_id:
                        sub_results = self.query_facts(next_entity, depth=depth-1)
                        results.extend(sub_results)

        return list(set(results))

    def get_node_attributes(self, entity_id):
        return self.graph.nodes.get(entity_id, {})

    def get_all_entities(self, entity_type=None):
        if entity_type:
            return [node for node, data in self.graph.nodes(data=True) if data.get("type") == entity_type]
        return list(self.graph.nodes())


class LLMService:
    def __init__(self, model_name="MockLLM"):
        self.model_name = model_name

    def generate(self, prompt, max_tokens=200):
        if "diagnose" in prompt.lower() and "fever" in prompt.lower() and "cough" in prompt.lower():
            return "Based on the provided symptoms (fever, cough), a common diagnosis could be Influenza. Further investigation needed. Supporting facts from KG: (Influenza, HAS_SYMPTOM, Fever), (Influenza, HAS_SYMPTOM, Cough)."
        elif "explain" in prompt.lower() and "influenza" in prompt.lower():
            return "Influenza, commonly known as the flu, is a contagious respiratory illness caused by influenza viruses. It can cause mild to severe illness, and at times can lead to death. Key symptoms include fever, cough, sore throat, and muscle aches. Supporting facts from KG: (Influenza, IS_A, Respiratory_Illness), (Influenza, CAUSED_BY, Influenza_Virus)."
        elif "treatment for influenza" in prompt.lower():
            return "Common treatments for Influenza include antiviral medications (e.g., Oseltamivir) and supportive care (rest, fluids). Supporting facts from KG: (Influenza, HAS_TREATMENT, Oseltamivir), (Influenza, HAS_TREATMENT, Supportive_Care)."
        elif "semantic query" in prompt.lower():
            return "MATCH (p:Patient)-[:HAS_SYMPTOM]->(s:Symptom {name: 'Fever'}) RETURN p, s"
        elif "next step for diagnosis" in prompt.lower() or "most critical next piece of information" in prompt.lower():
            if "fever" in prompt.lower() and "cough" in prompt.lower() and "influenza" not in prompt.lower():
                return "Find symptoms of Influenza and Common Cold."
            elif "influenza" in prompt.lower() and "treatments for" not in prompt.lower():
                return "Find treatments for Influenza."
            return "Considering the current paths, focusing on differential diagnosis between viral and bacterial infections would be the most promising next step. Explore paths related to 'Bacterial Pneumonia' and 'Common Cold'."
        elif "Evaluate these facts for their relevance" in prompt:
            if "Fever" in prompt and "Cough" in prompt and "Influenza" not in prompt:
                return json.dumps({
                    "next_steps": [
                        {"entity": "Influenza", "reason": "Common disease for fever and cough.", "score": 0.9},
                        {"entity": "Common_Cold", "reason": "Another common respiratory illness.", "score": 0.7},
                        {"entity": "Pneumonia", "reason": "More severe respiratory illness possibility.", "score": 0.6}
                    ]
                })
            elif "Influenza" in prompt and "Oseltamivir" not in prompt:
                 return json.dumps({
                    "next_steps": [
                        {"entity": "Oseltamivir", "reason": "Antiviral treatment for Influenza.", "score": 0.95},
                        {"entity": "Supportive_Care", "reason": "General care for Influenza.", "score": 0.8}
                    ]
                })
            return json.dumps({"next_steps": []})
        else:
            return f"LLM response to prompt: '{prompt[:100]}...' (Mock response for {self.model_name})"

def extract_topic_entities(llm_service, text):
    entities = []
    if "fever" in text.lower(): entities.append("Fever")
    if "cough" in text.lower(): entities.append("Cough")
    if "fatigue" in text.lower(): entities.append("Fatigue")
    if "sore throat" in text.lower(): entities.append("Sore_Throat")
    if "influenza" in text.lower(): entities.append("Influenza")
    if "diabetes" in text.lower(): entities.append("Diabetes_Type_2")
    if "insulin" in text.lower(): entities.append("Insulin")
    return list(set(entities))

def semantic_parse_query(llm_service, natural_language_query, kg_schema=None):
    if "diagnose" in natural_language_query.lower() and "fever" in natural_language_query.lower():
        return {"query_type": "DIAGNOSIS", "symptoms": ["Fever", "Cough"]}
    elif "explain influenza" in natural_language_query.lower():
        return {"query_type": "EXPLAIN", "entity": "Influenza"}
    elif "treatment for" in natural_language_query.lower():
        entity = natural_language_query.lower().split("treatment for")[-1].strip().replace("?", "").replace(" ", "_").title()
        return {"query_type": "TREATMENT", "entity": entity}
    return {"query_type": "GENERIC", "text": natural_language_query}

def knowledge_driven_chain_of_thought_prompt(initial_prompt, retrieved_facts):
    fact_str = "\n".join([f"({s}, {p}, {o})" for s, p, o in retrieved_facts])
    if fact_str:
        return f"{initial_prompt}\n\nConsider these relevant facts from the medical knowledge graph:\n{fact_str}\n\nStep-by-step reasoning:"
    return initial_prompt


class ThinkonGraphAgent:
    def __init__(self, llm_service, kg_service, max_iterations=5, beam_width=3, pruning_threshold=0.2):
        self.llm = llm_service
        self.kg = kg_service
        self.max_iterations = max_iterations
        self.beam_width = beam_width
        self.pruning_threshold = pruning_threshold

    def format_triple_path(self, path):
        formatted_path = []
        for s, p, o in path:
            s_attrs = self.kg.get_node_attributes(s)
            o_attrs = self.kg.get_node_attributes(o)
            s_str = f"{s} ({s_attrs.get('type', 'Entity')})"
            o_str = f"{o} ({o_attrs.get('type', 'Entity')})"
            formatted_path.append(f"({s_str} -[{p}]-> {o_str})")
        return " -> ".join(formatted_path)

    def hybrid_pruning_strategy(self, paths_with_scores):
        if not paths_with_scores:
            return []
        
        sorted_paths = sorted(paths_with_scores, key=lambda x: x[2], reverse=True)
        
        top_paths = sorted_paths[:self.beam_width]
        
        pruned_paths = list(top_paths)
        for path_data in sorted_paths[self.beam_width:]:
            if path_data[2] >= self.pruning_threshold:
                pruned_paths.append(path_data)
            else:
                break
        
        return pruned_paths

    def llm_guided_beam_search(self, initial_entities, diagnostic_goal):
        current_beams = [(initial_entities, [], 1.0)]
        
        for iteration in range(self.max_iterations):
            new_beams = []
            if not current_beams:
                break

            for entities, path_so_far, current_score in current_beams:
                if not entities:
                    new_beams.append((entities, path_so_far, current_score))
                    continue

                for entity in entities:
                    facts = self.kg.query_facts(entity, depth=1)
                    
                    if not facts:
                        new_beams.append((entities, path_so_far, current_score))
                        continue

                    prompt = (
                        f"Current diagnostic goal: {diagnostic_goal}\n"
                        f"Current path: {self.format_triple_path(path_so_far)}\n"
                        f"Considering entity: {entity}\n"
                        f"Available facts for {entity}:\n"
                        + "\n".join([self.format_triple_path([f]) for f in facts]) +
                        "\nEvaluate these facts for their relevance to the diagnostic goal. Propose next promising entities/paths and assign a relevance score (0.0-1.0) for each. Format as JSON: {\"next_steps\": [{\"entity\": \"<entity_id>\", \"reason\": \"<brief_reason>\", \"score\": <float_score>}]}"
                    )
                    
                    llm_response_text = self.llm.generate(prompt)
                    
                    try:
                        llm_response = json.loads(llm_response_text)
                        next_steps = llm_response.get("next_steps", [])
                    except json.JSONDecodeError:
                        next_steps = []

                    for step in next_steps:
                        next_entity_id_from_llm = step.get("entity")
                        score_from_llm = step.get("score", 0.5)

                        valid_next_facts = []
                        for s, p, o in facts:
                            if o == next_entity_id_from_llm:
                                valid_next_facts.append((s,p,o))

                        if valid_next_facts:
                            for chosen_fact in valid_next_facts:
                                new_path = path_so_far + [chosen_fact]
                                combined_score = current_score * score_from_llm
                                new_beams.append(([next_entity_id_from_llm], new_path, combined_score))
                        elif self.kg.graph.has_node(next_entity_id_from_llm):
                            new_beams.append(([next_entity_id_from_llm], path_so_far, current_score * score_from_llm * 0.7))

            pruned_new_beams = self.hybrid_pruning_strategy(new_beams)
            current_beams = pruned_new_beams
            
            if any(diagnostic_goal.lower() in str(entities).lower() for entities, _, _ in current_beams):
                break
            
        return sorted(current_beams, key=lambda x: x[2], reverse=True)


    def iterative_prompting(self, initial_query, initial_kg_facts=None):
        conversation_history = []
        current_state_info = []

        initial_prompt = initial_query
        if initial_kg_facts:
            initial_prompt = knowledge_driven_chain_of_thought_prompt(initial_prompt, initial_kg_facts)
        
        llm_response = self.llm.generate(initial_prompt)
        conversation_history.append({"role": "user", "content": initial_query})
        conversation_history.append({"role": "assistant", "content": llm_response})
        current_state_info.append(f"LLM initial response: {llm_response}")

        for i in range(self.max_iterations - 1):
            agent_prompt = (
                "Based on the conversation and current understanding, what is the most critical next piece of information needed from the knowledge graph "
                "to refine the diagnosis or treatment plan? Focus on specific entities, relationships, or questions. "
                "Respond with a short, specific query for the KG or a strategic next step. Example: 'Find symptoms of <Disease X>' or 'Compare treatments for <Condition Y>'. "
                "If enough information, propose a final diagnosis/treatment. Current conversation:\n" +
                "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-3:]]) +
                "\nCurrent KG context:\n" + "\n".join(current_state_info[-2:])
            )
            
            agent_decision = self.llm.generate(agent_prompt, max_tokens=100)
            
            if "final diagnosis" in agent_decision.lower() or "final treatment" in agent_decision.lower():
                break

            kg_query_results = []
            if "symptoms of" in agent_decision.lower():
                entity_name = agent_decision.split("symptoms of")[-1].strip().replace("?", "").replace(".", "").replace(" ", "_").title()
                kg_query_results = self.kg.query_facts(entity_name, rel_type="HAS_SYMPTOM")
                if not kg_query_results:
                    kg_query_results = self.kg.query_facts(entity_name, rel_type="ASSOCIATED_WITH")
            elif "treatments for" in agent_decision.lower():
                entity_name = agent_decision.split("treatments for")[-1].strip().replace("?", "").replace(".", "").replace(" ", "_").title()
                kg_query_results = self.kg.query_facts(entity_name, rel_type="HAS_TREATMENT")
            elif "compare treatments" in agent_decision.lower():
                kg_query_results = [("Oseltamivir", "HAS_SIDE_EFFECT", "Nausea"), ("Amoxicillin", "HAS_SIDE_EFFECT", "Rash")]
            
            if kg_query_results:
                formatted_facts = [self.format_triple_path([f]) for f in kg_query_results]
                fact_message = f"KG retrieved: {'; '.join(formatted_facts)}"
                current_state_info.append(fact_message)
                
                llm_prompt_with_facts = knowledge_driven_chain_of_thought_prompt(
                    f"Given the new information from the knowledge graph: {fact_message}\nHow does this refine our understanding or diagnosis? {agent_decision}",
                    kg_query_results
                )
                llm_response = self.llm.generate(llm_prompt_with_facts)
                conversation_history.append({"role": "user", "content": agent_decision + " (from KG search)"})
                conversation_history.append({"role": "assistant", "content": llm_response})
            else:
                current_state_info.append(f"KG search for '{agent_decision}' yielded no new facts.")
                llm_response = self.llm.generate(f"Considering our current understanding, and '{agent_decision}' yielded no new KG facts, what is the next best step for diagnosis/treatment?")
                conversation_history.append({"role": "user", "content": agent_decision})
                conversation_history.append({"role": "assistant", "content": llm_response})

        final_summary_prompt = (
            "Summarize the diagnostic process and propose a final diagnosis or treatment plan based on the entire conversation history and retrieved KG facts. "
            "Explain your reasoning concisely. Full conversation:\n" +
            "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        )
        final_diagnosis = self.llm.generate(final_summary_prompt)
        return final_diagnosis, conversation_history


kg_instance = MediGraphKG()
llm_service = LLMService()
agent = ThinkonGraphAgent(llm_service=llm_service, kg_service=kg_instance)

def populate_mock_kg(kg):
    kg.add_entity("Patient_Alice", "Patient", {"age": 45, "gender": "Female"})
    kg.add_entity("Patient_Bob", "Patient", {"age": 60, "gender": "Male"})
    
    kg.add_entity("Fever", "Symptom")
    kg.add_entity("Cough", "Symptom")
    kg.add_entity("Sore_Throat", "Symptom")
    kg.add_entity("Fatigue", "Symptom")
    kg.add_entity("Muscle_Aches", "Symptom")
    kg.add_entity("Headache", "Symptom")
    kg.add_entity("Rash", "Symptom")
    
    kg.add_entity("Influenza", "Disease")
    kg.add_entity("Common_Cold", "Disease")
    kg.add_entity("Streptococcal_Pharyngitis", "Disease", {"common_name": "Strep Throat"})
    kg.add_entity("COVID-19", "Disease")
    kg.add_entity("Pneumonia", "Disease")
    kg.add_entity("Diabetes_Type_2", "Disease")

    kg.add_entity("Influenza_Virus", "Pathogen")
    kg.add_entity("Rhinovirus", "Pathogen")
    kg.add_entity("Streptococcus_pyogenes", "Pathogen")
    
    kg.add_entity("Oseltamivir", "Drug", {"class": "Antiviral"})
    kg.add_entity("Amoxicillin", "Drug", {"class": "Antibiotic"})
    kg.add_entity("Insulin", "Drug")
    kg.add_entity("Supportive_Care", "Treatment")
    kg.add_entity("Rest", "Treatment")
    kg.add_entity("Fluids", "Treatment")

    kg.add_relationship("Patient_Alice", "HAS_SYMPTOM", "Fever")
    kg.add_relationship("Patient_Alice", "HAS_SYMPTOM", "Cough")
    kg.add_relationship("Patient_Alice", "HAS_SYMPTOM", "Fatigue")

    kg.add_relationship("Patient_Bob", "HAS_DISEASE", "Diabetes_Type_2")
    kg.add_relationship("Patient_Bob", "HAS_SYMPTOM", "Fatigue")

    kg.add_relationship("Influenza", "HAS_SYMPTOM", "Fever")
    kg.add_relationship("Influenza", "HAS_SYMPTOM", "Cough")
    kg.add_relationship("Influenza", "HAS_SYMPTOM", "Muscle_Aches")
    kg.add_relationship("Influenza", "HAS_SYMPTOM", "Sore_Throat")
    kg.add_relationship("Influenza", "CAUSED_BY", "Influenza_Virus")
    kg.add_relationship("Influenza", "HAS_TREATMENT", "Oseltamivir")
    kg.add_relationship("Influenza", "HAS_TREATMENT", "Supportive_Care")

    kg.add_relationship("Common_Cold", "HAS_SYMPTOM", "Cough")
    kg.add_relationship("Common_Cold", "HAS_SYMPTOM", "Sore_Throat")
    kg.add_relationship("Common_Cold", "HAS_SYMPTOM", "Headache")
    kg.add_relationship("Common_Cold", "CAUSED_BY", "Rhinovirus")
    kg.add_relationship("Common_Cold", "HAS_TREATMENT", "Supportive_Care")

    kg.add_relationship("Streptococcal_Pharyngitis", "HAS_SYMPTOM", "Sore_Throat")
    kg.add_relationship("Streptococcal_Pharyngitis", "HAS_SYMPTOM", "Fever")
    kg.add_relationship("Streptococcal_Pharyngitis", "CAUSED_BY", "Streptococcus_pyogenes")
    kg.add_relationship("Streptococcal_Pharyngitis", "HAS_TREATMENT", "Amoxicillin")

    kg.add_relationship("COVID-19", "HAS_SYMPTOM", "Fever")
    kg.add_relationship("COVID-19", "HAS_SYMPTOM", "Cough")
    kg.add_relationship("COVID-19", "HAS_SYMPTOM", "Fatigue")
    
    kg.add_relationship("Pneumonia", "HAS_SYMPTOM", "Cough")
    kg.add_relationship("Pneumonia", "HAS_SYMPTOM", "Fever")
    kg.add_relationship("Pneumonia", "IS_A", "Respiratory_Illness")

    kg.add_relationship("Diabetes_Type_2", "HAS_SYMPTOM", "Fatigue")
    kg.add_relationship("Diabetes_Type_2", "HAS_TREATMENT", "Insulin")


populate_mock_kg(kg_instance)

def run_medigraph_ai(patient_query, reasoning_mode="Iterative Prompting"):
    extracted_entities = extract_topic_entities(llm_service, patient_query)
    parsed_query = semantic_parse_query(llm_service, patient_query)

    response_text = f"Patient Query: {patient_query}\n"
    response_text += f"Extracted Entities: {', '.join(extracted_entities)}\n"
    response_text += f"Parsed Query Type: {parsed_query.get('query_type', 'N/A')}\n\n"

    if reasoning_mode == "Iterative Prompting":
        initial_facts = []
        if extracted_entities:
            for entity in extracted_entities:
                initial_facts.extend(kg_instance.query_facts(entity))
        
        diagnosis, conversation = agent.iterative_prompting(patient_query, initial_facts)
        response_text += "--- Iterative Prompting Dialogue ---\n"
        for msg in conversation:
            response_text += f"{msg['role'].title()}: {msg['content']}\n"
        response_text += "\n--- Final Diagnosis/Treatment ---\n"
        response_text += diagnosis
    elif reasoning_mode == "LLM-Guided Beam Search":
        if not extracted_entities:
            return "Please provide a query with identifiable entities for beam search."
        
        goal = f"Diagnose disease for symptoms: {', '.join(extracted_entities)}" if parsed_query.get('query_type') == 'DIAGNOSIS' else patient_query
        
        best_paths = agent.llm_guided_beam_search(extracted_entities, goal)
        response_text += "--- LLM-Guided Beam Search Results ---\n"
        if best_paths:
            for i, (entities, path, score) in enumerate(best_paths):
                response_text += f"Path {i+1} (Score: {score:.2f}):\n"
                response_text += f"  Goal entities: {entities}\n"
                response_text += f"  Path: {agent.format_triple_path(path)}\n"
                explanation_prompt = (
                    f"Explain the significance of this knowledge graph path in relation to the patient's query: '{patient_query}'. "
                    f"Path: {agent.format_triple_path(path)}\n"
                    f"Entities reached: {entities}"
                )
                explanation = llm_service.generate(explanation_prompt)
                response_text += f"  Explanation: {explanation}\n\n"
        else:
            response_text += "No relevant paths found.\n"
    
    return response_text


iface = gr.Interface(
    fn=run_medigraph_ai,
    inputs=[
        gr.Textbox(label="Patient Query / Symptoms", placeholder="e.g., Patient Alice has fever, cough, and fatigue."),
        gr.Radio(choices=["Iterative Prompting", "LLM-Guided Beam Search"], label="Reasoning Mode", value="Iterative Prompting")
    ],
    outputs=gr.Textbox(label="MediGraph AI Analysis", lines=20),
    title="MediGraph AI - Clinical Diagnostic Assistant",
    description="Leveraging LLMs and Knowledge Graphs for clinical diagnostics. Choose a reasoning mode for analysis."
)

if __name__ == "__main__":
    iface.launch()