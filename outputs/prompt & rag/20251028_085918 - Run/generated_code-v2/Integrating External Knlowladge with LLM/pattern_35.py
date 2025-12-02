import streamlit as st
import networkx as nx
import random

# 1. Specialized Prompts - Python Constants/Templates
RELATION_PRUNE_PROMPT = """Given the patient's symptoms: {symptoms}, medical history: {history}, and lab results: {lab_results}, and a list of candidate relations: {candidate_relations}. Identify and score the most relevant relations (e.g., 'has_symptom', 'associated_with_gene', 'leads_to_complication') from the candidate set. Return only the selected relations, comma-separated."""

ENTITY_PRUNE_PROMPT = """Based on the patient's symptoms: {symptoms}, medical history: {history}, lab results: {lab_results}, and the selected relations: {selected_relations}, and a list of candidate entities: {candidate_entities}. Score the contribution of these candidate entities (e.g., specific rare diseases, genes, diagnostic biomarkers). Return only the selected entities, comma-separated."""

REASONING_PROMPT = """Given the patient's data, the currently explored relations: {selected_relations}, and entities: {selected_entities}. Evaluate the sufficiency of the current reasoning paths for answering the diagnostic question. Is there enough evidence to form a diagnostic hypothesis? Respond 'SUFFICIENT' if enough, 'INSUFFICIENT' otherwise. If 'INSUFFICIENT', suggest what kind of information would be beneficial for further exploration."""

GENERATE_PROMPT = """Based on the accumulated knowledge from patient data, explored relations: {selected_relations}, and entities: {selected_entities}. Synthesize a ranked list of potential rare disease diagnoses, including supporting evidence and justifications for each diagnosis. Also, suggest potential next steps (e.g., further tests, specialist referrals)."""

# 2. Medical Knowledge Graph (KG) - NetworkX
def create_sample_kg():
    kg = nx.DiGraph()

    # Diseases
    kg.add_node("Cystic Fibrosis", type="disease")
    kg.add_node("Huntington's Disease", type="disease")
    kg.add_node("Sickle Cell Anemia", type="disease")
    kg.add_node("Marfan Syndrome", type="disease")
    kg.add_node("Amyotrophic Lateral Sclerosis (ALS)", type="disease")

    # Symptoms
    kg.add_node("Chronic Cough", type="symptom")
    kg.add_node("Difficulty Breathing", type="symptom")
    kg.add_node("Pancreatic Insufficiency", type="symptom")
    kg.add_node("Chorea", type="symptom")
    kg.add_node("Cognitive Decline", type="symptom")
    kg.add_node("Anemia", type="symptom")
    kg.add_node("Fatigue", type="symptom")
    kg.add_node("Skeletal Abnormalities", type="symptom")
    kg.add_node("Cardiovascular Issues", type="symptom")
    kg.add_node("Muscle Weakness", type="symptom")
    kg.add_node("Speech Difficulty", type="symptom")

    # Genes
    kg.add_node("CFTR gene", type="gene")
    kg.add_node("HTT gene", type="gene")
    kg.add_node("HBB gene", type="gene")
    kg.add_node("FBN1 gene", type="gene")
    kg.add_node("SOD1 gene", type="gene")

    # Biomarkers
    kg.add_node("Sweat Chloride Test", type="biomarker")
    kg.add_node("Genetic Testing (HTT)", type="biomarker")
    kg.add_node("Hemoglobin Electrophoresis", type="biomarker")
    kg.add_node("FBN1 Gene Sequencing", type="biomarker")
    kg.add_node("Nerve Conduction Study", type="biomarker")

    # Relations
    kg.add_edge("Cystic Fibrosis", "has_symptom", target="Chronic Cough")
    kg.add_edge("Cystic Fibrosis", "has_symptom", target="Difficulty Breathing")
    kg.add_edge("Cystic Fibrosis", "has_symptom", target="Pancreatic Insufficiency")
    kg.add_edge("Cystic Fibrosis", "associated_with_gene", target="CFTR gene")
    kg.add_edge("Cystic Fibrosis", "diagnostic_test", target="Sweat Chloride Test")

    kg.add_edge("Huntington's Disease", "has_symptom", target="Chorea")
    kg.add_edge("Huntington's Disease", "has_symptom", target="Cognitive Decline")
    kg.add_edge("Huntington's Disease", "associated_with_gene", target="HTT gene")
    kg.add_edge("Huntington's Disease", "diagnostic_test", target="Genetic Testing (HTT)")

    kg.add_edge("Sickle Cell Anemia", "has_symptom", target="Anemia")
    kg.add_edge("Sickle Cell Anemia", "has_symptom", target="Fatigue")
    kg.add_edge("Sickle Cell Anemia", "associated_with_gene", target="HBB gene")
    kg.add_edge("Sickle Cell Anemia", "diagnostic_test", target="Hemoglobin Electrophoresis")

    kg.add_edge("Marfan Syndrome", "has_symptom", target="Skeletal Abnormalities")
    kg.add_edge("Marfan Syndrome", "has_symptom", target="Cardiovascular Issues")
    kg.add_edge("Marfan Syndrome", "associated_with_gene", target="FBN1 gene")
    kg.add_edge("Marfan Syndrome", "diagnostic_test", target="FBN1 Gene Sequencing")

    kg.add_edge("Amyotrophic Lateral Sclerosis (ALS)", "has_symptom", target="Muscle Weakness")
    kg.add_edge("Amyotrophic Lateral Sclerosis (ALS)", "has_symptom", target="Speech Difficulty")
    kg.add_edge("Amyotrophic Lateral Sclerosis (ALS)", "associated_with_gene", target="SOD1 gene")
    kg.add_edge("Amyotrophic Lateral Sclerosis (ALS)", "diagnostic_test", target="Nerve Conduction Study")

    kg.add_edge("CFTR gene", "causes", target="Cystic Fibrosis")
    kg.add_edge("HTT gene", "causes", target="Huntington's Disease")
    kg.add_edge("HBB gene", "causes", target="Sickle Cell Anemia")
    kg.add_edge("FBN1 gene", "causes", target="Marfan Syndrome")
    kg.add_edge("SOD1 gene", "causes", target="Amyotrophic Lateral Sclerosis (ALS)")

    return kg

# 3. KG Interaction Layer - Custom Python Functions
def get_candidate_relations(kg, entities):
    candidate_relations = set()
    for entity in entities:
        if entity in kg:
            for u, v, data in kg.edges(entity, data=True):
                candidate_relations.add(data.get('target', data.get('relation', 'unknown_relation')))
            for u, v, data in kg.in_edges(entity, data=True):
                candidate_relations.add(data.get('target', data.get('relation', 'unknown_relation')))
    return list(candidate_relations)

def get_candidate_entities(kg, relations, source_entities):
    candidate_entities = set()
    for source_entity in source_entities:
        if source_entity in kg:
            for u, v, data in kg.edges(source_entity, data=True):
                if data.get('target', data.get('relation')) in relations:
                    candidate_entities.add(v)
            for u, v, data in kg.in_edges(source_entity, data=True):
                if data.get('target', data.get('relation')) in relations:
                    candidate_entities.add(u)
    return list(candidate_entities)

def get_subgraph_around_entities(kg, entities, depth=1):
    if not entities:
        return nx.DiGraph()
    nodes_to_explore = set(entities)
    subgraph_nodes = set(entities)
    for _ in range(depth):
        new_nodes = set()
        for node in nodes_to_explore:
            if node in kg:
                for neighbor in kg.neighbors(node):
                    new_nodes.add(neighbor)
                for predecessor in kg.predecessors(node):
                    new_nodes.add(predecessor)
        subgraph_nodes.update(new_nodes)
        nodes_to_explore = new_nodes
    return kg.subgraph(subgraph_nodes)

# 4. Large Language Model (LLM) - MockLLM
class MockLLM:
    def generate_response(self, prompt: str, context: dict) -> str:
        if "Identify and score relevant relations" in prompt:
            symptoms = context.get("symptoms", "").lower()
            candidate_relations = context.get("candidate_relations", [])
            selected = []
            if "cough" in symptoms or "breathing" in symptoms: selected.append("has_symptom")
            if "gene" in symptoms or "family history" in symptoms: selected.append("associated_with_gene")
            if not selected and candidate_relations: selected.append(random.choice(candidate_relations)) # Fallback
            return ",".join(list(set(selected))) if selected else "has_symptom,associated_with_gene"

        elif "Score the contribution of candidate entities" in prompt:
            symptoms = context.get("symptoms", "").lower()
            candidate_entities = context.get("candidate_entities", [])
            selected = []
            for entity in candidate_entities:
                if any(s in entity.lower() for s in symptoms.split()) or "disease" in entity.lower() or "gene" in entity.lower() or "test" in entity.lower():
                    selected.append(entity)
            if not selected and candidate_entities: selected.append(random.choice(candidate_entities)) # Fallback
            return ",".join(list(set(selected))) if selected else "Cystic Fibrosis,CFTR gene"

        elif "Evaluate the sufficiency of the current reasoning paths" in prompt:
            selected_entities = context.get("selected_entities", [])
            has_disease_entity = any("disease" in entity.lower() for entity in selected_entities)
            if has_disease_entity and len(selected_entities) > 2: # Simple heuristic for sufficiency
                return "SUFFICIENT: Enough evidence for a diagnostic hypothesis."
            else:
                return "INSUFFICIENT: Need to explore more entities and relations related to symptoms and genes."

        elif "Synthesize a ranked list of potential rare disease diagnoses" in prompt:
            selected_entities = context.get("selected_entities", [])
            selected_relations = context.get("selected_relations", [])
            patient_data = context.get("patient_data", {})

            diagnosis = f"Potential Diagnosis for Patient with symptoms: {patient_data.get('symptoms')}:\n"
            if "Cystic Fibrosis" in selected_entities and "CFTR gene" in selected_entities:
                diagnosis += "1. Cystic Fibrosis: Supported by symptoms like chronic cough, difficulty breathing, and association with CFTR gene. Recommend Sweat Chloride Test.\n"
            if "Huntington's Disease" in selected_entities and "HTT gene" in selected_entities:
                diagnosis += "2. Huntington's Disease: Supported by symptoms like chorea, cognitive decline, and association with HTT gene. Recommend Genetic Testing (HTT).\n"
            if "Sickle Cell Anemia" in selected_entities and "HBB gene" in selected_entities:
                diagnosis += "3. Sickle Cell Anemia: Supported by symptoms like anemia, fatigue, and association with HBB gene. Recommend Hemoglobin Electrophoresis.\n"
            if not diagnosis.endswith("\n"):
                diagnosis += "No specific rare disease identified with high confidence based on current information. Further investigation needed.\n"

            diagnosis += f"\nExplored Relations: {', '.join(selected_relations)}"
            diagnosis += f"\nExplored Entities: {', '.join(selected_entities)}"
            diagnosis += "\nNext Steps: Consider genetic counseling, specialized imaging, or consult a rare disease expert."
            return diagnosis

        return "" # Default empty response

# 5. Prompt Orchestration Module - Python Class (`PromptOrchestrator`)
class PromptOrchestrator:
    def __init__(self, llm, kg):
        self.llm = llm
        self.kg = kg

    def run_iterative_diagnosis(self, patient_data: dict) -> dict:
        st.subheader("Diagnostic Process")
        reasoning_steps = []
        exploration_state = {
            "selected_relations": [],
            "selected_entities": [],
            "reasoning_paths": []
        }

        current_patient_entities = patient_data["symptoms"].split(",") + patient_data["history"].split(",")
        current_patient_entities = [e.strip() for e in current_patient_entities if e.strip()]
        
        max_iterations = 5 # Prevent infinite loops
        for i in range(max_iterations):
            st.markdown(f"### Iteration {i+1}")
            step_output = {}

            # Relation Prune Step
            candidate_relations = get_candidate_relations(self.kg, current_patient_entities)
            relation_prompt = RELATION_PRUNE_PROMPT.format(
                symptoms=patient_data["symptoms"],
                history=patient_data["history"],
                lab_results=patient_data["lab_results"],
                candidate_relations=", ".join(candidate_relations)
            )
            st.markdown("**Relation Prune Prompt:**")
            st.text_area(label=f"Relation Prompt {i+1}", value=relation_prompt, height=150, disabled=True, key=f"rp_{i}")

            llm_relation_response = self.llm.generate_response(relation_prompt, {
                "symptoms": patient_data["symptoms"],
                "history": patient_data["history"],
                "lab_results": patient_data["lab_results"],
                "candidate_relations": candidate_relations
            })
            exploration_state["selected_relations"] = [r.strip() for r in llm_relation_response.split(",") if r.strip()]
            step_output["selected_relations"] = exploration_state["selected_relations"]
            st.info(f"LLM Selected Relations: {', '.join(exploration_state['selected_relations'])}")

            # Entity Prune Step
            initial_entities_for_kg = current_patient_entities + exploration_state["selected_relations"]
            candidate_entities = get_candidate_entities(self.kg, exploration_state["selected_relations"], current_patient_entities)
            
            # Augment candidate entities from the KG based on current patient entities
            related_from_kg = set()
            for p_entity in current_patient_entities:
                if p_entity in self.kg:
                    for neighbor in self.kg.neighbors(p_entity):
                        related_from_kg.add(neighbor)
                    for predecessor in self.kg.predecessors(p_entity):
                        related_from_kg.add(predecessor)
            candidate_entities.extend(list(related_from_kg))
            candidate_entities = list(set(candidate_entities))

            entity_prompt = ENTITY_PRUNE_PROMPT.format(
                symptoms=patient_data["symptoms"],
                history=patient_data["history"],
                lab_results=patient_data["lab_results"],
                selected_relations=", ".join(exploration_state["selected_relations"]),
                candidate_entities=", ".join(candidate_entities)
            )
            st.markdown("**Entity Prune Prompt:**")
            st.text_area(label=f"Entity Prompt {i+1}", value=entity_prompt, height=150, disabled=True, key=f"ep_{i}")

            llm_entity_response = self.llm.generate_response(entity_prompt, {
                "symptoms": patient_data["symptoms"],
                "history": patient_data["history"],
                "lab_results": patient_data["lab_results"],
                "selected_relations": exploration_state["selected_relations"],
                "candidate_entities": candidate_entities
            })
            exploration_state["selected_entities"] = [e.strip() for e in llm_entity_response.split(",") if e.strip()]
            step_output["selected_entities"] = exploration_state["selected_entities"]
            st.info(f"LLM Selected Entities: {', '.join(exploration_state['selected_entities'])}")

            # Reasoning Step
            reasoning_prompt = REASONING_PROMPT.format(
                selected_relations=", ".join(exploration_state["selected_relations"]),
                selected_entities=", ".join(exploration_state["selected_entities"])
            )
            st.markdown("**Reasoning Prompt:**")
            st.text_area(label=f"Reasoning Prompt {i+1}", value=reasoning_prompt, height=150, disabled=True, key=f"resp_{i}")

            llm_reasoning_response = self.llm.generate_response(reasoning_prompt, {
                "selected_relations": exploration_state["selected_relations"],
                "selected_entities": exploration_state["selected_entities"]
            })
            step_output["reasoning_response"] = llm_reasoning_response
            st.info(f"LLM Reasoning: {llm_reasoning_response}")
            reasoning_steps.append(step_output)

            if "SUFFICIENT" in llm_reasoning_response.upper():
                st.success("LLM determined reasoning is sufficient. Proceeding to final generation.")
                break
            else:
                st.warning("LLM determined reasoning is insufficient. Expanding exploration.")
                # Dynamic KG Exploration: For simplicity, we just allow the loop to continue with refined entities/relations.
                # In a real system, this might involve deeper KG queries or different prompt strategies.
                current_patient_entities.extend(exploration_state["selected_entities"])
                current_patient_entities = list(set(current_patient_entities))

        # Generate Prompt (Final Output)
        st.markdown("### Final Diagnosis Generation")
        generate_prompt = GENERATE_PROMPT.format(
            selected_relations=", ".join(exploration_state["selected_relations"]),
            selected_entities=", ".join(exploration_state["selected_entities"]),
            patient_data=patient_data
        )
        st.markdown("**Generate Prompt:**")
        st.text_area(label="Generate Prompt", value=generate_prompt, height=200, disabled=True, key="gp")

        final_diagnosis = self.llm.generate_response(generate_prompt, {
            "selected_relations": exploration_state["selected_relations"],
            "selected_entities": exploration_state["selected_entities"],
            "patient_data": patient_data
        })

        return {"final_diagnosis": final_diagnosis, "reasoning_steps": reasoning_steps}

# 6. Streamlit User Interface
st.title("🧠 AI-Powered Rare Disease Diagnosis Assistant")
st.markdown("Enter patient details to get potential rare disease diagnoses based on iterative LLM-guided KG exploration.")

with st.sidebar:
    st.header("Patient Information")
    patient_symptoms = st.text_area(
        "Symptoms (comma-separated):",
        "chronic cough, difficulty breathing, pancreatic insufficiency"
    )
    patient_history = st.text_area(
        "Medical History (comma-separated):",
        "family history of lung issues, recurrent infections"
    )
    patient_lab_results = st.text_area(
        "Lab Results (comma-separated):",
        "elevated sweat chloride, normal genetic panel"
    )

    if st.button("Diagnose Rare Disease"):
        if patient_symptoms and patient_history and patient_lab_results:
            patient_data = {
                "symptoms": patient_symptoms,
                "history": patient_history,
                "lab_results": patient_lab_results,
            }
            
            kg = create_sample_kg()
            llm = MockLLM()
            orchestrator = PromptOrchestrator(llm, kg)

            with st.spinner("Running iterative diagnosis..."):
                result = orchestrator.run_iterative_diagnosis(patient_data)

            st.subheader("Final Diagnostic Output")
            st.write(result["final_diagnosis"])
        else:
            st.error("Please fill in all patient information fields.")

