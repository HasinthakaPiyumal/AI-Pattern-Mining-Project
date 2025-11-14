import networkx as nx
from typing import Dict, Any, List

class ClinicalTrialKGAR:
    """
    Implements the Knowledge Graph Agentic Reasoning (KGAR) pattern for
    Clinical Trial Eligibility Screening.
    """

    def __init__(self, knowledge_graph: nx.MultiDiGraph, llm_agent_model_id: str = "simulated-llm-agent"):
        """
        Initializes the KGAR assistant with a knowledge graph and a simulated LLM agent.

        Args:
            knowledge_graph: A NetworkX MultiDiGraph representing the KG
                             (patient data, trial protocols, medical ontologies).
            llm_agent_model_id: Identifier for the LLM agent model (simulated).
        """
        self.kg = knowledge_graph
        self.llm_agent_model_id = llm_agent_model_id
        print(f"Initialized ClinicalTrialKGAR with KG containing {len(self.kg.nodes)} nodes "
              f"and using simulated LLM: {self.llm_agent_model_id}")

    def _simulate_llm_query_generation(self, patient_profile: Dict[str, Any], trial_criteria: Dict[str, Any]) -> str:
        """
        Simulates an LLM agent generating a SPARQL-like query based on patient and trial info.
        In a real scenario, this would involve a prompt to the LLM and parsing its response.
        """
        print("LLM Agent: Generating KG query...")
        # Example: Simple query for patient's conditions and trial's exclusion criteria
        patient_name = patient_profile.get("name", "patient")
        trial_id = trial_criteria.get("trial_id", "trial")
        query = (
            f"SELECT ?entity ?relation ?value WHERE {{\n"
            f"  ?patient a :Patient ; :name \"{patient_name}\" ; ?relation ?value .\n"
            f"  ?trial a :ClinicalTrial ; :id \"{trial_id}\" ; ?rel_crit ?crit_value .\n"
            f"  FILTER ((?relation = :hasCondition || ?relation = :hasMedication) && (?rel_crit = :excludesCondition || ?rel_crit = :includesCondition))\n"
            f"}}"
        )
        return query

    def _execute_kg_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Simulates executing a query against the NetworkX Knowledge Graph.
        In a real system, this would interact with a SPARQL endpoint or graph database API.
        """
        print(f"Executing simulated KG query: {query[:100]}...")
        results = []
        # This is a very simplistic simulation. A real SPARQL engine would be far more complex.
        # For demonstration, we'll try to find nodes and edges that might match parts of the query.
        # We're looking for patient conditions and trial exclusion criteria.

        # Simulate finding patient conditions
        patient_name_in_query = next((self._extract_literal(query, ':name \"', '\"') for _ in [None] if ':name \"' in query), None)
        if patient_name_in_query:
            for node, data in self.kg.nodes(data=True):
                if data.get("type") == "Patient" and data.get("name") == patient_name_in_query:
                    for _, target, edge_data in self.kg.edges(node, data=True):
                        if edge_data.get("type") in ["hasCondition", "hasMedication"]:
                            results.append({
                                "subject": node,
                                "predicate": edge_data["type"],
                                "object": self.kg.nodes[target].get("name", target)
                            })

        # Simulate finding trial criteria (e.g., exclusions)
        trial_id_in_query = next((self._extract_literal(query, ':id \"', '\"') for _ in [None] if ':id \"' in query), None)
        if trial_id_in_query:
            for node, data in self.kg.nodes(data=True):
                if data.get("type") == "ClinicalTrial" and data.get("id") == trial_id_in_query:
                    for _, target, edge_data in self.kg.edges(node, data=True):
                        if edge_data.get("type") in ["excludesCondition", "includesCondition"]:
                            results.append({
                                "subject": node,
                                "predicate": edge_data["type"],
                                "object": self.kg.nodes[target].get("name", target)
                            })

        return results
    
    def _extract_literal(self, text, start_tag, end_tag):
        """Helper to extract a literal string between tags from a simulated query."""
        start_index = text.find(start_tag)
        if start_index == -1: return None
        start_index += len(start_tag)
        end_index = text.find(end_tag, start_index)
        if end_index == -1: return None
        return text[start_index:end_index]


    def _prune_knowledge(self, retrieved_knowledge: List[Dict[str, Any]], eligibility_question: str) -> List[Dict[str, Any]]:
        """
        Simulates pruning irrelevant knowledge based on semantic relevance to the eligibility question.
        In a real scenario, this could involve embedding-based similarity or LLM-driven filtering.
        """
        print("LLM Agent: Pruning retrieved knowledge...")
        # For simplicity, we'll keep everything for now or apply a very basic filter.
        # A more advanced system would use LLM to decide relevance.
        relevant_knowledge = []
        keywords = eligibility_question.lower().split()
        for fact in retrieved_knowledge:
            fact_str = f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}".lower()
            if any(k in fact_str for k in keywords):
                relevant_knowledge.append(fact)
            elif not keywords: # if no specific keywords, consider all retrieved relevant
                relevant_knowledge.append(fact)

        if not relevant_knowledge and retrieved_knowledge: # Fallback if pruning was too aggressive
            print("Pruning resulted in empty set, returning original retrieved knowledge.")
            return retrieved_knowledge

        return relevant_knowledge

    def _reason_eligibility(self, patient_facts: List[Dict[str, Any]], trial_criteria_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulates the LLM agent performing faithful reasoning based on structured facts.
        """
        print("LLM Agent: Performing eligibility reasoning...")
        is_eligible = True
        reasons = []
        
        patient_conditions = [fact['object'] for fact in patient_facts if fact['predicate'] == 'hasCondition']
        trial_exclusions = [fact['object'] for fact in trial_criteria_facts if fact['predicate'] == 'excludesCondition']
        trial_inclusions = [fact['object'] for fact in trial_criteria_facts if fact['predicate'] == 'includesCondition']

        # Check for exclusions
        for exclusion in trial_exclusions:
            if exclusion in patient_conditions:
                is_eligible = False
                reasons.append(f"Patient has condition '{exclusion}' which is an exclusion criterion.")
                break # One exclusion is enough to disqualify
        
        # Check for inclusions if not already excluded
        if is_eligible and trial_inclusions:
            all_inclusions_met = True
            for inclusion in trial_inclusions:
                if inclusion not in patient_conditions:
                    all_inclusions_met = False
                    is_eligible = False
                    reasons.append(f"Patient does not have required condition '{inclusion}'.")
                    break
            if not all_inclusions_met:
                pass # Reasons already added
        elif is_eligible and not trial_inclusions: # If no specific inclusions, and no exclusions, still eligible
            reasons.append("No specific inclusion criteria specified or all met, and no exclusion criteria violated.")

        if not reasons and is_eligible: # Default reason if nothing specific was found but still eligible
            reasons.append("Based on the available information, the patient appears to meet the trial's general eligibility requirements.")
        elif not reasons and not is_eligible: # Fallback for exclusion if no specific reason found, but somehow not eligible
            reasons.append("Eligibility could not be determined or patient is ineligible based on complex interactions.")

        return {"eligible": is_eligible, "reasons": reasons}

    def _generate_explanation(self, decision: Dict[str, Any]) -> str:
        """
        Simulates the LLM agent generating an interpretable explanation for the decision.
        """
        print("LLM Agent: Generating explanation...")
        status = "eligible" if decision["eligible"] else "not eligible"
        explanation = f"The patient is {status} for the clinical trial. Reasons:\n"
        for reason in decision["reasons"]:
            explanation += f"- {reason}\n"
        return explanation

    def screen_patient_for_trial(self, patient_profile: Dict[str, Any], trial_protocol: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates the KGAR process for screening a patient against a clinical trial.

        Args:
            patient_profile: Dictionary containing patient information (e.g., {"name": "Alice Smith", "conditions": [...], ...})
            trial_protocol: Dictionary containing trial criteria (e.g., {"trial_id": "T123", "inclusion": [...], "exclusion": [...]})

        Returns:
            A dictionary containing the eligibility decision and an explanation.
        """
        print(f"--- Screening Patient '{patient_profile.get('name', 'N/A')}' for Trial '{trial_protocol.get('trial_id', 'N/A')}' ---")
        
        # Step 1: LLM Agent generates grounded plan/query
        eligibility_question = (
            f"Determine if patient {patient_profile.get('name', 'N/A')} is eligible for "
            f"clinical trial {trial_protocol.get('trial_id', 'N/A')} based on conditions and exclusions."
        )
        kg_query = self._simulate_llm_query_generation(patient_profile, trial_protocol)

        # Step 2: Iterative graph exploration and retrieval
        retrieved_knowledge = self._execute_kg_query(kg_query)
        print(f"Retrieved {len(retrieved_knowledge)} facts from KG.")

        # Step 3: Pruning entities and relations based on semantic relevance
        pruned_knowledge = self._prune_knowledge(retrieved_knowledge, eligibility_question)
        print(f"Pruned to {len(pruned_knowledge)} relevant facts.")

        # Separate patient facts from trial criteria facts for reasoning
        patient_facts = [fact for fact in pruned_knowledge if 'patient' in str(fact.get('subject', '')).lower() or 'patient' in str(fact.get('object', '')).lower()]
        trial_criteria_facts = [fact for fact in pruned_knowledge if 'trial' in str(fact.get('subject', '')).lower() or 'trial' in str(fact.get('object', '')).lower()]
        
        # For this simple example, we'll enrich patient and trial facts directly from inputs as well
        # to ensure critical info is available for reasoning, supplementing KG retrieval simulation.
        # In a real system, the KG would be the authoritative source for these.
        for cond in patient_profile.get('conditions', []):
            patient_facts.append({"subject": patient_profile["name"], "predicate": "hasCondition", "object": cond})
        for exc in trial_protocol.get('exclusion_criteria', []):
            trial_criteria_facts.append({"subject": trial_protocol["trial_id"], "predicate": "excludesCondition", "object": exc})
        for inc in trial_protocol.get('inclusion_criteria', []):
            trial_criteria_facts.append({"subject": trial_protocol["trial_id"], "predicate": "includesCondition", "object": inc})

        # Step 4: LLM performs faithful reasoning
        decision = self._reason_eligibility(patient_facts, trial_criteria_facts)

        # Step 5: Generate interpretable explanations
        explanation = self._generate_explanation(decision)

        return {"decision": decision, "explanation": explanation}


# Example Usage:
if __name__ == "__main__":
    # 1. Simulate Knowledge Graph Construction using NetworkX
    kg = nx.MultiDiGraph()

    # Add patient nodes and their conditions/medications
    kg.add_node("patient_alice", type="Patient", name="Alice Smith")
    kg.add_node("condition_diabetes", type="Condition", name="Diabetes Mellitus Type 2")
    kg.add_node("condition_hypertension", type="Condition", name="Hypertension")
    kg.add_node("med_metformin", type="Medication", name="Metformin")
    kg.add_edge("patient_alice", "condition_diabetes", type="hasCondition")
    kg.add_edge("patient_alice", "condition_hypertension", type="hasCondition")
    kg.add_edge("patient_alice", "med_metformin", type="hasMedication")

    kg.add_node("patient_bob", type="Patient", name="Bob Johnson")
    kg.add_node("condition_asthma", type="Condition", name="Asthma")
    kg.add_edge("patient_bob", "condition_asthma", type="hasCondition")

    # Add clinical trial nodes and their criteria
    kg.add_node("trial_T123", type="ClinicalTrial", id="T123", phase="3")
    kg.add_node("exclusion_diabetes_type2", type="Exclusion", name="Diabetes Mellitus Type 2")
    kg.add_node("inclusion_hypertension", type="Inclusion", name="Hypertension")
    kg.add_edge("trial_T123", "exclusion_diabetes_type2", type="excludesCondition")
    kg.add_edge("trial_T123", "inclusion_hypertension", type="includesCondition")

    kg.add_node("trial_T456", type="ClinicalTrial", id="T456", phase="2")
    kg.add_node("exclusion_asthma", type="Exclusion", name="Asthma")
    kg.add_edge("trial_T456", "exclusion_asthma", type="excludesCondition")

    # 2. Instantiate the KGAR assistant
    assistant = ClinicalTrialKGAR(knowledge_graph=kg)

    # 3. Define patient profiles and trial protocols
    patient_alice_profile = {
        "name": "Alice Smith",
        "age": 65,
        "conditions": ["Diabetes Mellitus Type 2", "Hypertension"],
        "medications": ["Metformin"]
    }

    patient_bob_profile = {
        "name": "Bob Johnson",
        "age": 40,
        "conditions": ["Asthma"],
        "medications": []
    }

    trial_T123_protocol = {
        "trial_id": "T123",
        "name": "Study on New Hypertension Drug",
        "inclusion_criteria": ["Hypertension"],
        "exclusion_criteria": ["Diabetes Mellitus Type 2", "Kidney Failure"]
    }

    trial_T456_protocol = {
        "trial_id": "T456",
        "name": "Asthma Treatment Efficacy Study",
        "inclusion_criteria": ["Asthma"],
        "exclusion_criteria": ["Cardiovascular Disease"]
    }
    
    trial_T789_protocol_no_exclusions = {
        "trial_id": "T789",
        "name": "General Health Study",
        "inclusion_criteria": [],
        "exclusion_criteria": []
    }


    # 4. Screen patients for trials
    print("\n--- Screening Alice for T123 ---")
    result_alice_t123 = assistant.screen_patient_for_trial(patient_alice_profile, trial_T123_protocol)
    print(result_alice_t123["explanation"])

    print("\n--- Screening Bob for T123 ---")
    result_bob_t123 = assistant.screen_patient_for_trial(patient_bob_profile, trial_T123_protocol)
    print(result_bob_t123["explanation"])

    print("\n--- Screening Bob for T456 ---")
    result_bob_t456 = assistant.screen_patient_for_trial(patient_bob_profile, trial_T456_protocol)
    print(result_bob_t456["explanation"])
    
    print("\n--- Screening Alice for T456 ---")
    result_alice_t456 = assistant.screen_patient_for_trial(patient_alice_profile, trial_T456_protocol)
    print(result_alice_t456["explanation"])
    
    print("\n--- Screening Alice for T789 (no specific criteria) ---")
    result_alice_t789 = assistant.screen_patient_for_trial(patient_alice_profile, trial_T789_protocol_no_exclusions)
    print(result_alice_t789["explanation"])

